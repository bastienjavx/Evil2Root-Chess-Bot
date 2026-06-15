"""Federated averaging (local SGD) pour PLUSIEURS trainers principaux.

Permet à plusieurs machines GPU réparties sur Internet d'entraîner LE MÊME modèle
ensemble (data-parallel asynchrone, façon FedAvg), tout en gardant le coordinateur
Railway **sans torch** : la moyenne des poids se fait ICI, côté trainers.

Cycle d'un round R (base = version globale v), cf. CLUSTER.md :
  1. télécharger le modèle global v (et l'écrire dans latest.pt local pour le hot-reload) ;
  2. entraîner `local_steps` pas de gradient sur le replay buffer (alimenté par la
     synchro des parties du trainer) ;
  3. POST /cluster/trainer/contribute : envoyer ses poids (blob opaque) + num_samples ;
  4. attendre la fermeture du round, puis :
     - le finalizer désigné (1er contributeur) moyenne toutes les contributions
       (pondérées par num_samples) et publie -> version v+1, round R+1 ;
     - les autres attendent que la version monte, puis re-téléchargent (étape 1).
  La correction repose sur la garde idempotente du publish côté serveur (un seul
  publish accepté par round), pas sur un averaging déterministe partagé : si le
  finalizer meurt, après le délai de grâce un autre contributeur reprend la main.

Réutilise tel quel l'encodage et l'ingestion d'`online.py`, le `split_batch` du
dataset, les pertes, et les utilitaires de checkpoint.
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
import time
from collections import deque
from pathlib import Path

import requests
import torch

from ..data.samples import iter_samples_full
from ..model import build_model, build_model_from_checkpoint
from ..train.dataset import split_batch
from ..train.losses import moves_left_loss, policy_loss, value_loss
from ..train.online import _encode_batch, _ingest_new_shards
from ..utils import (amp_enabled, autocast_ctx, load_checkpoint, load_model_state,
                     resolve_amp_dtype, resolve_device)
from . import protocol as P


# --- Moyenne des poids -------------------------------------------------------

def average_state_dicts(payloads: list[dict], weights: list[float]) -> dict:
    """Moyenne PONDÉRÉE (par num_samples) des `model_state` de plusieurs trainers.

    Les tenseurs flottants sont moyennés en fp32 puis recastés ; les buffers entiers
    (ex. BatchNorm `num_batches_tracked`) ne se moyennent pas -> on copie ceux du
    contributeur ayant vu le plus de samples. Tous les contributeurs partagent la même
    architecture (même base_version), donc les mêmes clés."""
    if not payloads:
        raise ValueError("aucune contribution à moyenner")
    total = float(sum(weights)) or 1.0
    dom = max(range(len(payloads)), key=lambda i: weights[i])  # contributeur dominant
    ref = payloads[0]["model_state"]
    out: dict = {}
    for k, ref_t in ref.items():
        if torch.is_floating_point(ref_t):
            acc = torch.zeros(ref_t.shape, dtype=torch.float32)
            for p, w in zip(payloads, weights):
                acc += p["model_state"][k].to(torch.float32) * (w / total)
            out[k] = acc.to(ref_t.dtype)
        else:
            out[k] = payloads[dom]["model_state"][k].clone()
    step = max(int(p.get("step", 0)) for p in payloads)
    return {"model_state": out, "model_cfg": payloads[0].get("model_cfg", {}), "step": step}


# --- Sérialisation / réseau --------------------------------------------------

def _serialize(payload: dict) -> bytes:
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        return Path(tmp).read_bytes()
    finally:
        os.unlink(tmp)


def _load_blob(content: bytes, map_location: str = "cpu") -> dict:
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        Path(tmp).write_bytes(content)
        return load_checkpoint(tmp, map_location)
    finally:
        os.unlink(tmp)


def _model_info(server: str) -> P.ModelInfo:
    r = requests.get(server + P.EP_MODEL_CURRENT, timeout=30)
    r.raise_for_status()
    return P.ModelInfo.from_dict(r.json())


def _publish(server: str, token: str, payload: dict, round_no: int) -> requests.Response:
    blob = _serialize(payload)
    return requests.post(
        server + P.EP_MODEL_PUBLISH,
        headers={"Authorization": f"Bearer {token}"},
        files={"model": ("weights.pt", blob, "application/octet-stream")},
        data={"step": int(payload.get("step", 0)), "round": round_no}, timeout=300)


def _fetch_global(server: str, latest: Path, cfg: dict, device: str):
    """Télécharge le modèle global courant, le copie dans latest.pt (hot-reload local),
    et renvoie (ModelInfo, checkpoint|None). None si rien n'est encore publié."""
    info = _model_info(server)
    if not info.has_model:
        return info, None
    r = requests.get(server + P.EP_MODEL_DOWNLOAD, timeout=300)
    r.raise_for_status()
    if info.sha256 and hashlib.sha256(r.content).hexdigest() != info.sha256:
        raise RuntimeError("sha256 du modèle global ne correspond pas")
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest.with_suffix(".pt.fedtmp")
    tmp.write_bytes(r.content)
    os.replace(tmp, latest)              # ce process est le seul écrivain de latest.pt
    return info, load_checkpoint(latest, device)


# --- Entraînement local d'un round -------------------------------------------

def _train_local(model, opt, buf, buffer_dir, processed, steps, *, device, batch_size,
                 min_buffer, input_features, mask_policy, with_ml, mlw, vlw, grad_clip,
                 label_smooth, scaler, use_amp, amp_dtype, trainer_id):
    done = 0
    samples_seen = 0
    waited = 0.0
    while done < steps:
        _ingest_new_shards(buffer_dir, processed, buf)
        if len(buf) < min_buffer:
            if waited >= 120.0:           # toujours pas de données : on s'arrête là
                break
            time.sleep(1.0)
            waited += 1.0
            continue
        waited = 0.0
        rows = random.sample(buf, min(batch_size, len(buf)))
        batch = _encode_batch(rows, device, input_features, mask_policy, with_ml)
        (planes, target_policy, legal_mask, val,
         target_ml, ml_mask) = split_batch(batch, mask_policy, with_ml)
        if device == "cuda":
            planes = planes.contiguous(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, use_amp, amp_dtype):
            logits, value, moves_left = model(planes)
            loss_p = policy_loss(logits, target_policy, label_smooth, legal_mask)
            loss_v = value_loss(value, val)
            loss = loss_p + vlw * loss_v
            if mlw and moves_left is not None:
                loss = loss + mlw * moves_left_loss(moves_left, target_ml, ml_mask)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        done += 1
        samples_seen += len(rows)
        if done % 20 == 0:
            print(f"[fedavg:{trainer_id}] step {done}/{steps} | loss {loss.item():.4f} "
                  f"| buffer {len(buf)}", flush=True)
    return done, samples_seen


# --- Finalisation d'un round (moyenne + publish) -----------------------------

def _finalize_round(server: str, token: str, round_no: int) -> bool:
    """Télécharge toutes les contributions du round, les moyenne et publie. Renvoie
    True si le round est finalisé (par nous OU par un autre : 409 = déjà fait)."""
    r = requests.get(server + P.EP_CONTRIBUTIONS, params={"round": round_no}, timeout=60)
    r.raise_for_status()
    items = r.json().get("contributions", [])
    payloads, weights = [], []
    for it in items:
        d = requests.get(server + P.EP_CONTRIBUTIONS,
                         params={"round": round_no, "download": it["trainer_id"]},
                         timeout=300)
        if d.status_code != 200:
            continue
        payloads.append(_load_blob(d.content, "cpu"))
        weights.append(max(1, int(it.get("num_samples", 1))))
    if not payloads:
        return False
    avg = average_state_dicts(payloads, weights)
    resp = _publish(server, token, avg, round_no)
    if resp.status_code == 200:
        info = resp.json()
        print(f"[fedavg] round {round_no} finalisé -> modèle v{info['version']} "
              f"(moyenne de {len(payloads)} trainer(s))", flush=True)
        return True
    if resp.status_code == 409:           # un autre trainer a déjà finalisé : OK
        return True
    print(f"[fedavg] publish refusé : {resp.status_code} {resp.text}", flush=True)
    return False


def _wait_and_finalize(server: str, token: str, round_no: int, base_version: int,
                       is_finalizer: bool, stop, poll_sec: float) -> None:
    """Bloque jusqu'à ce que le round soit finalisé (par nous ou un autre). Le finalizer
    désigné agit dès la fermeture ; les autres ne reprennent la main qu'après le délai de
    grâce (can_finalize) — la garde idempotente du publish évite les doublons."""
    while not stop.is_set():
        try:
            if _model_info(server).version > base_version:
                return                    # quelqu'un a publié la moyenne -> on reprendra
            rs = P.RoundInfo.from_dict(
                requests.get(server + P.EP_ROUND, timeout=30).json())
            if rs.round != round_no:      # round déjà avancé : publish imminent/effectué
                return
            ready = rs.closed if is_finalizer else rs.can_finalize
            if ready and _finalize_round(server, token, round_no):
                return
        except requests.RequestException as exc:
            print(f"[fedavg] attente round : {exc}", flush=True)
        stop.wait(poll_sec)


# --- Boucle principale -------------------------------------------------------

def run_fedavg(server: str, token: str, cfg: dict, *, trainer_id: str, local_steps: int,
               stop, batch_size: int | None = None, seed_shards: str | None = None,
               poll_sec: float = 5.0) -> None:
    server = server.rstrip("/")
    device = resolve_device(cfg["train"].get("device", "auto"))
    ocfg = cfg["online"]
    tcfg = cfg.get("train", {})
    buffer_dir = Path(cfg["data"]["buffer_dir"])
    buffer_dir.mkdir(parents=True, exist_ok=True)
    latest = Path(cfg["paths"]["latest"])
    bs = int(batch_size or ocfg["batch_size"])
    min_buffer = int(ocfg["min_buffer"])
    lr = float(ocfg["lr"])
    wd = float(tcfg.get("weight_decay", 0.0))
    grad_clip = tcfg.get("grad_clip", 1.0)
    label_smooth = tcfg.get("label_smoothing", 0.0)
    mask_policy = bool(tcfg.get("mask_policy_loss", False))
    vlw = tcfg.get("value_loss_weight", 1.0)

    use_amp = amp_enabled(device, tcfg.get("amp", True))
    amp_dtype = resolve_amp_dtype(device, tcfg.get("amp_dtype")) if use_amp else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    buf: deque = deque(maxlen=ocfg["buffer_capacity"])
    processed: set = set()
    if seed_shards:
        for shard in sorted(Path(seed_shards).glob("*.txt.gz")):
            for row in iter_samples_full(shard):
                buf.append(row)
        print(f"[fedavg:{trainer_id}] buffer amorcé : {len(buf)} samples (seed)", flush=True)

    print(f"[fedavg:{trainer_id}] device={device} | local_steps={local_steps} "
          f"| batch={bs} | serveur={server}", flush=True)

    # Amorçage : si AUCUN modèle global n'est publié et qu'on a un latest.pt local,
    # on le publie pour donner une base commune aux autres trainers.
    if not _model_info(server).has_model and latest.exists():
        ck = load_checkpoint(latest, "cpu")
        seed_payload = {"model_state": ck.get("raw_state", ck["model_state"]),
                        "model_cfg": ck.get("model_cfg", {}), "step": int(ck.get("step", 0))}
        resp = _publish(server, token, seed_payload, round_no=-1)
        if resp.status_code == 200:
            print(f"[fedavg:{trainer_id}] modèle local publié comme base "
                  f"(v{resp.json()['version']})", flush=True)

    while not stop.is_set():
        info, ck = _fetch_global(server, latest, cfg, device)
        base_version = info.version
        if ck is not None:
            model = build_model_from_checkpoint(ck, fallback_cfg=cfg).to(device)
            load_model_state(model, ck.get("raw_state", ck["model_state"]))
            model_cfg = ck.get("model_cfg") or cfg.get("model", {})
            base_step = int(ck.get("step", 0))
        else:
            model = build_model(cfg).to(device)     # rien de publié : départ à froid
            model_cfg = cfg.get("model", {})
            base_step = 0
        if device == "cuda":
            model = model.to(memory_format=torch.channels_last)
        input_features = getattr(model, "input_features", model_cfg.get("input_features", "base"))
        with_ml = bool(getattr(model, "has_moves_left", False))
        mlw = tcfg.get("moves_left_weight", 0.3) if with_ml else 0.0
        # Optimiseur RECRÉÉ à chaque round : après une moyenne, les moments Adam de la
        # version précédente ne sont plus cohérents (local SGD standard).
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        model.train()

        done, samples_seen = _train_local(
            model, opt, buf, buffer_dir, processed, local_steps, device=device,
            batch_size=bs, min_buffer=min_buffer, input_features=input_features,
            mask_policy=mask_policy, with_ml=with_ml, mlw=mlw, vlw=vlw,
            grad_clip=grad_clip, label_smooth=label_smooth, scaler=scaler,
            use_amp=use_amp, amp_dtype=amp_dtype, trainer_id=trainer_id)

        if done == 0:
            print(f"[fedavg:{trainer_id}] pas assez de données ce round ; attente…",
                  flush=True)
            stop.wait(poll_sec)
            continue

        payload = {"model_state": model.state_dict(), "model_cfg": model_cfg,
                   "step": base_step + done}
        resp = requests.post(
            server + P.EP_CONTRIBUTE, headers={"Authorization": f"Bearer {token}"},
            files={"weights": ("w.pt", _serialize(payload), "application/octet-stream")},
            data={"trainer_id": trainer_id, "base_version": base_version,
                  "num_samples": samples_seen, "step": base_step + done}, timeout=300)
        if resp.status_code == 409:
            # base_version périmée : un round a déjà avancé pendant qu'on entraînait.
            print(f"[fedavg:{trainer_id}] base périmée, resynchro.", flush=True)
            continue
        if resp.status_code != 200:
            print(f"[fedavg:{trainer_id}] contribute refusé : {resp.status_code} "
                  f"{resp.text}", flush=True)
            stop.wait(poll_sec)
            continue
        ack = resp.json()
        print(f"[fedavg:{trainer_id}] contribué au round {ack['round']} "
              f"({samples_seen} samples, finalizer={ack['is_finalizer']})", flush=True)
        _wait_and_finalize(server, token, ack["round"], base_version,
                           ack["is_finalizer"], stop, poll_sec)
