"""Entraînement continu en arrière-plan sur le replay buffer alimenté par stream.py.

Boucle : ingère les nouveaux shards du buffer -> fenêtre glissante en mémoire ->
pas de gradient réguliers (LR faible pour limiter l'oubli catastrophique) ->
écrit périodiquement `checkpoints/latest.pt` que l'UCI recharge à chaud.

Usage :  python -m sanchess.train.online [--seed-shards data/shards]
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import chess
import numpy as np
import torch

from ..data.samples import iter_samples_full
from ..encoding import encode_board, legal_policy_mask
from ..model import build_model, build_model_from_checkpoint
from ..utils import (amp_enabled, autocast_ctx, load_checkpoint, load_config,
                     load_model_state, resolve_amp_dtype, resolve_device,
                     save_checkpoint)
from .dataset import split_batch
from .losses import (dense_policy_target, policy_loss, value_loss,
                     moves_left_loss, moves_left_target)


_RESULT_RE = re.compile(
    r"\bscore=(?P<score>[0-9.]+)\b.*\belo_diff=(?P<elo>[+-]?[0-9.]+)\b"
)


def _encode_batch(rows, device, input_features: str = "base",
                  mask_policy: bool = False, moves_left: bool = False):
    # rows = (fen, move, value, pi, plies_to_end) — cf. samples.iter_samples_full.
    boards = [chess.Board(r[0]) for r in rows]
    planes = np.stack([encode_board(b, input_features) for b in boards])
    # Cible politique DENSE : distribution de visites (self-play) ou one-hot (humain).
    policy = np.stack([dense_policy_target(b, r[1], r[3])
                       for b, r in zip(boards, rows)])
    val = np.fromiter((r[2] for r in rows), dtype=np.float32, count=len(rows))
    out = [torch.from_numpy(planes).to(device),
           torch.from_numpy(policy).to(device)]
    if mask_policy:
        mask = np.stack([legal_policy_mask(b) for b in boards])
        out.append(torch.from_numpy(mask).to(device))
    out.append(torch.from_numpy(val).to(device))
    if moves_left:
        pairs = [moves_left_target(r[4] if len(r) > 4 else None) for r in rows]
        ml_t = np.fromiter((t for t, _ in pairs), dtype=np.float32, count=len(pairs))
        ml_m = np.fromiter((m for _, m in pairs), dtype=np.float32, count=len(pairs))
        out.append(torch.from_numpy(ml_t).to(device))
        out.append(torch.from_numpy(ml_m).to(device))
    return tuple(out)


def _ingest_new_shards(buffer_dir: Path, processed: set, buf: deque) -> int:
    n = 0
    for shard in sorted(buffer_dir.glob("*.txt.gz")):
        if shard.name in processed:
            continue
        try:
            for row in iter_samples_full(shard):
                buf.append(row)
                n += 1
        except (OSError, EOFError):
            continue          # shard en cours d'écriture : on retentera
        processed.add(shard.name)
    return n


def _start_promotion(candidate: Path, incumbent: Path, games: int, nodes: int,
                     rand_plies: int, max_plies: int, device: str | None,
                     config: str | None) -> subprocess.Popen:
    """Lance l'arène candidat vs latest SANS bloquer : retourne le process.

    L'éval tourne ainsi EN PARALLÈLE des pas de gradient online et du self-play
    (file d'attente d'une seule éval à la fois). On récolte son verdict plus tard
    via `_finish_promotion` quand `proc.poll()` n'est plus None.
    """
    arena = Path(__file__).resolve().parents[2] / "scripts" / "arena.py"
    cmd = [
        sys.executable, str(arena),
        "--a", str(candidate),
        "--b", str(incumbent),
        "--games", str(games),
        "--nodes", str(nodes),
        "--rand-plies", str(rand_plies),
        "--max-plies", str(max_plies),
    ]
    if device:
        cmd += ["--device", device]
    if config:
        cmd += ["--config", config]
    return subprocess.Popen(cmd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _finish_promotion(proc: subprocess.Popen) -> tuple[bool, str, float, float]:
    """Récolte un process d'arène TERMINÉ : (ok, ligne RESULT, score, elo)."""
    out, err = proc.communicate()
    line = next((ln for ln in out.splitlines() if ln.startswith("RESULT")), "")
    match = _RESULT_RE.search(line)
    if proc.returncode != 0 or not match:
        detail = (err or out).strip().splitlines()
        msg = detail[-1] if detail else f"arena exit={proc.returncode}"
        return False, f"arena invalide: {msg}", 0.0, 0.0
    score = float(match.group("score"))
    elo = float(match.group("elo"))
    return True, line, score, elo


def run(cfg: dict, seed_shards: str | None, args: argparse.Namespace):
    ocfg = cfg["online"]
    device = resolve_device(cfg["train"].get("device", "auto"))
    print(f"Appareil : {device}")

    latest = Path(cfg["paths"]["latest"])
    candidate = latest.with_name(args.promotion_candidate)
    # `model_cfg` ré-embarqué au save : par défaut celui du config.yaml local, mais
    # remplacé par l'archi du checkpoint dès qu'on en reprend un (cf. plus bas).
    save_model_cfg = cfg.get("model", {})
    # Garde-fou : tant que les poids du checkpoint n'ont pas été chargés SANS
    # divergence d'archi, on s'INTERDIT de réécrire `latest.pt`. Sinon un mismatch
    # (ex. checkpoint cloud 24x320 vs config.yaml local 256x20) écraserait
    # silencieusement un bon checkpoint par un réseau à moitié ré-initialisé.
    may_overwrite = True

    if latest.exists():
        ck = load_checkpoint(latest, device)
        # On RESPECTE l'archi enregistrée dans le checkpoint (comme le bot/UCI/web)
        # plutôt que le config.yaml local : évite le mismatch silencieux quand on
        # ramène un checkpoint cloud (24x320) sur une machine au config 256x20.
        model = build_model_from_checkpoint(ck, fallback_cfg=cfg).to(device)
        save_model_cfg = ck.get("model_cfg") or save_model_cfg
        report = load_model_state(model, ck.get("raw_state", ck["model_state"]))
        if report["mismatched"]:
            # L'archi reconstruite ne colle TOUJOURS pas aux poids : checkpoint
            # incohérent. On continue en lecture seule (pas d'écrasement) pour ne
            # rien détruire ; à l'utilisateur de régénérer un checkpoint propre.
            may_overwrite = False
            print(f"ATTENTION : {report['mismatched']} tenseurs de forme "
                  "incompatible — checkpoint NON réécrit (lecture seule) pour "
                  "éviter de corrompre les poids existants.")
        else:
            print(f"Repris depuis {latest}")
    else:
        model = build_model(cfg).to(device)
        print("Pas de checkpoint : démarrage à froid (idéalement pré-entraîner d'abord).")
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    tcfg = cfg.get("train", {})
    use_amp = amp_enabled(device, tcfg.get("amp", True))
    amp_dtype = resolve_amp_dtype(device, tcfg.get("amp_dtype")) if use_amp else None
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    grad_clip = tcfg.get("grad_clip", 1.0)
    label_smooth = tcfg.get("label_smoothing", 0.0)
    mask_policy = bool(tcfg.get("mask_policy_loss", False))
    input_features = getattr(model, "input_features", save_model_cfg.get("input_features", "base"))
    with_ml = bool(getattr(model, "has_moves_left", False))
    mlw = tcfg.get("moves_left_weight", 0.3) if with_ml else 0.0
    model.train()

    buf: deque = deque(maxlen=ocfg["buffer_capacity"])
    processed: set = set()
    buffer_dir = Path(cfg["data"]["buffer_dir"])
    buffer_dir.mkdir(parents=True, exist_ok=True)

    if seed_shards:
        for shard in sorted(Path(seed_shards).glob("*.txt.gz")):
            for row in iter_samples_full(shard):
                buf.append(row)
        print(f"Buffer initialisé avec {len(buf)} samples (seed).")

    batch_size = ocfg["batch_size"]
    min_buffer = ocfg["min_buffer"]
    step_every = ocfg["step_every_sec"]
    ckpt_every = ocfg["checkpoint_every_sec"]

    step = 0
    last_step_t = last_ingest_t = 0.0
    # On initialise last_ckpt_t à MAINTENANT : sinon (0.0) la toute première
    # itération déclenche un candidat dès le step 1 — un réseau quasi identique à
    # latest -> éval lente pour ZÉRO signal. On laisse l'entraînement avancer.
    last_ckpt_t = time.time()
    # Step du dernier candidat évalué : on n'en relance pas un avant d'avoir
    # accumulé assez de nouveaux pas (sinon on dépense l'éval — coûteuse — sur un
    # réseau à peine différent du précédent).
    last_promo_step = 0
    # Éval de promotion en cours (arène lancée en arrière-plan), ou None. Une seule
    # à la fois : tant qu'elle tourne, on CONTINUE d'entraîner et le self-play
    # continue aussi — l'éval ne bloque plus la boucle.
    pending: dict | None = None
    try:
        while True:
            now = time.time()

            if now - last_ingest_t > 10:
                added = _ingest_new_shards(buffer_dir, processed, buf)
                if added:
                    print(f"ingéré +{added} samples (buffer={len(buf)})")
                last_ingest_t = now

            if len(buf) >= min_buffer and now - last_step_t >= step_every:
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
                    loss = loss_p + cfg["train"]["value_loss_weight"] * loss_v
                    if mlw and moves_left is not None:
                        loss = loss + mlw * moves_left_loss(moves_left, target_ml, ml_mask)
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                step += 1
                last_step_t = now
                if step % 20 == 0:
                    print(f"online step {step} | policy {loss_p.item():.4f} "
                          f"| value {loss_v.item():.4f} | loss {loss.item():.4f} "
                          f"| buffer {len(buf)}")

            # Récolte une éval de promotion TERMINÉE (elle a tourné en parallèle des
            # pas de gradient ci-dessus et du self-play). Décision puis on libère.
            if pending is not None and pending["proc"].poll() is not None:
                ok, line, score, elo = _finish_promotion(pending["proc"])
                cand = pending["candidate"]
                if ok:
                    print(f"  éval promotion (candidat step {pending['step']}): {line}")
                if ok and score > args.promotion_min_score:
                    os.replace(cand, latest)
                    print(f"  PROMU -> {latest} (score={score:.4f}, elo={elo:+.1f})")
                else:
                    try:
                        cand.unlink()
                    except FileNotFoundError:
                        pass
                    reason = (f"score={score:.4f} <= {args.promotion_min_score:.4f}"
                              if ok else line)
                    print(f"  non promu: {reason}")
                pending = None

            if may_overwrite and now - last_ckpt_t >= ckpt_every and step > 0:
                # Réécrit l'archi RÉELLEMENT entraînée (celle du checkpoint repris),
                # pas forcément la section `model` du config.yaml local.
                save_cfg = {**cfg, "model": save_model_cfg}
                if args.promote_only_better and latest.exists():
                    # File d'attente d'une seule éval : si l'arène précédente tourne
                    # encore, on n'en relance pas — on continue d'entraîner. Et on
                    # exige `promotion_min_steps` nouveaux pas depuis le dernier
                    # candidat : inutile d'évaluer (éval lente) un réseau à peine
                    # bougé -> il ferait nulle et ne serait pas promu de toute façon.
                    if pending is None and step - last_promo_step >= args.promotion_min_steps:
                        save_checkpoint(candidate, model, save_cfg, opt, step)
                        print(f"  candidat checkpoint -> {candidate} (step {step})")
                        proc = _start_promotion(
                            candidate, latest, args.promotion_games,
                            args.promotion_nodes, args.promotion_rand_plies,
                            args.promotion_max_plies, args.promotion_device,
                            args.config,
                        )
                        pending = {"proc": proc, "candidate": candidate, "step": step}
                        last_promo_step = step
                        print(f"  éval promotion lancée en parallèle (arena pid {proc.pid})")
                        last_ckpt_t = now
                else:
                    save_checkpoint(latest, model, save_cfg, opt, step)
                    print(f"  hot-reload checkpoint -> {latest} (step {step})")
                    last_ckpt_t = now

            time.sleep(0.2)
    finally:
        # Ctrl-C / arrêt : ne pas laisser l'arène orpheline.
        if pending is not None and pending["proc"].poll() is None:
            pending["proc"].terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-shards", default=None,
                    help="shards de pretrain pour amorcer le buffer (anti-oubli)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="surcharge online.batch_size (ex. 128 pour tenir sur un GPU 8 Go)")
    ap.add_argument("--promote-only-better", action="store_true",
                    help="écrit latest.pt seulement si un candidat bat le latest courant")
    ap.add_argument("--promotion-candidate", default="candidate.pt",
                    help="nom du checkpoint candidat dans le dossier checkpoints/")
    ap.add_argument("--promotion-games", type=int, default=12,
                    help="nombre de parties candidat vs latest avant promotion")
    ap.add_argument("--promotion-nodes", type=int, default=80,
                    help="simulations MCTS/coup pendant l'évaluation de promotion")
    ap.add_argument("--promotion-min-score", type=float, default=0.5,
                    help="score strictement dépassé pour promouvoir le candidat")
    ap.add_argument("--promotion-min-steps", type=int, default=200,
                    help="pas de gradient minimaux entre deux candidats (évite "
                         "d'évaluer un réseau à peine différent du précédent)")
    ap.add_argument("--promotion-rand-plies", type=int, default=6)
    ap.add_argument("--promotion-max-plies", type=int, default=250)
    ap.add_argument("--promotion-device", default="cpu",
                    help="appareil pour l'arena de promotion (cpu évite la VRAM trainer)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.batch_size:
        cfg["online"]["batch_size"] = args.batch_size
    run(cfg, args.seed_shards, args)


if __name__ == "__main__":
    main()
