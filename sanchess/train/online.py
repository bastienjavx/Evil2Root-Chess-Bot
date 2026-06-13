"""Entraînement continu en arrière-plan sur le replay buffer alimenté par stream.py.

Boucle : ingère les nouveaux shards du buffer -> fenêtre glissante en mémoire ->
pas de gradient réguliers (LR faible pour limiter l'oubli catastrophique) ->
écrit périodiquement `checkpoints/latest.pt` que l'UCI recharge à chaud.

Usage :  python -m sanchess.train.online [--seed-shards data/shards]
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path

import chess
import numpy as np
import torch

from ..data.samples import iter_samples_pi
from ..encoding import encode_board
from ..model import build_model, build_model_from_checkpoint
from ..utils import (load_checkpoint, load_config, load_model_state,
                     resolve_device, save_checkpoint)
from .losses import dense_policy_target, policy_loss, value_loss


def _encode_batch(rows, device):
    boards = [chess.Board(f) for f, _, _, _ in rows]
    planes = np.stack([encode_board(b) for b in boards])
    # Cible politique DENSE : distribution de visites (self-play) ou one-hot (humain).
    policy = np.stack([dense_policy_target(b, m, pi)
                       for b, (_, m, _, pi) in zip(boards, rows)])
    val = np.fromiter((v for _, _, v, _ in rows), dtype=np.float32, count=len(rows))
    return (torch.from_numpy(planes).to(device),
            torch.from_numpy(policy).to(device),
            torch.from_numpy(val).to(device))


def _ingest_new_shards(buffer_dir: Path, processed: set, buf: deque) -> int:
    n = 0
    for shard in sorted(buffer_dir.glob("*.txt.gz")):
        if shard.name in processed:
            continue
        try:
            for row in iter_samples_pi(shard):
                buf.append(row)
                n += 1
        except (OSError, EOFError):
            continue          # shard en cours d'écriture : on retentera
        processed.add(shard.name)
    return n


def run(cfg: dict, seed_shards: str | None):
    ocfg = cfg["online"]
    device = resolve_device(cfg["train"].get("device", "auto"))
    print(f"Appareil : {device}")

    latest = Path(cfg["paths"]["latest"])
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

    opt = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    model.train()

    buf: deque = deque(maxlen=ocfg["buffer_capacity"])
    processed: set = set()
    buffer_dir = Path(cfg["data"]["buffer_dir"])
    buffer_dir.mkdir(parents=True, exist_ok=True)

    if seed_shards:
        for shard in sorted(Path(seed_shards).glob("*.txt.gz")):
            for row in iter_samples_pi(shard):
                buf.append(row)
        print(f"Buffer initialisé avec {len(buf)} samples (seed).")

    batch_size = ocfg["batch_size"]
    min_buffer = ocfg["min_buffer"]
    step_every = ocfg["step_every_sec"]
    ckpt_every = ocfg["checkpoint_every_sec"]

    step = 0
    last_step_t = last_ckpt_t = last_ingest_t = 0.0
    while True:
        now = time.time()

        if now - last_ingest_t > 10:
            added = _ingest_new_shards(buffer_dir, processed, buf)
            if added:
                print(f"ingéré +{added} samples (buffer={len(buf)})")
            last_ingest_t = now

        if len(buf) >= min_buffer and now - last_step_t >= step_every:
            rows = random.sample(buf, min(batch_size, len(buf)))
            planes, target_policy, val = _encode_batch(rows, device)
            opt.zero_grad(set_to_none=True)
            logits, value = model(planes)
            loss = (policy_loss(logits, target_policy)
                    + cfg["train"]["value_loss_weight"] * value_loss(value, val))
            loss.backward()
            opt.step()
            step += 1
            last_step_t = now
            if step % 20 == 0:
                print(f"online step {step} | loss {loss.item():.4f} | buffer {len(buf)}")

        if may_overwrite and now - last_ckpt_t >= ckpt_every and step > 0:
            # Réécrit l'archi RÉELLEMENT entraînée (celle du checkpoint repris),
            # pas forcément la section `model` du config.yaml local.
            save_cfg = {**cfg, "model": save_model_cfg}
            save_checkpoint(latest, model, save_cfg, opt, step)
            print(f"  hot-reload checkpoint -> {latest} (step {step})")
            last_ckpt_t = now

        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-shards", default=None,
                    help="shards de pretrain pour amorcer le buffer (anti-oubli)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config), args.seed_shards)


if __name__ == "__main__":
    main()
