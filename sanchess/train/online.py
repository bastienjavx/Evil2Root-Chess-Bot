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
import torch.nn.functional as F

from ..data.samples import iter_samples
from ..encoding import encode_board, move_to_index
from ..model import build_model
from ..utils import load_checkpoint, load_config, save_checkpoint


def _encode_batch(rows, device):
    planes = np.stack([encode_board(chess.Board(f)) for f, _, _ in rows])
    idx = np.fromiter((move_to_index(chess.Move.from_uci(m), chess.Board(f).turn)
                       for f, m, _ in rows), dtype=np.int64, count=len(rows))
    val = np.fromiter((v for _, _, v in rows), dtype=np.float32, count=len(rows))
    return (torch.from_numpy(planes).to(device),
            torch.from_numpy(idx).to(device),
            torch.from_numpy(val).to(device))


def _ingest_new_shards(buffer_dir: Path, processed: set, buf: deque) -> int:
    n = 0
    for shard in sorted(buffer_dir.glob("*.txt.gz")):
        if shard.name in processed:
            continue
        try:
            for row in iter_samples(shard):
                buf.append(row)
                n += 1
        except (OSError, EOFError):
            continue          # shard en cours d'écriture : on retentera
        processed.add(shard.name)
    return n


def run(cfg: dict, seed_shards: str | None):
    ocfg = cfg["online"]
    device = cfg["train"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA indisponible -> CPU"); device = "cpu"

    model = build_model(cfg).to(device)
    latest = Path(cfg["paths"]["latest"])
    if latest.exists():
        model.load_state_dict(load_checkpoint(latest, device)["model_state"])
        print(f"Repris depuis {latest}")
    else:
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
            for row in iter_samples(shard):
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
            planes, idx, val = _encode_batch(rows, device)
            opt.zero_grad(set_to_none=True)
            logits, value = model(planes)
            loss = (F.cross_entropy(logits, idx)
                    + cfg["train"]["value_loss_weight"] * F.mse_loss(value.squeeze(1), val))
            loss.backward()
            opt.step()
            step += 1
            last_step_t = now
            if step % 20 == 0:
                print(f"online step {step} | loss {loss.item():.4f} | buffer {len(buf)}")

        if now - last_ckpt_t >= ckpt_every and step > 0:
            save_checkpoint(latest, model, cfg, opt, step)
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
