"""Pré-entraînement supervisé sur les samples Lichess.

Loss = cross-entropy(policy, coup joué) + λ · MSE(value, résultat).
Checkpoints écrits dans `checkpoints/`, dont `latest.pt` (rechargé à chaud par l'UCI).

Usage :  python -m sanchess.train.pretrain [--shards data/shards]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model import build_model
from ..utils import load_checkpoint, load_config, save_checkpoint
from .dataset import ShardDataset, find_shards


def train(cfg: dict, shards_dir: str | None, resume: bool = True):
    tcfg = cfg["train"]
    device = tcfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA indisponible -> CPU"); device = "cpu"

    shards = find_shards(shards_dir or cfg["data"]["shards_dir"])
    if not shards:
        raise SystemExit("Aucun shard. Lance d'abord pgn_to_samples.")
    max_samples = cfg["data"].get("max_train_samples")
    cap_txt = f" (plafond {max_samples} samples)" if max_samples else ""
    print(f"{len(shards)} shards trouvés. Chargement{cap_txt}…")
    ds = ShardDataset(shards, max_samples=max_samples)
    print(f"{len(ds)} samples chargés.")

    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=4, pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=True)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    use_amp = tcfg.get("amp", True) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    vlw = tcfg.get("value_loss_weight", 1.0)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    latest = Path(cfg["paths"]["latest"])
    max_steps = tcfg["pretrain_steps"]
    log_every = tcfg["log_every"]
    ckpt_every = tcfg["checkpoint_every"]

    step = 0
    if resume and latest.exists():
        ckpt = load_checkpoint(latest, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        step = int(ckpt.get("step", 0))
        if step >= max_steps:
            print(f"Reprise depuis {latest} : déjà {step} steps >= {max_steps} "
                  f"(pretrain_steps). Rien à faire.")
            return
        print(f"Reprise depuis {latest} au step {step}/{max_steps}.")

    model.train()
    t0 = time.time()
    running_p = running_v = 0.0
    while step < max_steps:
        for planes, target_idx, target_val in loader:
            planes = planes.to(device, non_blocking=True)
            target_idx = target_idx.to(device, non_blocking=True)
            target_val = target_val.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                enabled=use_amp):
                logits, value = model(planes)
                loss_p = F.cross_entropy(logits, target_idx)
                loss_v = F.mse_loss(value.squeeze(1), target_val)
                loss = loss_p + vlw * loss_v

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_p += loss_p.item()
            running_v += loss_v.item()
            step += 1

            if step % log_every == 0:
                sps = log_every / (time.time() - t0)
                print(f"step {step:>7} | policy {running_p/log_every:.4f} "
                      f"| value {running_v/log_every:.4f} | {sps:.1f} steps/s")
                running_p = running_v = 0.0
                t0 = time.time()

            if step % ckpt_every == 0:
                save_checkpoint(ckpt_dir / f"step_{step}.pt", model, cfg, opt, step)
                save_checkpoint(latest, model, cfg, opt, step)
                print(f"  checkpoint -> {latest}")

            if step >= max_steps:
                break

    save_checkpoint(latest, model, cfg, opt, step)
    print(f"Fini. Checkpoint final -> {latest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="Repartir de zéro au lieu de reprendre latest.pt.")
    args = ap.parse_args()
    train(load_config(args.config), args.shards, resume=not args.no_resume)


if __name__ == "__main__":
    main()
