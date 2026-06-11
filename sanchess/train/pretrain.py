"""Pré-entraînement supervisé sur les samples Lichess.

Loss = cross-entropy(policy, coup joué) + λ · MSE(value, résultat).
Checkpoints écrits dans `checkpoints/`, dont `latest.pt` (rechargé à chaud par l'UCI).

Améliorations : sélection d'appareil auto (CUDA/MPS/CPU), schedule de LR
(warmup + cosine), clip de gradient, label smoothing sur la politique, et EMA
optionnelle des poids (meilleure qualité de jeu).

Usage :  python -m sanchess.train.pretrain [--shards data/shards]
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..model import build_model
from ..utils import (amp_enabled, autocast_ctx, load_checkpoint, load_config,
                     load_model_state, resolve_device, save_checkpoint)
from .dataset import ShardDataset, find_shards
from .losses import policy_loss, value_loss


def lr_at_step(step: int, base_lr: float, warmup: int, total: int,
               schedule: str, min_ratio: float = 0.05) -> float:
    """LR avec warmup linéaire puis décroissance cosinus (ou constante)."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule != "cosine" or total <= warmup:
        return base_lr
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(max(prog, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return base_lr * (min_ratio + (1 - min_ratio) * cos)


class EMA:
    """Moyenne mobile exponentielle des poids (pour de meilleurs checkpoints)."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self, model) -> dict:
        out = {k: v.detach().clone() for k, v in model.state_dict().items()}
        out.update(self.shadow)
        return out


def _save(path: Path, model, cfg, opt, step, ema: EMA | None) -> None:
    """Sauvegarde atomique. Avec EMA : model_state=EMA (jeu), raw_state pour reprise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if ema is not None:
        payload = {
            "model_state": ema.state_dict(model),
            "raw_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "model_cfg": cfg.get("model", {}),
            "step": step,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
    else:
        save_checkpoint(path, model, cfg, opt, step)


def train(cfg: dict, shards_dir: str | None, resume: bool = True):
    tcfg = cfg["train"]
    device = resolve_device(tcfg.get("device", "auto"))
    print(f"Appareil : {device}")

    shards = find_shards(shards_dir or cfg["data"]["shards_dir"])
    if not shards:
        raise SystemExit("Aucun shard. Lance d'abord pgn_to_samples.")
    max_samples = cfg["data"].get("max_train_samples")
    cap_txt = f" (plafond {max_samples} samples)" if max_samples else ""
    print(f"{len(shards)} shards trouvés. Chargement{cap_txt}…")
    ds = ShardDataset(shards, max_samples=max_samples)
    print(f"{len(ds)} samples chargés.")

    pin = device == "cuda"
    nworkers = tcfg.get("num_workers", 4)
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=nworkers, pin_memory=pin, drop_last=True,
                        persistent_workers=nworkers > 0)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    use_amp = amp_enabled(device, tcfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    vlw = tcfg.get("value_loss_weight", 1.0)
    grad_clip = tcfg.get("grad_clip", 1.0)
    label_smooth = tcfg.get("label_smoothing", 0.0)
    warmup = tcfg.get("warmup_steps", 2000)
    schedule = tcfg.get("lr_schedule", "cosine")
    base_lr = tcfg["lr"]

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    latest = Path(cfg["paths"]["latest"])
    max_steps = tcfg["pretrain_steps"]
    log_every = tcfg["log_every"]
    ckpt_every = tcfg["checkpoint_every"]

    step = 0
    if resume and latest.exists():
        ckpt = load_checkpoint(latest, map_location=device)
        info = load_model_state(model, ckpt.get("raw_state", ckpt["model_state"]))
        # Ne réutiliser l'optimiseur QUE si l'architecture est identique. Sinon
        # (warm-start après changement de tête, ex. scalar -> wdl) ses moments ont
        # des formes incompatibles et plantent au 1ᵉʳ step -> on repart à neuf.
        arch_identical = info and not any(info.values())
        if "optimizer_state" in ckpt and arch_identical:
            try:
                opt.load_state_dict(ckpt["optimizer_state"])
            except (ValueError, KeyError):
                print("État optimizer incompatible (archi modifiée) -> repart à neuf.")
        elif not arch_identical:
            print("Warm-start (archi modifiée) -> optimiseur réinitialisé.")
        step = int(ckpt.get("step", 0))
        if step >= max_steps:
            print(f"Reprise depuis {latest} : déjà {step} steps >= {max_steps} "
                  f"(pretrain_steps). Rien à faire.")
            return
        print(f"Reprise depuis {latest} au step {step}/{max_steps}.")

    ema_decay = tcfg.get("ema_decay", 0.0)
    ema = EMA(model, ema_decay) if ema_decay and ema_decay > 0 else None
    if ema is not None:
        print(f"EMA des poids activée (decay={ema_decay}).")

    model.train()
    t0 = time.time()
    running_p = running_v = 0.0
    while step < max_steps:
        for planes, target_policy, target_val in loader:
            planes = planes.to(device, non_blocking=True)
            target_policy = target_policy.to(device, non_blocking=True)
            target_val = target_val.to(device, non_blocking=True)

            lr = lr_at_step(step, base_lr, warmup, max_steps, schedule)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, use_amp):
                logits, value = model(planes)
                loss_p = policy_loss(logits, target_policy, label_smooth)
                loss_v = value_loss(value, target_val)
                loss = loss_p + vlw * loss_v

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running_p += loss_p.item()
            running_v += loss_v.item()
            step += 1

            if step % log_every == 0:
                sps = log_every / (time.time() - t0)
                print(f"step {step:>7} | policy {running_p/log_every:.4f} "
                      f"| value {running_v/log_every:.4f} | lr {lr:.2e} "
                      f"| {sps:.1f} steps/s")
                running_p = running_v = 0.0
                t0 = time.time()

            if step % ckpt_every == 0:
                _save(ckpt_dir / f"step_{step}.pt", model, cfg, opt, step, ema)
                _save(latest, model, cfg, opt, step, ema)
                print(f"  checkpoint -> {latest}")

            if step >= max_steps:
                break

    _save(latest, model, cfg, opt, step, ema)
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
