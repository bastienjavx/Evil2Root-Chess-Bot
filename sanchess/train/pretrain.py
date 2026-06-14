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

from ..model import build_model, _model_kwargs
from ..utils import (amp_enabled, autocast_ctx, load_checkpoint, load_config,
                     load_model_state, resolve_amp_dtype, resolve_device,
                     save_checkpoint, unwrap_model)
from .dataset import ShardDataset, find_shards, split_batch
from .losses import policy_loss, value_loss, moves_left_loss


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


def _prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    """Ne conserve que les `keep` snapshots `step_*.pt` les plus récents (par numéro
    de step). Chaque .pt pèse ~0.5 Go (état optimiseur inclus) : sans rotation,
    `checkpoints/` gonfle sans fin. `latest.pt` / `ref_strength.pt` ne matchent pas
    le motif -> jamais supprimés. keep <= 0 = tout garder (comportement historique).
    """
    if not keep or keep <= 0:
        return
    snaps = []
    for p in ckpt_dir.glob("step_*.pt"):
        try:
            snaps.append((int(p.stem.split("_")[1]), p))
        except (IndexError, ValueError):
            continue
    snaps.sort()
    for _, p in snaps[:-keep]:
        try:
            p.unlink()
        except OSError:
            pass


def _save(path: Path, model, cfg, opt, step, ema: EMA | None) -> None:
    """Sauvegarde atomique. Avec EMA : model_state=EMA (jeu), raw_state pour reprise.

    On dé-wrappe un éventuel modèle `torch.compile` : sinon les clés `_orig_mod.`
    rendraient le checkpoint illisible par le moteur/online au hot-reload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model = unwrap_model(model)
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
    mask_policy = bool(tcfg.get("mask_policy_loss", False))
    input_features = cfg.get("model", {}).get("input_features", "base")
    with_ml = bool(_model_kwargs(cfg.get("model", {}))["moves_left_head"])
    ds = ShardDataset(shards, max_samples=max_samples,
                      input_features=input_features, mask_policy=mask_policy,
                      moves_left=with_ml)
    print(f"{len(ds)} samples chargés.")

    pin = device == "cuda"
    nworkers = tcfg.get("num_workers", 4)
    # prefetch_factor n'est valide qu'avec des workers (>0) ; sinon il faut None.
    extra_loader = ({"prefetch_factor": tcfg.get("prefetch_factor", 2)}
                    if nworkers > 0 else {})
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=nworkers, pin_memory=pin, drop_last=True,
                        persistent_workers=nworkers > 0, **extra_loader)

    model = build_model(cfg).to(device)
    if device == "cuda":
        # channels_last : conv 2D plus rapides sur tensor cores (Ampere/Ada/Hopper).
        model = model.to(memory_format=torch.channels_last)
    # torch.compile (formes fixes 8x8 -> gros gain sur gros GPU). On le garde
    # OPTIONNEL et tolérant : si la compilation échoue (vieux GPU, build sans
    # Triton…), on retombe sur le modèle eager. Le save dé-wrappe le `_orig_mod.`.
    if tcfg.get("compile", False) and device == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            print("torch.compile activé (max-autotune).")
        except Exception as e:
            print(f"torch.compile ignoré : {e}")
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    use_amp = amp_enabled(device, tcfg.get("amp", True))
    amp_dtype = resolve_amp_dtype(device, tcfg.get("amp_dtype")) if use_amp else None
    # GradScaler UNIQUEMENT en fp16 : bf16 a la plage dynamique d'un float32, donc
    # pas d'underflow de gradient à compenser -> pas de scaler (cf. doc PyTorch).
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if use_amp:
        print(f"AMP : {('bf16' if amp_dtype == torch.bfloat16 else 'fp16')}"
              f" (GradScaler {'actif' if use_scaler else 'désactivé'}).")
    vlw = tcfg.get("value_loss_weight", 1.0)
    mlw = tcfg.get("moves_left_weight", 0.3) if with_ml else 0.0
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
    keep_last_ckpt = tcfg.get("keep_last_checkpoints", 0)

    step = 0
    if resume and latest.exists():
        ckpt = load_checkpoint(latest, map_location=device)
        # unwrap : si le modèle est compilé, ses clés portent `_orig_mod.` et les
        # poids (clés propres) ne se chargeraient PAS -> on charge dans le module nu.
        info = load_model_state(unwrap_model(model),
                                ckpt.get("raw_state", ckpt["model_state"]))
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
    # EMA toujours pilotée par le module NU (clés sans `_orig_mod.`) : sinon, sous
    # torch.compile, `k in shadow` serait toujours faux et l'EMA ne bougerait jamais.
    ema = EMA(unwrap_model(model), ema_decay) if ema_decay and ema_decay > 0 else None
    if ema is not None:
        print(f"EMA des poids activée (decay={ema_decay}).")

    model.train()
    t0 = time.time()
    running_p = running_v = 0.0
    while step < max_steps:
        for batch in loader:
            (planes, target_policy, legal_mask, target_val,
             target_ml, ml_mask) = split_batch(batch, mask_policy, with_ml)
            planes = planes.to(device, non_blocking=True)
            if device == "cuda":
                planes = planes.contiguous(memory_format=torch.channels_last)
            target_policy = target_policy.to(device, non_blocking=True)
            if legal_mask is not None:
                legal_mask = legal_mask.to(device, non_blocking=True)
            target_val = target_val.to(device, non_blocking=True)
            if target_ml is not None:
                target_ml = target_ml.to(device, non_blocking=True)
                ml_mask = ml_mask.to(device, non_blocking=True)

            lr = lr_at_step(step, base_lr, warmup, max_steps, schedule)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, use_amp, amp_dtype):
                logits, value, moves_left = model(planes)
                loss_p = policy_loss(logits, target_policy, label_smooth, legal_mask)
                loss_v = value_loss(value, target_val)
                loss = loss_p + vlw * loss_v
                if mlw and moves_left is not None:
                    loss = loss + mlw * moves_left_loss(moves_left, target_ml, ml_mask)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(unwrap_model(model))

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
                _prune_checkpoints(ckpt_dir, keep_last_ckpt)
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
