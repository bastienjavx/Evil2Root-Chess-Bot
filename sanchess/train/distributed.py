"""Entraînement distribué multi-appareils (data-parallel) — Mac M1 + Linux.

Conçu pour un cluster HÉTÉROGÈNE : chaque nœud calcule sur son accélérateur
local (CUDA sur Linux, **MPS sur Apple Silicon**, ou CPU), et la synchronisation
des gradients se fait via le backend **gloo** sur des tenseurs CPU. C'est la clé
de l'interopérabilité Mac/Linux : NCCL est CUDA-only et `DistributedDataParallel`
ne sait pas mélanger CUDA et MPS, alors qu'un all-reduce gloo sur CPU fonctionne
partout, quel que soit l'accélérateur de chaque machine.

Algorithme (SGD synchrone / "manual DDP") :
  1. chaque rang charge la même config et construit le même réseau ;
  2. les poids du rang 0 sont diffusés à tous (départ identique) ;
  3. chaque rang voit un sous-ensemble disjoint des données (DistributedSampler) ;
  4. forward/backward en local, puis all-reduce (moyenne) des gradients via gloo ;
  5. tous appliquent le même pas d'optimiseur -> les modèles restent synchronisés ;
  6. le rang 0 journalise et écrit `checkpoints/latest.pt` (hot-reload du moteur).

Lancement (exemple à 2 machines, master = la Linux 192.168.1.10) :

  # Sur la Linux (rang 0, master) :
  MASTER_ADDR=192.168.1.10 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 \
      python -m sanchess.train.distributed --shards data/shards

  # Sur le Mac M1 (rang 1) :
  MASTER_ADDR=192.168.1.10 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 \
      python -m sanchess.train.distributed --shards data/shards

Voir aussi `scripts/run_distributed.sh`.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..model import build_model
from ..utils import (amp_enabled, autocast_ctx, load_checkpoint, load_config,
                     load_model_state, resolve_device, save_checkpoint)
from .dataset import ShardDataset, find_shards
from .losses import policy_loss, value_loss
from .pretrain import EMA, lr_at_step


# --- Initialisation du groupe de processus ------------------------------------

def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def init_distributed(cfg: dict, args) -> tuple[int, int, str]:
    """Initialise le process group. Retourne (rank, world_size, backend)."""
    dcfg = cfg.get("distributed", {})
    rank = int(args.rank if args.rank is not None else _env("RANK", "0"))
    world = int(args.world_size if args.world_size is not None
                else _env("WORLD_SIZE", "1"))
    backend = (args.backend or _env("DIST_BACKEND") or dcfg.get("backend", "gloo")).lower()

    os.environ.setdefault("MASTER_ADDR",
                          args.master_addr or _env("MASTER_ADDR")
                          or dcfg.get("master_addr", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT",
                          str(args.master_port or _env("MASTER_PORT")
                              or dcfg.get("master_port", 29500)))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)

    if world > 1:
        timeout = __import__("datetime").timedelta(
            seconds=int(dcfg.get("timeout_sec", 1800)))
        dist.init_process_group(backend=backend, rank=rank,
                                world_size=world, timeout=timeout)
    return rank, world, backend


# --- Synchronisation des paramètres / gradients via gloo (CPU) ----------------

@torch.no_grad()
def _flatten(tensors) -> torch.Tensor:
    return torch.cat([t.detach().to("cpu", dtype=torch.float32).reshape(-1)
                      for t in tensors])


@torch.no_grad()
def _unflatten_into(flat: torch.Tensor, tensors) -> None:
    offset = 0
    for t in tensors:
        n = t.numel()
        t.copy_(flat[offset:offset + n].reshape(t.shape).to(t.device, t.dtype))
        offset += n


@torch.no_grad()
def broadcast_params(model, world: int, src: int = 0) -> None:
    """Aligne tous les rangs sur les poids du rang `src`."""
    if world <= 1:
        return
    params = list(model.parameters())
    flat = _flatten(params)
    dist.broadcast(flat, src=src)
    _unflatten_into(flat, params)


@torch.no_grad()
def all_reduce_grads(model, world: int) -> None:
    """Moyenne les gradients de tous les rangs (all-reduce gloo sur CPU)."""
    if world <= 1:
        return
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    flat = _flatten(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world
    _unflatten_into(flat, grads)


# --- Boucle d'entraînement ----------------------------------------------------

def run(cfg: dict, shards_dir: str | None, args, resume: bool = True):
    rank, world, backend = init_distributed(cfg, args)
    is_main = rank == 0
    tcfg = cfg["train"]
    # Appareil PAR RANG : permet, sur une seule machine, un rang GPU (cuda) ET un
    # rang CPU qui s'entraînent ensemble. Priorité : --device > $DEVICE > train.device.
    device = resolve_device(args.device or _env("DEVICE") or tcfg.get("device", "auto"))

    def log(msg: str) -> None:
        if is_main:
            print(msg, flush=True)

    log(f"Distribué : world_size={world}, backend={backend}")
    print(f"[rang {rank}] appareil={device}", flush=True)

    shards = find_shards(shards_dir or cfg["data"]["shards_dir"])
    if not shards:
        raise SystemExit("Aucun shard. Lance d'abord pgn_to_samples.")
    max_samples = cfg["data"].get("max_train_samples")
    ds = ShardDataset(shards, max_samples=max_samples)
    log(f"{len(ds)} samples chargés par rang.")

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                 shuffle=True, drop_last=True) if world > 1 else None
    pin = device == "cuda"
    nworkers = tcfg.get("num_workers", 4)
    loader = DataLoader(ds, batch_size=tcfg["batch_size"],
                        shuffle=(sampler is None), sampler=sampler,
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
        load_model_state(model, ckpt.get("raw_state", ckpt["model_state"]))
        step = int(ckpt.get("step", 0))
        log(f"Reprise depuis {latest} au step {step}/{max_steps}.")

    # Tous les rangs partent EXACTEMENT des mêmes poids.
    broadcast_params(model, world, src=0)

    ema_decay = tcfg.get("ema_decay", 0.0)
    ema = EMA(model, ema_decay) if (is_main and ema_decay and ema_decay > 0) else None

    model.train()
    t0 = time.time()
    running_p = running_v = 0.0
    epoch = 0
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
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
            scaler.unscale_(opt)
            all_reduce_grads(model, world)          # <-- synchro inter-machines
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running_p += loss_p.item()
            running_v += loss_v.item()
            step += 1

            if is_main and step % log_every == 0:
                sps = log_every / (time.time() - t0)
                eff = sps * tcfg["batch_size"] * world
                log(f"step {step:>7} | policy {running_p/log_every:.4f} "
                    f"| value {running_v/log_every:.4f} | lr {lr:.2e} "
                    f"| {sps:.1f} steps/s (~{eff:.0f} samples/s global)")
                running_p = running_v = 0.0
                t0 = time.time()

            if is_main and step % ckpt_every == 0:
                _save_dist(ckpt_dir / f"step_{step}.pt", model, cfg, opt, step, ema)
                _save_dist(latest, model, cfg, opt, step, ema)
                log(f"  checkpoint -> {latest}")

            if step >= max_steps:
                break
        epoch += 1

    if is_main:
        _save_dist(latest, model, cfg, opt, step, ema)
        log(f"Fini. Checkpoint final -> {latest}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _save_dist(path: Path, model, cfg, opt, step, ema) -> None:
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


def main():
    ap = argparse.ArgumentParser(description="Entraînement distribué Mac M1 + Linux.")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--world-size", type=int, default=None, dest="world_size")
    ap.add_argument("--master-addr", default=None, dest="master_addr")
    ap.add_argument("--master-port", type=int, default=None, dest="master_port")
    ap.add_argument("--backend", default=None, help="gloo (défaut, cross-platform) ou nccl")
    ap.add_argument("--device", default=None,
                    help="appareil de CE rang (cuda/mps/cpu) ; sinon $DEVICE puis train.device")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    run(load_config(args.config), args.shards, args, resume=not args.no_resume)


if __name__ == "__main__":
    main()
