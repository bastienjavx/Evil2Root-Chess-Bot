#!/usr/bin/env python
"""Bench de DÉBIT D'ÉVALUATION du réseau sur le GPU local (avant de payer du cloud).

But : choisir la taille d'archi (blocs × canaux) qu'on pourra RÉELLEMENT faire
tourner ici. Sur cette machine la VRAM n'est pas la contrainte (même un 40×512
tient dans 8 Go) ; la force en blitz vient du **nombre de nœuds MCTS à budget de
temps fixe**, donc du **débit d'évals/s** du réseau. Ce bench mesure ce débit
pour plusieurs tailles candidates, EXACTEMENT comme le fait `mcts.Evaluator`
(eval(), no_grad, torch.autocast CUDA) afin que les chiffres soient transposables.

Métrique clé : évals/s à batch = mcts.eval_batch_size (plafond GPU pur ; le MCTS
réel est un peu en dessous à cause du surcoût CPU encode_board/arbre). Le bench
en déduit une estimation du nb de nœuds atteignables à un budget de temps donné.

Exemples :
  python scripts/bench_eval.py                       # tailles candidates par défaut
  python scripts/bench_eval.py --sizes 20x256 24x320 30x384 40x512
  python scripts/bench_eval.py --batch 16 --budget 5 10 --compile
  python scripts/bench_eval.py --dtype fp32          # sans autocast (référence)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanchess.encoding import INPUT_PLANES
from sanchess.model import SanChessNet, _model_kwargs
from sanchess.utils import load_config


def parse_size(s: str) -> tuple[int, int]:
    """'20x256' -> (blocks=20, channels=256)."""
    b, c = s.lower().split("x")
    return int(b), int(c)


def build(blocks: int, channels: int, base_model_cfg: dict) -> SanChessNet:
    """Réseau aux options de config.yaml (se/policy_head/value_head/activation…),
    seuls blocs et canaux étant remplacés par la taille candidate."""
    mcfg = dict(base_model_cfg)
    mcfg["blocks"] = blocks
    mcfg["channels"] = channels
    return SanChessNet(**_model_kwargs(mcfg))


@torch.no_grad()
def measure(model, device, batch, dtype, iters, warmup):
    """Renvoie évals/s pour un batch donné (autocast si dtype != fp32)."""
    x = torch.randn(batch, INPUT_PLANES, 8, 8, device=device)
    use_amp = device.startswith("cuda") and dtype != "fp32"
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    cm = (torch.autocast(device_type="cuda", dtype=amp_dtype)
          if use_amp else torch.autocast(device_type="cuda", enabled=False))

    def one():
        with cm:
            model(x)

    for _ in range(warmup):
        one()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch * iters / dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=["20x256", "24x320", "30x384", "40x512"],
                    help="tailles candidates 'BLOCSxCANAUX'")
    ap.add_argument("--batch", type=int, default=None,
                    help="taille de lot (défaut: mcts.eval_batch_size du config.yaml)")
    ap.add_argument("--budget", type=float, nargs="+", default=[5.0],
                    help="budgets de temps (s) pour estimer le nb de nœuds")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="précision d'éval (fp16 = ce que fait l'Evaluator sur CUDA)")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--compile", action="store_true",
                    help="essaie torch.compile (peut être plus lent à formes variables)")
    ap.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (défaut: auto)")
    args = ap.parse_args()

    cfg = load_config()
    base_model_cfg = cfg.get("model", {})
    batch = args.batch if args.batch is not None else int(cfg.get("mcts", {}).get("eval_batch_size", 16))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True  # formes fixes ici -> autotune OK
        name = torch.cuda.get_device_name(0)
    else:
        name = "CPU"

    print(f"GPU      : {name}")
    print(f"device   : {device}   dtype: {args.dtype}   batch: {batch}"
          f"   iters: {args.iters} (warmup {args.warmup})")
    print(f"archi de base (config.yaml): se={base_model_cfg.get('se')} "
          f"policy={base_model_cfg.get('policy_head')} value={base_model_cfg.get('value_head')} "
          f"act={base_model_cfg.get('activation')}")
    print()
    hdr = f"{'taille':>10} | {'params':>8} | {'VRAM':>7} | {'évals/s':>8} | {'b1/s':>6} | "
    hdr += " | ".join(f"n@{int(b)}s" if b == int(b) else f"n@{b}s" for b in args.budget)
    print(hdr)
    print("-" * len(hdr))

    for s in args.sizes:
        blocks, channels = parse_size(s)
        try:
            model = build(blocks, channels, base_model_cfg).to(device).eval()
        except Exception as e:  # OOM à la construction d'un très gros réseau
            print(f"{s:>10} | échec construction: {e}")
            continue
        params = sum(p.numel() for p in model.parameters())

        if args.compile:
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"  (torch.compile indisponible: {e})")

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        try:
            eps = measure(model, device, batch, args.dtype, args.iters, args.warmup)
            eps1 = measure(model, device, 1, args.dtype, max(args.iters, 100), args.warmup)
        except RuntimeError as e:
            print(f"{s:>10} | {params/1e6:7.1f}M | OOM/err: {e}")
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue

        vram = (torch.cuda.max_memory_allocated() / 1e6) if device.startswith("cuda") else 0.0
        nodes = " | ".join(f"{eps*b:7.0f}" for b in args.budget)
        print(f"{s:>10} | {params/1e6:7.1f}M | {vram:5.0f}Mo | {eps:8.0f} | {eps1:6.0f} | {nodes}")

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print()
    print("évals/s = débit GPU pur à batch plein (plafond). Le MCTS réel est ~30-40 %")
    print("en dessous (encode_board + arbre côté CPU). b1/s = débit à batch 1 (racine).")
    print("Règle : viser une archi qui garde assez de nœuds au budget blitz (~5000 pour")
    print("les tactiques 2-3 coups, cf. config bot.max_think_seconds).")


if __name__ == "__main__":
    main()
