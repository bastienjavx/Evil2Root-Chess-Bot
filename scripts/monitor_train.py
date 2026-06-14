#!/usr/bin/env python
"""Monitoring de l'entraînement local (pretrain / online) — snapshot ou --watch.

Le log stdout du pretrain est BLOCK-BUFFERISÉ quand il est redirigé vers un
fichier (les lignes `step ...` s'accumulent sans être flushées). Ce moniteur ne
dépend donc PAS du log pour l'essentiel : il lit le **step réel du checkpoint**
`latest.pt` (écrit tous les `checkpoint_every` steps) et le **temps écoulé du
process** pour en déduire cadence, progression et ETA. Les dernières pertes
réellement flushées dans le log sont affichées en bonus si disponibles.

Usage :
  python scripts/monitor_train.py                 # un instantané
  python scripts/monitor_train.py --watch 30       # rafraîchit toutes les 30 s
  python scripts/monitor_train.py --log data/pretrain_v3.log
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanchess.utils import load_config

_STEP_RE = re.compile(r"^step\s+(\d+)\s*\|.*policy\s+([0-9.]+).*value\s+([0-9.]+)"
                      r".*?([0-9.]+)\s*steps/s", re.IGNORECASE)
_PROC_RE = "sanchess.train.pretrai[n]"


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _pid_and_elapsed(pattern: str) -> tuple[int | None, float]:
    """PID du process d'entraînement (hors flock) et son temps écoulé (s)."""
    try:
        out = subprocess.run(["pgrep", "-f", pattern], text=True,
                             capture_output=True).stdout.split()
    except FileNotFoundError:
        return None, 0.0
    for pid in out:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "flock" in cmd:                 # on saute le wrapper flock
            continue
        try:
            etimes = subprocess.run(["ps", "-o", "etimes=", "-p", pid],
                                    text=True, capture_output=True).stdout.strip()
            return int(pid), float(etimes)
        except (ValueError, FileNotFoundError):
            return int(pid), 0.0
    return None, 0.0


def _checkpoint_step(latest: Path) -> tuple[int | None, float]:
    """(step, mtime) du checkpoint sans charger les poids (mmap, ~instantané)."""
    if not latest.exists():
        return None, 0.0
    try:
        import torch
        ck = torch.load(str(latest), map_location="cpu", weights_only=False,
                        mmap=True)
        return int(ck.get("step", 0)), latest.stat().st_mtime
    except Exception:
        return None, latest.stat().st_mtime if latest.exists() else 0.0


def _last_log_losses(log_path: Path, n: int = 3) -> list[str]:
    if not log_path or not log_path.exists():
        return []
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return [ln for ln in lines if _STEP_RE.match(ln)][-n:]


def _gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True).stdout.strip().splitlines()[0]
        util, used, total = (s.strip() for s in out.split(","))
        return f"{util}% | {used}/{total} Mo"
    except Exception:
        return "n/a"


def snapshot(cfg: dict, latest: Path, log_path: Path,
             prev: tuple[int, float] | None) -> tuple[int, float] | None:
    max_steps = int(cfg.get("train", {}).get("pretrain_steps", 0)) or None
    pid, elapsed = _pid_and_elapsed(_PROC_RE)
    step, ckpt_mtime = _checkpoint_step(latest)
    now = time.time()
    proc_start = now - elapsed if elapsed else None

    print("=" * 58)
    print(time.strftime("  %H:%M:%S") + ("  [entraînement ACTIF]" if pid
          else "  [AUCUN process pretrain — arrêté ?]"))
    if step is None:
        print("  latest.pt absent : checkpoint pas encore écrit (chargement"
              " des shards / avant le 1ᵉʳ checkpoint).")
    else:
        pct = f" ({100*step/max_steps:.1f}%)" if max_steps else ""
        print(f"  step (checkpoint) : {step:>8}{('/' + str(max_steps)) if max_steps else ''}{pct}")

    # Cadence : delta de step entre 2 snapshots (--watch) sinon moyenne globale.
    rate = None
    if prev and step is not None and step > prev[0] and now > prev[1]:
        rate = (step - prev[0]) / (now - prev[1])
        src = "instantanée"
    elif step is not None and proc_start and ckpt_mtime > proc_start:
        # Borne au mtime du checkpoint (et non `now`) : sinon le temps écoulé
        # APRÈS le dernier checkpoint (jusqu'à checkpoint_every steps non encore
        # enregistrés) fait chuter artificiellement la cadence.
        rate = step / (ckpt_mtime - proc_start)
        src = "moyenne (jusqu'au dernier checkpoint)"
    if rate:
        print(f"  cadence           : {rate:.1f} steps/s ({src})")
        if max_steps and step is not None:
            print(f"  ETA fin           : ~{_fmt_hms((max_steps - step) / rate)}"
                  f"  | écoulé {_fmt_hms(elapsed)}")
    print(f"  GPU               : {_gpu()}")

    losses = _last_log_losses(log_path)
    if losses:
        print("  dernières pertes (log flushé) :")
        for ln in losses:
            print("    " + ln.strip())
    else:
        print("  pertes log        : (buffer non flushé — normal ; le step"
              " checkpoint fait foi)")
    return (step, now) if step is not None else prev


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latest", default=cfg["paths"]["latest"],
                    help="checkpoint suivi (défaut: paths.latest)")
    ap.add_argument("--log", default="data/pretrain_v3.log",
                    help="log de l'entraînement (pour les pertes flushées)")
    ap.add_argument("--watch", type=float, default=0,
                    help="rafraîchir toutes les N secondes (0 = un seul snapshot)")
    args = ap.parse_args()

    latest, log_path = Path(args.latest), Path(args.log)
    prev = None
    while True:
        prev = snapshot(cfg, latest, log_path, prev)
        if args.watch <= 0:
            break
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nArrêt du monitoring.")
            break


if __name__ == "__main__":
    main()
