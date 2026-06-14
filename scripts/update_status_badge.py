#!/usr/bin/env python3
"""Génère les JSON shields.io (endpoint) de l'état d'entraînement et pousse
le gist GitHub qui les héberge, pour des badges « temps réel » dans le README.

Deux badges :
  - training : phase courante (pretrain / online / idle) + step + métrique ;
  - machine  : hostname de la machine qui entraîne + GPU + utilisation.

L'ID du gist est lu dans $SANO1_STATUS_GIST puis dans scripts/status_gist_id.txt.
Si aucun ID n'est trouvé et que --create est passé, un gist est créé et son ID
écrit dans ce fichier. La mise à jour utilise la CLI `gh` (déjà authentifiée).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIST_ID_FILE = ROOT / "scripts" / "status_gist_id.txt"

TRAINING_FILE = "sano1-training.json"
MACHINE_FILE = "sano1-machine.json"

# shields.io clampe l'endpoint à >=300s ; on demande le minimum.
CACHE_SECONDS = 300


def _load_stats():
    """Charge sanchess/web/stats.py sans importer le package (pas de torch)."""
    path = ROOT / "sanchess" / "web" / "stats.py"
    spec = importlib.util.spec_from_file_location("sano1_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fmt_step(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _short_gpu(name: str) -> str:
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _checkpoint_progress() -> tuple[int | None, int | None]:
    """(step, total) du pretrain depuis latest.pt (mmap, ~instantané) + config.

    Source faisant foi : le log stdout du pretrain est block-bufferisé et le
    symlink `*_latest.log` peut être périmé (cf. monitor_train.py). Le step réel
    est celui écrit dans le checkpoint tous les `checkpoint_every` steps.
    """
    latest = ROOT / "checkpoints" / "latest.pt"
    total = None
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        latest = ROOT / cfg.get("paths", {}).get("latest", "checkpoints/latest.pt")
        total = int(cfg.get("train", {}).get("pretrain_steps", 0)) or None
    except Exception:
        pass
    if not latest.exists():
        return None, total
    try:
        import torch
        ck = torch.load(str(latest), map_location="cpu",
                        weights_only=False, mmap=True)
        return int(ck.get("step", 0)), total
    except Exception:  # checkpoint en cours d'écriture / torch absent
        return None, total


def build_badges(stats) -> dict[str, dict]:
    svc = {s["name"]: s for s in stats.services_status()}
    summary = stats.training_summary()
    gpus = stats.gpu_info()

    # Repli hors systemd : détection des process via /proc (entraînement lancé
    # à la main). On combine services actifs OU process correspondant.
    mods = [p.get("module", "") for p in stats.training_processes()]

    def _proc(*names) -> bool:
        return any(m.startswith(n) for m in mods for n in names)

    pretrain_running = bool(svc.get("sano1-train", {}).get("active")) or \
        _proc("train.pretrain")
    online_running = bool(svc.get("sano1-online", {}).get("active")) or \
        _proc("train.online")
    distributed_running = _proc("train.distributed")
    selfplay_running = _proc("train.selfplay")  # couvre selfplay et selfplay_gpu

    # --- badge training ---
    if pretrain_running:
        step, total = _checkpoint_progress()  # step réel (checkpoint fait foi)
        if step is None and "pretrain" in summary:
            step = summary["pretrain"]["step"]  # repli log si checkpoint illisible
        if step is not None and total:
            msg = f"pretrain · {_fmt_step(step)}/{_fmt_step(total)} ({100*step/total:.1f}%)"
        elif step is not None:
            msg = f"pretrain · step {_fmt_step(step)}"
        else:
            msg = "pretrain · starting…"
        color = "brightgreen"
    elif online_running and "online" in summary:
        o = summary["online"]
        msg = f"online · step {_fmt_step(o['step'])} · loss {o['loss']:.2f}"
        color = "brightgreen"
    elif distributed_running:
        msg = "distributed"
        color = "brightgreen"
    elif pretrain_running or online_running:
        msg = "starting…"
        color = "yellow"
    elif selfplay_running:
        msg = "self-play (data gen)"
        color = "blue"
    else:
        msg = "idle"
        color = "lightgrey"

    training = {
        "schemaVersion": 1,
        "label": "training",
        "message": msg,
        "color": color,
        "cacheSeconds": CACHE_SECONDS,
    }

    # --- badge machine ---
    host = socket.gethostname()
    if gpus:
        g = gpus[0]
        gpu_name = _short_gpu(g.get("name") or "GPU")
        util = g.get("util_pct")
        temp = g.get("temp_c")
        bits = [host, gpu_name]
        if util is not None:
            bits.append(f"{util}%")
        if temp is not None:
            bits.append(f"{temp}°C")
        mmsg = " · ".join(str(b) for b in bits)
        # vert si la carte travaille, gris si oisive
        mcolor = "blue" if (util or 0) >= 20 else "lightgrey"
    else:
        mmsg = f"{host} · no GPU"
        mcolor = "lightgrey"

    machine = {
        "schemaVersion": 1,
        "label": "machine",
        "message": mmsg,
        "color": mcolor,
        "cacheSeconds": CACHE_SECONDS,
    }
    return {TRAINING_FILE: training, MACHINE_FILE: machine}


def _read_gist_id() -> str | None:
    import os
    env = os.environ.get("SANO1_STATUS_GIST")
    if env:
        return env.strip()
    if GIST_ID_FILE.exists():
        v = GIST_ID_FILE.read_text().strip()
        if v:
            return v
    return None


def _write_files(badges: dict[str, dict], dest: Path) -> None:
    for name, payload in badges.items():
        (dest / name).write_text(json.dumps(payload, ensure_ascii=False) + "\n")


def _create_gist(badges: dict[str, dict]) -> str:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_files(badges, d)
        files = [str(d / n) for n in badges]
        out = subprocess.run(
            ["gh", "gist", "create", "--public",
             "--desc", "San-o1 live training status (shields.io endpoints)",
             *files],
            capture_output=True, text=True, check=True).stdout.strip()
    gid = out.rstrip("/").split("/")[-1]
    GIST_ID_FILE.write_text(gid + "\n")
    return gid


def _update_gist(gid: str, badges: dict[str, dict]) -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_files(badges, d)
        for name in badges:
            subprocess.run(
                ["gh", "gist", "edit", gid, "-f", name, str(d / name)],
                capture_output=True, text=True, check=True)


def _gist_user() -> str:
    return subprocess.run(["gh", "api", "user", "-q", ".login"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true",
                    help="créer le gist s'il n'existe pas et afficher les URLs")
    ap.add_argument("--print-urls", action="store_true",
                    help="afficher les URLs shields.io et quitter")
    args = ap.parse_args()

    stats = _load_stats()
    badges = build_badges(stats)

    gid = _read_gist_id()
    if gid is None:
        if not args.create:
            print("Aucun gist configuré. Lancer une fois avec --create.",
                  file=sys.stderr)
            return 2
        gid = _create_gist(badges)
        print(f"Gist créé : {gid}")
    else:
        _update_gist(gid, badges)

    if args.create or args.print_urls:
        user = _gist_user()
        for name in (TRAINING_FILE, MACHINE_FILE):
            raw = f"https://gist.githubusercontent.com/{user}/{gid}/raw/{name}"
            shield = f"https://img.shields.io/endpoint?url={raw}"
            print(f"{name}: {shield}")
    print(f"[badge] {time.strftime('%H:%M:%S')} "
          f"{badges[TRAINING_FILE]['message']} | {badges[MACHINE_FILE]['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
