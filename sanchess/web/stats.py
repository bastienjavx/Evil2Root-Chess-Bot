"""Statistiques d'entraînement et état système pour l'interface web.

Lit les logs (`data/*.log`), interroge `nvidia-smi`, `systemctl`, et compte le
replay buffer / les shards. Tout est best-effort : une source absente renvoie
un champ vide plutôt qu'une erreur.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"

# step  207000 | policy 1.6184 | value 0.7889 | 3.5 steps/s
_PRETRAIN_RE = re.compile(
    r"step\s+(\d+)\s*\|\s*policy\s+([\d.]+)\s*\|\s*value\s+([\d.]+)\s*\|\s*([\d.]+)\s*steps/s")
# online step 1234 | loss 0.1234 | buffer 50000
_ONLINE_RE = re.compile(
    r"online step\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*buffer\s+(\d+)")

_SERVICES = ["sano1-train", "sano1-online", "sano1-stream",
             "sano1-download", "sano1-bot"]


def _tail_lines(path: Path, n: int = 4000) -> list[str]:
    """Dernières lignes d'un fichier (lecture bornée mémoire)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, n * 120)
            f.seek(size - block)
            data = f.read().decode("utf-8", "replace")
        return data.splitlines()[-n:]
    except OSError:
        return []


def _latest_log(prefix: str) -> Path | None:
    link = DATA / f"{prefix}_latest.log"
    if link.exists():
        return link
    cands = sorted(DATA.glob(f"{prefix}_*.log"))
    return cands[-1] if cands else None


def training_series(max_points: int = 400) -> dict:
    """Séries temporelles d'entraînement (pretrain + online)."""
    out = {"pretrain": {"steps": [], "policy": [], "value": []},
           "online": {"steps": [], "loss": [], "buffer": []},
           "pretrain_log": None, "online_log": None}

    plog = _latest_log("pretrain")
    if plog:
        out["pretrain_log"] = plog.name
        pts = []
        for line in _tail_lines(plog):
            m = _PRETRAIN_RE.search(line)
            if m:
                pts.append((int(m[1]), float(m[2]), float(m[3])))
        pts = _downsample(pts, max_points)
        out["pretrain"]["steps"] = [p[0] for p in pts]
        out["pretrain"]["policy"] = [p[1] for p in pts]
        out["pretrain"]["value"] = [p[2] for p in pts]

    olog = _latest_log("online")
    if olog:
        out["online_log"] = olog.name
        pts = []
        for line in _tail_lines(olog):
            m = _ONLINE_RE.search(line)
            if m:
                pts.append((int(m[1]), float(m[2]), int(m[3])))
        pts = _downsample(pts, max_points)
        out["online"]["steps"] = [p[0] for p in pts]
        out["online"]["loss"] = [p[1] for p in pts]
        out["online"]["buffer"] = [p[2] for p in pts]
    return out


def _downsample(pts: list, max_points: int) -> list:
    if len(pts) <= max_points:
        return pts
    stride = len(pts) // max_points + 1
    return pts[::stride]


def training_summary() -> dict:
    """Dernier point connu de chaque phase d'entraînement."""
    s = training_series(max_points=10_000)
    out = {}
    ps = s["pretrain"]["steps"]
    if ps:
        out["pretrain"] = {
            "step": ps[-1],
            "policy": s["pretrain"]["policy"][-1],
            "value": s["pretrain"]["value"][-1],
            "log": s["pretrain_log"],
        }
    os_ = s["online"]["steps"]
    if os_:
        out["online"] = {
            "step": os_[-1],
            "loss": s["online"]["loss"][-1],
            "buffer": s["online"]["buffer"][-1],
            "log": s["online_log"],
        }
    return out


def gpu_info() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append({
            "name": parts[0],
            "mem_used_mb": _num(parts[1]),
            "mem_total_mb": _num(parts[2]),
            "util_pct": _num(parts[3]),
            "temp_c": _num(parts[4]) if len(parts) > 4 else None,
            "power_w": _num(parts[5]) if len(parts) > 5 else None,
        })
    return gpus


def _num(x):
    try:
        return float(x) if "." in x else int(x)
    except (ValueError, TypeError):
        return None


def services_status() -> list[dict]:
    if not shutil.which("systemctl"):
        return []
    out = []
    for svc in _SERVICES:
        try:
            r = subprocess.run(["systemctl", "is-active", svc],
                               capture_output=True, text=True, timeout=5)
            state = r.stdout.strip() or "unknown"
        except Exception:
            state = "unknown"
        out.append({"name": svc, "active": state == "active", "state": state})
    return out


def data_status() -> dict:
    buf_dir = DATA / "replay_buffer"
    shards_dir = DATA / "shards"
    buf = list(buf_dir.glob("*.gz")) if buf_dir.exists() else []
    shards = list(shards_dir.glob("*.txt.gz")) if shards_dir.exists() else []
    buf_bytes = sum(p.stat().st_size for p in buf) if buf else 0
    return {
        "replay_buffer_files": len(buf),
        "replay_buffer_mb": round(buf_bytes / 1e6, 1),
        "shard_files": len(shards),
    }


def training_processes() -> list[dict]:
    """Process Python sanchess en cours (best-effort via /proc)."""
    procs = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "sanchess." in cmd and "sanchess.web" not in cmd:
            mod = "?"
            m = re.search(r"sanchess\.(\S+)", cmd)
            if m:
                mod = m.group(1)
            procs.append({"pid": int(p.name), "module": mod})
    return procs


def system_status() -> dict:
    return {
        "time": time.time(),
        "gpu": gpu_info(),
        "services": services_status(),
        "data": data_status(),
        "training": training_summary(),
        "processes": training_processes(),
    }
