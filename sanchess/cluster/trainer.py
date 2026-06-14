"""Connecteur trainer : relie l'entraînement local (GPU) au coordinateur du cluster.

À lancer sur UNE machine GPU (ex. la RTX 2070S). Trois tâches concurrentes :

  1. SYNC SHARDS   : tire les nouvelles parties du coordinateur (/cluster/shards) et les
                     écrit dans data/replay_buffer/ — exactement le dossier que
                     train/online.py surveille déjà.
  2. ENTRAÎNEMENT  : lance `python -m sanchess.train.online` (INCHANGÉ), qui ingère le
                     buffer, fait les pas de gradient (LR faible) et écrit
                     checkpoints/latest.pt par hot-reload.
  3. PUBLICATION   : surveille latest.pt ; à chaque mise à jour, strippe l'optimiseur
                     (poids seuls -> ~2-3x plus léger) et publie sur /cluster/model/publish
                     (auth TRAINER_TOKEN). Les workers détectent la nouvelle version et
                     re-téléchargent : la boucle d'amélioration est bouclée.

Usage :
    TRAINER_TOKEN=monsecret python -m sanchess.cluster.trainer --server https://<app>.up.railway.app
    python -m sanchess.cluster.trainer --server http://localhost:8001 --token dev --no-train
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

from ..utils import load_config
from . import protocol as P


# --- 1. Synchronisation des shards (serveur -> replay buffer local) ----------

def _sync_shards(server: str, buffer_dir: Path, stop: threading.Event,
                 poll_sec: float) -> None:
    buffer_dir.mkdir(parents=True, exist_ok=True)
    cursor = ""
    seen: set[str] = {p.name for p in buffer_dir.glob("cluster_*.txt.gz")}
    while not stop.is_set():
        try:
            r = requests.get(server + P.EP_SHARDS, params={"since": cursor}, timeout=30)
            r.raise_for_status()
            data = r.json()
            names = data.get("shards", [])
            for name in names:
                local = buffer_dir / f"cluster_{name}"
                if local.name in seen or local.exists():
                    cursor = name
                    continue
                d = requests.get(server + P.EP_SHARDS, params={"download": name}, timeout=120)
                if d.status_code != 200:
                    continue
                tmp = buffer_dir / f".{local.name}.part"
                tmp.write_bytes(d.content)
                os.replace(tmp, local)
                seen.add(local.name)
                cursor = name
            if names:
                print(f"[trainer] {len(names)} shard(s) synchronisés "
                      f"(curseur={cursor})", flush=True)
        except Exception as exc:
            print(f"[trainer] sync shards : {exc}", flush=True)
        stop.wait(poll_sec)


# --- 3. Publication du modèle (latest.pt strippé -> coordinateur) ------------

def _strip_and_publish(server: str, token: str, latest: Path) -> bool:
    """Charge latest.pt, retire l'optimiseur, publie les poids. Retourne True si OK."""
    import torch
    from ..utils import load_checkpoint, unwrap_model  # noqa: F401 (cohérence d'API)

    ck = load_checkpoint(latest, "cpu")
    payload = {
        "model_state": ck.get("raw_state", ck["model_state"]),
        "model_cfg": ck.get("model_cfg", {}),
        "step": int(ck.get("step", 0)),
    }
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        blob = Path(tmp).read_bytes()
    finally:
        os.unlink(tmp)
    r = requests.post(server + P.EP_MODEL_PUBLISH,
                      headers={"Authorization": f"Bearer {token}"},
                      files={"model": ("weights.pt", blob, "application/octet-stream")},
                      data={"step": payload["step"]}, timeout=120)
    if r.status_code != 200:
        print(f"[trainer] publish refusé : {r.status_code} {r.text}", flush=True)
        return False
    info = r.json()
    print(f"[trainer] modèle publié v{info['version']} (step {info['step']}, "
          f"{info['size']//1024} Ko)", flush=True)
    return True


def _publish_loop(server: str, token: str, latest: Path, stop: threading.Event,
                  poll_sec: float) -> None:
    last_mtime = None
    # Publication initiale si un checkpoint existe déjà (les workers ont un modèle dès le départ).
    while not stop.is_set():
        try:
            if latest.exists():
                m = latest.stat().st_mtime
                if m != last_mtime:
                    if _strip_and_publish(server, token, latest):
                        last_mtime = m
        except Exception as exc:
            print(f"[trainer] publish : {exc}", flush=True)
        stop.wait(poll_sec)


# --- 2. Entraînement (online.py inchangé) ------------------------------------

def _start_online(args) -> subprocess.Popen | None:
    cmd = [sys.executable, "-m", "sanchess.train.online"]
    if args.config:
        cmd += ["--config", args.config]
    if args.seed_shards:
        cmd += ["--seed-shards", args.seed_shards]
    print(f"[trainer] lancement entraînement : {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Connecteur trainer San-o1 (sync parties + entraînement + publication).")
    ap.add_argument("--server", required=True, help="URL du coordinateur (Railway)")
    ap.add_argument("--token", default=os.environ.get("TRAINER_TOKEN", ""),
                    help="TRAINER_TOKEN (sinon variable d'environnement)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed-shards", default=None, dest="seed_shards",
                    help="dossier de shards initiaux passé à online.py (amorçage)")
    ap.add_argument("--no-train", action="store_true", dest="no_train",
                    help="ne pas lancer online.py (tu le lances toi-même) : ne fait que "
                         "synchroniser les parties et publier latest.pt")
    ap.add_argument("--sync-sec", type=float, default=10.0, dest="sync_sec",
                    help="période de synchro des shards")
    ap.add_argument("--publish-sec", type=float, default=15.0, dest="publish_sec",
                    help="période de vérification de latest.pt pour publication")
    args = ap.parse_args()

    if not args.token:
        sys.exit("Erreur : TRAINER_TOKEN requis (--token ou variable d'env).")

    server = args.server.rstrip("/")
    cfg = load_config(args.config) if args.config else load_config()
    buffer_dir = Path(cfg["data"]["buffer_dir"])
    latest = Path(cfg["paths"]["latest"])

    print(f"[trainer] serveur={server} | buffer={buffer_dir} | latest={latest}", flush=True)

    stop = threading.Event()
    threads = [
        threading.Thread(target=_sync_shards,
                         args=(server, buffer_dir, stop, args.sync_sec), daemon=True),
        threading.Thread(target=_publish_loop,
                         args=(server, args.token, latest, stop, args.publish_sec),
                         daemon=True),
    ]
    for t in threads:
        t.start()

    online = None if args.no_train else _start_online(args)
    try:
        while True:
            if online is not None and online.poll() is not None:
                print(f"[trainer] online.py s'est arrêté (code {online.returncode}).",
                      flush=True)
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[trainer] arrêt…", flush=True)
    finally:
        stop.set()
        if online is not None and online.poll() is None:
            online.terminate()
            try:
                online.wait(timeout=10)
            except subprocess.TimeoutExpired:
                online.kill()


if __name__ == "__main__":
    main()
