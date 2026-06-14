"""Worker bénévole : joue du self-play pour le cluster San-o1.

Boucle : récupère le modèle courant du coordinateur -> demande un job -> joue des
parties contre lui-même (MCTS) -> renvoie les samples -> recommence. Le matériel est
détecté automatiquement (CUDA / MPS Apple Silicon / CPU) ; le chemin GPU utilise le
self-play BATCHÉ (BatchedSelfPlay) et le chemin CPU/MPS le `play_game` séquentiel —
exactement le code déjà éprouvé de train/selfplay*.py, on ne réimplémente rien.

Usage :
    python -m sanchess.cluster.worker --server https://<app>.up.railway.app --name "MonPseudo"
    python -m sanchess.cluster.worker --server http://localhost:8001 --device cpu --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tempfile
import time
import uuid
from pathlib import Path

import requests

from ..utils import load_config, resolve_device, device_kind
from . import protocol as P


def _cache_dir() -> Path:
    d = Path(os.environ.get("SANO1_CLUSTER_CACHE",
                            Path.home() / ".cache" / "sano1_cluster"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Synchronisation du modèle ------------------------------------------------

class ModelSync:
    """Maintient une copie locale à jour du modèle publié par le coordinateur.

    Re-télécharge uniquement quand la version change (le worker passe son temps à
    jouer, pas à transférer). Vérifie le sha256 pour écarter un transfert corrompu.
    """

    def __init__(self, server: str, cache: Path, fallback_cfg: dict, device: str):
        self.server = server.rstrip("/")
        self.cache = cache
        self.fallback_cfg = fallback_cfg
        self.device = device
        self.version = -1
        self.model = None
        self.evaluator = None

    def _current(self) -> P.ModelInfo:
        r = requests.get(self.server + P.EP_MODEL_CURRENT, timeout=30)
        r.raise_for_status()
        return P.ModelInfo.from_dict(r.json())

    def ensure_latest(self) -> P.ModelInfo:
        """Garantit que self.model correspond à la dernière version. Retourne l'info.
        Si aucun modèle n'est publié, construit un réseau frais (fallback config)."""
        import torch
        from ..model import build_model, build_model_from_checkpoint
        from ..utils import load_checkpoint, load_model_state
        from ..search.mcts import Evaluator

        info = self._current()
        if not info.has_model:
            if self.model is None:                 # rien à télécharger : réseau frais
                self.model = build_model(self.fallback_cfg)
                self.evaluator = Evaluator(self.model, self.device)
                self.version = 0
            return info
        if info.version == self.version and self.model is not None:
            return info                            # déjà à jour

        path = self.cache / f"weights_v{info.version}.pt"
        if not (path.exists() and _sha256_file(path) == info.sha256):
            r = requests.get(self.server + P.EP_MODEL_DOWNLOAD, timeout=120)
            r.raise_for_status()
            if info.sha256 and hashlib.sha256(r.content).hexdigest() != info.sha256:
                raise RuntimeError("sha256 du modèle téléchargé ne correspond pas")
            tmp = path.with_suffix(".pt.part")
            tmp.write_bytes(r.content)
            os.replace(tmp, path)

        ck = load_checkpoint(path, self.device)
        self.model = build_model_from_checkpoint(ck, fallback_cfg=self.fallback_cfg)
        load_model_state(self.model, ck.get("raw_state", ck["model_state"]))
        self.evaluator = Evaluator(self.model, self.device)
        self.version = info.version
        print(f"[worker] modèle v{info.version} chargé (step {info.step})", flush=True)
        return info


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Génération de parties (réutilise train/selfplay*.py) ---------------------

def _play_cpu(sync: ModelSync, job: P.JobSpec, cfg: dict) -> list[tuple]:
    """Chemin CPU/MPS : parties séquentielles via train/selfplay.play_game."""
    from ..search.mcts import MCTS
    from ..train.selfplay import play_game

    mcts = MCTS(sync.evaluator, cfg)
    mcts.dir_eps = float(job.dirichlet_eps)
    mcts.dir_alpha = float(job.dirichlet_alpha)
    rows: list[tuple] = []
    while len(rows) < job.target_samples:
        rows.extend(play_game(mcts, job.nodes, job.max_plies,
                              job.temp_moves, job.temperature))
    return rows


def _play_gpu(sync: ModelSync, job: P.JobSpec, cfg: dict) -> list[tuple]:
    """Chemin CUDA : self-play batché via train/selfplay_gpu.BatchedSelfPlay."""
    from ..train.selfplay_gpu import BatchedSelfPlay

    search_cfg = {
        "c_puct": job.c_puct, "fpu": job.fpu, "nodes": job.nodes,
        "max_plies": job.max_plies, "temp_moves": job.temp_moves,
        "temperature": job.temperature, "dirichlet_eps": job.dirichlet_eps,
        "dirichlet_alpha": job.dirichlet_alpha,
    }
    sp = BatchedSelfPlay(sync.evaluator, search_cfg, job.gpu_games,
                         job.gpu_leaves_per_game, seed=int(time.time()) & 0xFFFF)
    rows: list[tuple] = []
    while len(rows) < job.target_samples:
        for finished in sp.step():
            rows.extend(finished)
    return rows


def _upload(server: str, rows: list[tuple], worker_id: str, name: str) -> P.UploadAck:
    from ..data.samples import write_samples
    fd, path = tempfile.mkstemp(suffix=".txt.gz")
    os.close(fd)
    try:
        write_samples(path, rows)
        data = Path(path).read_bytes()
    finally:
        os.unlink(path)
    r = requests.post(server.rstrip("/") + P.EP_UPLOAD,
                      files={"shard": ("shard.txt.gz", data, "application/gzip")},
                      data={"worker_id": worker_id, "name": name}, timeout=120)
    r.raise_for_status()
    return P.UploadAck(**{k: v for k, v in r.json().items()
                          if k in P.UploadAck.__dataclass_fields__})


# --- Boucle worker ------------------------------------------------------------

def run_loop(args, wid: str) -> None:
    import torch
    cfg = load_config(args.config) if args.config else load_config()
    device = resolve_device(args.device)
    kind = device_kind(device)
    if kind == "cpu":
        torch.set_num_threads(max(1, int(args.threads)))
        try:
            os.nice(int(args.nice))                # priorité basse : ne fige pas le poste
        except (OSError, AttributeError):
            pass

    server = args.server.rstrip("/")
    sync = ModelSync(server, _cache_dir(), cfg, device)
    print(f"[worker {wid[:8]}] device={device} ({kind}) serveur={server} "
          f"pseudo={args.name!r}", flush=True)

    backoff = 2.0
    while True:
        try:
            # 1. job + enregistrement (renseigne le device pour le leaderboard).
            r = requests.post(server + P.EP_WORK,
                              json={"name": args.name, "device": kind, "worker_id": wid},
                              timeout=30)
            r.raise_for_status()
            payload = r.json()
            wid = payload.get("worker_id", wid)
            job = P.JobSpec.from_dict(payload.get("job", {}))

            # 2. modèle à jour.
            sync.ensure_latest()

            # 3. jouer.
            t0 = time.time()
            rows = _play_gpu(sync, job, cfg) if kind == "cuda" else _play_cpu(sync, job, cfg)

            # 4. renvoyer.
            ack = _upload(server, rows, wid, args.name)
            dt = time.time() - t0
            print(f"[worker {wid[:8]}] +{ack.accepted_samples} positions en {dt:.0f}s "
                  f"| total {ack.total_samples} | rang #{ack.rank} | modèle v{sync.version}",
                  flush=True)
            backoff = 2.0
            if args.once:
                return
        except KeyboardInterrupt:
            print(f"\n[worker {wid[:8]}] arrêt.", flush=True)
            return
        except Exception as exc:                   # réseau coupé, serveur down, etc.
            print(f"[worker {wid[:8]}] erreur : {exc} — nouvelle tentative dans "
                  f"{backoff:.0f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Worker bénévole San-o1 (self-play distribué).")
    ap.add_argument("--server", required=True, help="URL du coordinateur (Railway)")
    ap.add_argument("--name", default=os.environ.get("SANO1_WORKER_NAME", "anon"),
                    help="pseudo affiché au leaderboard")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--config", default=None, help="config.yaml (mcts/model fallback)")
    ap.add_argument("--workers", type=int, default=1,
                    help="processus de self-play parallèles (CPU). GPU : laisser 1 "
                         "(le batch interne parallélise déjà des dizaines de parties).")
    ap.add_argument("--threads", type=int, default=1,
                    help="threads torch par worker CPU (1 = un cœur/worker)")
    ap.add_argument("--nice", type=int, default=10, help="priorité CPU (anti-freeze)")
    ap.add_argument("--once", action="store_true",
                    help="un seul lot puis arrêt (debug / test)")
    args = ap.parse_args()

    if args.workers <= 1:
        run_loop(args, uuid.uuid4().hex)
        return

    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=run_loop, args=(args, uuid.uuid4().hex), daemon=False)
             for _ in range(args.workers)]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
