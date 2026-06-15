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
import math
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


# --- Dimensionnement local ----------------------------------------------------

_AUTO = "auto"


def _auto_default(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default)


def _parse_positive_int_or_auto(value, *, name: str) -> int | None:
    """Parse une option positive ou 'auto'. Retourne None pour auto."""
    if isinstance(value, int):
        n = value
    else:
        text = str(value or _AUTO).strip().lower()
        if text in ("", _AUTO):
            return None
        try:
            n = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} doit être un entier positif ou 'auto'"
            ) from exc
    if n < 1:
        raise argparse.ArgumentTypeError(f"{name} doit être >= 1")
    return n


def _resolve_workers(value, reserve_cores: int, cpu_count: int | None = None,
                     *, cuda: bool = False) -> int:
    explicit = _parse_positive_int_or_auto(value, name="workers")
    if explicit is not None:
        return explicit
    if cuda:
        # Sur GPU, un seul process suffit : BatchedSelfPlay batche déjà TOUTES les
        # parties en une seule éval NN. Lancer cpu_count process CPU-bound
        # affamerait le GPU (chacun ne batchant qu'1/N des parties, micro-batches)
        # et dupliquerait le modèle en VRAM jusqu'à l'OOM. On veut 1 process qui
        # sature le GPU via --gpu-games. Override possible avec --workers N.
        return 1
    ncpu = int(cpu_count or os.cpu_count() or 1)
    reserve = max(0, int(reserve_cores))
    return max(1, ncpu - reserve)


def _resolve_threads(value) -> int:
    explicit = _parse_positive_int_or_auto(value, name="threads")
    return explicit if explicit is not None else 1


def _resolve_optional_positive(value, *, name: str) -> int | None:
    return _parse_positive_int_or_auto(value, name=name)


def _split_gpu_games(total_games: int, workers: int) -> int:
    return max(1, int(math.ceil(max(1, total_games) / max(1, workers))))


def _configure_thread_env() -> None:
    """Un process worker = un cœur Python ; éviter les pools BLAS par process."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _apply_local_job_overrides(job: P.JobSpec, args, kind: str) -> P.JobSpec:
    if kind == "cuda":
        total_games = int(args.gpu_games_total or job.gpu_games)
        job.gpu_games = _split_gpu_games(total_games, int(args.workers))
        if args.gpu_leaves_per_game:
            job.gpu_leaves_per_game = int(args.gpu_leaves_per_game)
    return job


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
                print("[worker] aucun modèle publié sur le serveur ; "
                      "initialisation d'un réseau frais local", flush=True)
                self.model = build_model(self.fallback_cfg)
                self.evaluator = Evaluator(self.model, self.device)
                self.version = 0
            return info
        if info.version == self.version and self.model is not None:
            return info                            # déjà à jour

        path = self.cache / f"weights_v{info.version}.pt"
        cached = path.exists() and _sha256_file(path) == info.sha256
        if not cached:
            print(f"[worker] téléchargement du modèle v{info.version} "
                  f"({info.size // 1024} Ko)", flush=True)
            r = requests.get(self.server + P.EP_MODEL_DOWNLOAD, timeout=120)
            r.raise_for_status()
            if info.sha256 and hashlib.sha256(r.content).hexdigest() != info.sha256:
                raise RuntimeError("sha256 du modèle téléchargé ne correspond pas")
            tmp = path.with_suffix(".pt.part")
            tmp.write_bytes(r.content)
            os.replace(tmp, path)
        else:
            print(f"[worker] modèle v{info.version} trouvé dans le cache local",
                  flush=True)

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
    torch.set_num_threads(max(1, int(args.threads)))
    if kind in ("cpu", "mps"):
        try:
            os.nice(int(args.nice))                # priorité basse : ne fige pas le poste
        except (OSError, AttributeError):
            pass

    server = args.server.rstrip("/")
    sync = ModelSync(server, _cache_dir(), cfg, device)
    print(f"[worker {wid[:8]}] device={device} ({kind}) serveur={server} "
          f"pseudo={args.name!r} | puissance={args.power_profile} "
          f"workers={args.workers} threads={args.threads} "
          f"reserve_cores={args.reserve_cores}", flush=True)

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
            job = _apply_local_job_overrides(job, args, kind)

            # 2. modèle à jour.
            sync.ensure_latest()

            # 3. jouer.
            t0 = time.time()
            if kind == "cuda":
                print(f"[worker {wid[:8]}] job: nodes={job.nodes} "
                      f"gpu_games={job.gpu_games} leaves={job.gpu_leaves_per_game} "
                      f"target_samples={job.target_samples}", flush=True)
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
    ap.add_argument("--device", default=os.environ.get("SANO1_WORKER_DEVICE", "auto"),
                    help="auto / cuda / mps / cpu")
    ap.add_argument("--config", default=None, help="config.yaml (mcts/model fallback)")
    ap.add_argument("--workers", default=_auto_default("SANO1_WORKER_WORKERS", _AUTO),
                    help="'auto' utilise presque tous les cœurs, ou un entier explicite")
    ap.add_argument("--threads", default=_auto_default("SANO1_WORKER_THREADS", _AUTO),
                    help="threads torch par process ('auto' = 1)")
    ap.add_argument("--reserve-cores", type=int,
                    default=int(os.environ.get("SANO1_WORKER_RESERVE_CORES", "1")),
                    dest="reserve_cores",
                    help="cœurs gardés libres quand --workers=auto")
    ap.add_argument("--gpu-games", default=_auto_default("SANO1_WORKER_GPU_GAMES", _AUTO),
                    dest="gpu_games",
                    help="parties GPU batchées au total sur cette machine ('auto' = job serveur)")
    ap.add_argument("--gpu-leaves", default=_auto_default("SANO1_WORKER_GPU_LEAVES", _AUTO),
                    dest="gpu_leaves",
                    help="feuilles GPU par partie ('auto' = job serveur)")
    ap.add_argument("--nice", type=int,
                    default=int(os.environ.get("SANO1_WORKER_NICE", "10")),
                    help="priorité CPU/MPS (anti-freeze)")
    ap.add_argument("--once", action="store_true",
                    help="un seul lot puis arrêt (debug / test)")
    args = ap.parse_args()

    args.power_profile = "auto-agressif"
    # Le matériel décide le dimensionnement auto : sur CUDA, --workers auto = 1
    # process qui batche tout sur le GPU (sinon le GPU est affamé, cf. _resolve_workers).
    is_cuda = device_kind(resolve_device(args.device)) == "cuda"
    try:
        args.workers = _resolve_workers(args.workers, args.reserve_cores, cuda=is_cuda)
        args.threads = _resolve_threads(args.threads)
        args.gpu_games_total = _resolve_optional_positive(args.gpu_games,
                                                          name="gpu-games")
        args.gpu_leaves_per_game = _resolve_optional_positive(args.gpu_leaves,
                                                              name="gpu-leaves")
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
    _configure_thread_env()

    print("[worker] configuration locale : "
          f"workers={args.workers} threads={args.threads} "
          f"reserve_cores={args.reserve_cores} "
          f"gpu_games={args.gpu_games_total or 'serveur'} "
          f"gpu_leaves={args.gpu_leaves_per_game or 'serveur'}",
          flush=True)

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
