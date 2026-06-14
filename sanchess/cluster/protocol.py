"""Contrat partagé entre le coordinateur, les workers et le trainer.

Volontairement SANS dépendance lourde (pas de torch, pas de fastapi) : ce module est
importé aussi bien par le serveur Railway que par le client bénévole. On reste sur des
`dataclasses` + conversions dict <-> JSON, source unique de vérité du format des routes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

# --- Endpoints (préfixe commun) ----------------------------------------------
API_PREFIX = "/cluster"
EP_MODEL_CURRENT = f"{API_PREFIX}/model/current"
EP_MODEL_DOWNLOAD = f"{API_PREFIX}/model/download"
EP_MODEL_PUBLISH = f"{API_PREFIX}/model/publish"
EP_WORK = f"{API_PREFIX}/work"
EP_UPLOAD = f"{API_PREFIX}/upload"
EP_SHARDS = f"{API_PREFIX}/shards"
EP_STATS = f"{API_PREFIX}/stats"

# --- Noms de fichiers / agencement du Volume ---------------------------------
MODEL_FILENAME = "weights.pt"        # checkpoint weights-only servi aux workers
MODEL_META_FILENAME = "weights.json"  # métadonnées (version, step, sha256, size)
INCOMING_DIRNAME = "incoming"        # shards reçus des workers (buffer glissant)
DB_FILENAME = "pool.sqlite"          # registre workers + leaderboard + compteurs

# --- Garde-fous (valeurs par défaut ; surchargées par env côté serveur) ------
DEFAULT_MAX_UPLOAD_BYTES = 16 * 1024 * 1024   # taille max d'un shard gzip (compressé)
DEFAULT_MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024  # garde-fou anti zip-bomb
DEFAULT_MAX_SHARD_LINES = 200_000             # lignes max par shard
DEFAULT_VALIDATE_SAMPLE = 64                  # nb de lignes vérifiées (FEN+coup légal)
DEFAULT_MAX_SHARDS = 4000                     # rotation du buffer (plus vieux supprimés)
DEFAULT_UPLOADS_PER_MIN = 120                 # rate-limit par IP


def _coerce(cls, data: dict):
    """Construit une dataclass en ignorant les clés inconnues (compat ascendante)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class JobSpec:
    """Paramètres d'un lot de self-play remis à un worker. `mode` = "cpu" (une partie
    par process via play_game) ou "gpu" (batché via BatchedSelfPlay) ; le worker choisit
    selon son device mais le serveur peut suggérer des budgets différents."""
    model_version: int = 0
    nodes: int = 160
    max_plies: int = 250
    temp_moves: int = 20
    temperature: float = 1.0
    dirichlet_eps: float = 0.25
    dirichlet_alpha: float = 0.3
    c_puct: float = 1.5
    fpu: float = 0.2
    # Combien de parties grouper par batch GPU (mode gpu) et combien de feuilles/partie.
    gpu_games: int = 64
    gpu_leaves_per_game: int = 8
    # Le worker accumule au moins ce nb de samples avant d'uploader un shard.
    target_samples: int = 2000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "JobSpec":
        return _coerce(cls, data or {})


@dataclass
class ModelInfo:
    """Métadonnées du modèle courant. `version` s'incrémente à chaque publication ;
    le worker re-télécharge le checkpoint dès qu'elle change. `sha256` permet de
    vérifier l'intégrité du téléchargement."""
    version: int = 0
    step: int = 0
    sha256: str = ""
    size: int = 0
    has_model: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        return _coerce(cls, data or {})


@dataclass
class UploadAck:
    """Réponse à un upload accepté."""
    ok: bool = True
    accepted_samples: int = 0
    shard: str = ""
    total_samples: int = 0       # cumul du contributeur
    rank: int = 0                # position au leaderboard

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Stats:
    """Instantané exposé par /cluster/stats et le dashboard."""
    model_version: int = 0
    model_step: int = 0
    has_model: bool = False
    total_games: int = 0
    total_samples: int = 0
    active_workers: int = 0
    games_per_min: float = 0.0
    pending_shards: int = 0
    leaderboard: list = field(default_factory=list)  # [{name, games, samples}]

    def to_dict(self) -> dict:
        return asdict(self)
