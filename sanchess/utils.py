"""Chargement de config, sélection d'appareil et gestion des checkpoints."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch
import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


# --- Sélection d'appareil (CUDA / MPS Apple Silicon / CPU) --------------------

def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str | None = None) -> str:
    """Résout l'appareil à utiliser.

    "auto" (ou None) -> CUDA si dispo, sinon MPS (Mac M1/M2/M3), sinon CPU.
    Une valeur explicite ("cuda"/"mps"/"cpu") est respectée si possible, avec
    repli automatique vers CPU si l'accélérateur demandé est indisponible.
    """
    req = (requested or "auto").lower()
    if req in ("auto", ""):
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if req == "cuda":
        return "cuda" if torch.cuda.is_available() else ("mps" if _mps_available() else "cpu")
    if req == "mps":
        return "mps" if _mps_available() else "cpu"
    return "cpu"


def device_kind(device: str) -> str:
    """Famille de l'appareil ('cuda', 'mps' ou 'cpu') à partir d'un id Torch."""
    d = str(device)
    if d.startswith("cuda"):
        return "cuda"
    if d.startswith("mps"):
        return "mps"
    return "cpu"


def amp_enabled(device: str, want_amp: bool = True) -> bool:
    """L'AMP (mixed precision) n'est fiable/utile que sur CUDA ici."""
    return bool(want_amp) and device_kind(device) == "cuda"


def autocast_ctx(device: str, enabled: bool):
    """Contexte autocast adapté à l'appareil (no-op si désactivé)."""
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_kind(device), enabled=True)


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Charge un .env (KEY=VALUE par ligne) dans os.environ si présent.
    N'écrase pas une variable déjà définie."""
    path = Path(path) if path else (_DEFAULT_CONFIG.parent / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: str | os.PathLike, model, cfg: dict,
                    optimizer=None, step: int = 0) -> None:
    """Écrit le checkpoint de façon atomique (tmp + rename) pour le hot-reload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_cfg": cfg.get("model", {}),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_state(model, state: dict, strict: bool = False) -> None:
    """Charge des poids en tolérant les divergences d'architecture.

    Utile pour le hot-reload entre versions du réseau (ex. ajout de blocs SE) ou
    le warm-start lors d'un changement de tête (ex. valeur scalar -> wdl) : on
    charge ce qui correspond et on signale le reste plutôt que de planter.

    Note : `strict=False` ignore les clés manquantes/en trop, mais PAS les tenseurs
    présents de forme DIFFÉRENTE (PyTorch lève quand même). On filtre donc d'abord
    les poids dont la forme ne correspond pas au modèle courant.
    """
    own = model.state_dict()
    mismatched = [k for k, v in state.items()
                  if k in own and hasattr(v, "shape") and v.shape != own[k].shape]
    if mismatched:
        state = {k: v for k, v in state.items() if k not in mismatched}
    result = model.load_state_dict(state, strict=strict)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing or unexpected or mismatched:
        import sys
        sys.stderr.write(
            f"info string poids partiellement chargés "
            f"(manquants={len(missing)}, inattendus={len(unexpected)}, "
            f"formes incompatibles={len(mismatched)})\n")
    # Bilan : permet à l'appelant de savoir si l'architecture est IDENTIQUE
    # (tout à 0) et donc s'il est sûr de réutiliser l'état de l'optimiseur.
    return {"missing": len(missing), "unexpected": len(unexpected),
            "mismatched": len(mismatched)}
