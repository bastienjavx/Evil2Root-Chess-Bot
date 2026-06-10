"""Chargement de config et gestion des checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


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
