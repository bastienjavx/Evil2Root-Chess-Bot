"""Dataset PyTorch : lit les shards de samples et encode les positions à la volée."""

from __future__ import annotations

import random
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.samples import iter_samples_pi
from ..encoding import encode_board
from .losses import dense_policy_target


def find_shards(shards_dir: str | Path) -> list[Path]:
    return sorted(Path(shards_dir).glob("*.txt.gz"))


class ShardDataset(Dataset):
    """Charge les lignes (fen, move, value) en mémoire ; encode dans __getitem__.

    Garde-fou OOM : si `max_samples` est fixé, on s'arrête une fois ce nombre de
    samples atteint. Les shards sont d'abord mélangés (graine fixe = reproductible)
    pour obtenir un échantillon représentatif de TOUS les mois, et pas seulement
    des premiers shards (sinon on ne verrait que les mois les plus récents).
    """

    def __init__(self, shard_paths: list[Path],
                 max_samples: int | None = None, seed: int = 0):
        paths = list(shard_paths)
        if max_samples:
            random.Random(seed).shuffle(paths)
        self.rows: list[tuple] = []
        for p in paths:
            self.rows.extend(iter_samples_pi(p))
            if max_samples and len(self.rows) >= max_samples:
                del self.rows[max_samples:]
                break
        if not self.rows:
            raise RuntimeError("Aucun sample trouvé dans les shards fournis.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        fen, move_uci, value, pi = self.rows[i]
        board = chess.Board(fen)
        planes = encode_board(board)
        # Cible politique DENSE (POLICY_SIZE) : distribution de visites si dispo
        # (self-play), sinon one-hot du coup joué (samples humains) -> CE souple
        # identique à l'ancienne CE dure dans ce cas.
        policy = dense_policy_target(board, move_uci, pi)
        return (torch.from_numpy(planes),
                torch.from_numpy(policy),
                torch.tensor(value, dtype=torch.float32))
