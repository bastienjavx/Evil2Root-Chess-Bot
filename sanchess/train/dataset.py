"""Dataset PyTorch : lit les shards de samples et encode les positions à la volée."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.samples import iter_samples
from ..encoding import encode_board, move_to_index


def find_shards(shards_dir: str | Path) -> list[Path]:
    return sorted(Path(shards_dir).glob("*.txt.gz"))


class ShardDataset(Dataset):
    """Charge les lignes (fen, move, value) en mémoire ; encode dans __getitem__."""

    def __init__(self, shard_paths: list[Path]):
        self.rows: list[tuple[str, str, int]] = []
        for p in shard_paths:
            self.rows.extend(iter_samples(p))
        if not self.rows:
            raise RuntimeError("Aucun sample trouvé dans les shards fournis.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        fen, move_uci, value = self.rows[i]
        board = chess.Board(fen)
        planes = encode_board(board)
        idx = move_to_index(chess.Move.from_uci(move_uci), board.turn)
        return (torch.from_numpy(planes),
                torch.tensor(idx, dtype=torch.long),
                torch.tensor(value, dtype=torch.float32))
