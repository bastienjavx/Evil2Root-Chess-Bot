"""Dataset PyTorch : lit les shards de samples et encode les positions à la volée."""

from __future__ import annotations

import gzip
import random
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.samples import _parse_policy
from ..encoding import encode_board, legal_policy_mask
from .losses import dense_policy_target, moves_left_target


def find_shards(shards_dir: str | Path) -> list[Path]:
    return sorted(Path(shards_dir).glob("*.txt.gz"))


def split_batch(batch, mask_policy: bool, moves_left: bool):
    """Déballe un batch du DataLoader en (planes, policy, legal_mask|None,
    value, ml_target|None, ml_mask|None) quels que soient les flags actifs.
    Centralise l'ordre des tenseurs pour pretrain / online / distributed."""
    i = 0
    planes = batch[i]; i += 1
    policy = batch[i]; i += 1
    legal_mask = None
    if mask_policy:
        legal_mask = batch[i]; i += 1
    value = batch[i]; i += 1
    ml_target = ml_mask = None
    if moves_left:
        ml_target = batch[i]; i += 1
        ml_mask = batch[i]; i += 1
    return planes, policy, legal_mask, value, ml_target, ml_mask


class ShardDataset(Dataset):
    """Charge les lignes brutes en mémoire ; parse + encode dans __getitem__.

    Stockage : UN seul buffer d'octets contigu (toutes les lignes concaténées) +
    un tableau numpy d'offsets, au lieu d'une liste de millions de tuples Python.
    Raison : sous DataLoader multi-workers, chaque accès à un objet Python
    incrémente son refcount, ce qui modifie la page et la dé-partage du copy-on-write
    -> les workers recopient peu à peu tout le dataset et la RAM explose (×num_workers).
    Avec ~2 objets (le buffer + les offsets), il n'y a plus de churn de refcount :
    la mémoire reste partagée et stable. Bonus : empreinte de base plus faible (pas
    d'overhead par str/tuple/dict).

    Garde-fou OOM : si `max_samples` est fixé, on s'arrête une fois ce nombre de
    samples atteint. Les shards sont d'abord mélangés (graine fixe = reproductible)
    pour obtenir un échantillon représentatif de TOUS les mois, et pas seulement
    des premiers shards (sinon on ne verrait que les mois les plus récents).
    """

    def __init__(self, shard_paths: list[Path],
                 max_samples: int | None = None, seed: int = 0,
                 input_features: str = "base", mask_policy: bool = False,
                 moves_left: bool = False):
        self.input_features = input_features
        self.mask_policy = bool(mask_policy)
        self.moves_left = bool(moves_left)
        paths = list(shard_paths)
        if max_samples:
            random.Random(seed).shuffle(paths)
        buf = bytearray()
        offsets = [0]
        stop = False
        for p in paths:
            with gzip.open(p, "rb") as f:
                for line in f:
                    line = line.rstrip(b"\n")
                    if not line:
                        continue
                    buf += line
                    offsets.append(len(buf))
                    if max_samples and len(offsets) - 1 >= max_samples:
                        stop = True
                        break
            if stop:
                break
        if len(offsets) <= 1:
            raise RuntimeError("Aucun sample trouvé dans les shards fournis.")
        # frombuffer ne copie pas : la vue numpy partage les octets du bytearray.
        self.buf = np.frombuffer(buf, dtype=np.uint8)
        self.offsets = np.asarray(offsets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int):
        line = self.buf[self.offsets[i]:self.offsets[i + 1]].tobytes().decode("utf-8")
        parts = line.split("\t")
        fen, move_uci, value = parts[0], parts[1], int(parts[2])
        pi = _parse_policy(parts[3]) if len(parts) > 3 else None
        board = chess.Board(fen)
        planes = encode_board(board, self.input_features)
        # Cible politique DENSE (POLICY_SIZE) : distribution de visites si dispo
        # (self-play), sinon one-hot du coup joué (samples humains) -> CE souple
        # identique à l'ancienne CE dure dans ce cas.
        policy = dense_policy_target(board, move_uci, pi)
        out = [torch.from_numpy(planes), torch.from_numpy(policy)]
        if self.mask_policy:
            out.append(torch.from_numpy(legal_policy_mask(board)))
        out.append(torch.tensor(value, dtype=torch.float32))
        if self.moves_left:
            ml = int(parts[4]) if len(parts) > 4 and parts[4].strip() else None
            tgt, mask = moves_left_target(ml)
            out.append(torch.tensor(tgt, dtype=torch.float32))
            out.append(torch.tensor(mask, dtype=torch.float32))
        return tuple(out)
