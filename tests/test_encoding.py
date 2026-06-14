"""Tests de l'encodage : round-trip coup<->index et forme des plans.

Exécuter :  python -m tests.test_encoding   (ou pytest)
"""

from __future__ import annotations

import random

import chess

from sanchess.encoding import (INPUT_PLANES, POLICY_SIZE, TACTICAL_INPUT_PLANES,
                               encode_board, index_to_move, legal_move_indices,
                               legal_policy_mask, move_to_index)


def _random_positions(n: int, max_plies: int = 60) -> list[chess.Board]:
    boards, rng = [], random.Random(42)
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randint(0, max_plies)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        boards.append(b)
    return boards


def test_move_index_roundtrip():
    """Pour chaque coup légal : index_to_move(move_to_index(m)) == m."""
    for board in _random_positions(200):
        for move in board.legal_moves:
            idx = move_to_index(move, board.turn)
            assert 0 <= idx < 4672, idx
            back = index_to_move(idx, board)
            assert back == move, (board.fen(), move.uci(), back.uci(), idx)


def test_indices_unique_and_legal():
    """Les indices des coups légaux d'une position sont tous distincts."""
    for board in _random_positions(200):
        moves, idx = legal_move_indices(board)
        assert len(set(idx.tolist())) == len(moves), board.fen()


def test_legal_policy_mask_matches_legal_indices():
    for board in _random_positions(50):
        _, idx = legal_move_indices(board)
        mask = legal_policy_mask(board)
        assert mask.shape == (POLICY_SIZE,)
        assert mask.dtype.name == "bool"
        assert mask.sum() == len(idx)
        assert mask[idx].all()


def test_encode_shape():
    for board in _random_positions(20):
        planes = encode_board(board)
        assert planes.shape == (INPUT_PLANES, 8, 8)
        assert planes.dtype.name == "float32"
        tactical = encode_board(board, "tactical")
        assert tactical.shape == (TACTICAL_INPUT_PLANES, 8, 8)
        assert tactical.dtype.name == "float32"


if __name__ == "__main__":
    test_move_index_roundtrip()
    test_indices_unique_and_legal()
    test_legal_policy_mask_matches_legal_indices()
    test_encode_shape()
    print("OK — tous les tests d'encodage passent.")
