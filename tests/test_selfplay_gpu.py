"""Self-play batché (`train/selfplay_gpu.py`), exécuté sur CPU avec un réseau
minuscule : on teste la MÉCANIQUE (parties qui se terminent, samples au bon
format, batching inter-parties), pas la force de jeu.
"""
from __future__ import annotations

import chess

from sanchess.model import SanChessNet
from sanchess.search.mcts import Evaluator
from sanchess.train.selfplay_gpu import BatchedSelfPlay


def _evaluator():
    model = SanChessNet(channels=8, blocks=1, policy_head="conv", value_head="wdl")
    return Evaluator(model, "cpu")


def _search_cfg(nodes=24, max_plies=30):
    return {
        "c_puct": 1.5, "fpu": 0.2, "nodes": nodes, "max_plies": max_plies,
        "temp_moves": 4, "temperature": 1.0,
        "dirichlet_eps": 0.25, "dirichlet_alpha": 0.3,
    }


def _valid_samples(rows):
    n = len(rows)
    for i, (fen, uci, value, pi, plies_to_end) in enumerate(rows):
        chess.Board(fen)                       # FEN parsable
        chess.Move.from_uci(uci)               # coup UCI valide
        assert value in (-1, 0, 1)
        assert isinstance(pi, dict) and pi     # distribution de visites non vide
        assert all(int(w) >= 0 for w in pi.values())
        # plies_to_end décroît de 1 à chaque position (cible moves-left).
        assert plies_to_end == n - i


def test_batched_selfplay_completes_games():
    sp = BatchedSelfPlay(_evaluator(), _search_cfg(nodes=16, max_plies=24),
                         n_games=6, leaves_per_game=4, seed=1)
    collected = []
    for _ in range(4000):                      # borne dure anti-boucle
        for rows in sp.step():
            collected.append(rows)
        if len(collected) >= 3:
            break
    assert len(collected) >= 3, "des parties doivent se terminer"
    for rows in collected:
        assert rows, "une partie terminée produit des samples"
        _valid_samples(rows)


def test_value_sign_alternates_along_game():
    """La valeur cible alterne de signe entre demi-coups consécutifs (résultat vu
    du trait) : si la partie n'est pas nulle, deux positions successives ont des
    valeurs opposées."""
    sp = BatchedSelfPlay(_evaluator(), _search_cfg(nodes=16, max_plies=24),
                         n_games=4, leaves_per_game=4, seed=2)
    game = None
    for _ in range(4000):
        for rows in sp.step():
            if rows and rows[0][2] != 0:       # première partie non nulle
                game = rows
                break
        if game:
            break
    if game is None:
        return                                 # toutes nulles (rare) : rien à vérifier
    values = [r[2] for r in game]
    for a, b in zip(values, values[1:]):
        assert a == -b, "les valeurs doivent alterner de signe (POV trait)"


def test_games_are_replaced_after_finish():
    sp = BatchedSelfPlay(_evaluator(), _search_cfg(nodes=12, max_plies=16),
                         n_games=3, leaves_per_game=4, seed=3)
    before = [id(g) for g in sp.games]
    finished_any = False
    for _ in range(4000):
        out = sp.step()
        if out:
            finished_any = True
            break
    assert finished_any
    # Au moins une partie a été remplacée par une nouvelle instance.
    after = [id(g) for g in sp.games]
    assert any(a != b for a, b in zip(before, after))
    assert sp.finished_games >= 1
