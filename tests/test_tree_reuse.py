"""Réutilisation d'arbre MCTS entre coups (`MCTS.advance_root`).

On vérifie que le sous-arbre retrouvé est BIEN celui de la position cible, que les
visites déjà faites sont conservées (la racine réutilisée porte ses stats), et que
les cas « position absente » / « partie différente » retombent proprement sur None.
Réseau aléatoire minuscule (CPU) : on teste la LOGIQUE d'arbre, pas la force.
"""
from __future__ import annotations

import chess

from sanchess.model import SanChessNet
from sanchess.search.mcts import MCTS, Evaluator, best_move


def _mcts(nodes_cfg=None):
    # Tout petit réseau aléatoire : rapide, suffisant pour exercer l'arbre.
    model = SanChessNet(channels=8, blocks=1, policy_head="conv", value_head="wdl")
    cfg = {"mcts": {"c_puct": 1.5, "eval_batch_size": 8, "fpu": 0.2,
                    "dirichlet_eps": 0.0}}
    return MCTS(Evaluator(model, "cpu"), cfg)


def test_advance_root_finds_subtree_after_two_plies():
    mcts = _mcts()
    board = chess.Board()
    root = mcts.run(board, 200)

    # Joue le meilleur coup, puis une réponse adverse plausible (un coup visité).
    our = best_move(root)
    assert our is not None
    child = root.children[our]
    # Une réponse adverse qui a été explorée (N>0) pour pouvoir être retrouvée.
    after_our = board.copy(); after_our.push(our)

    reply = best_move(child) if child.expanded else None
    assert reply is not None, "le fils de notre coup doit être développé"

    new_board = after_our.copy(); new_board.push(reply)

    reused = mcts.advance_root(root, board, new_board)
    assert reused is not None, "le sous-arbre de la position atteinte doit être retrouvé"
    assert reused is child.children[reply], "doit pointer EXACTEMENT sur ce nœud"
    assert reused.N > 0, "les visites déjà faites sont conservées"


def test_advance_root_reuse_accumulates_visits():
    mcts = _mcts()
    board = chess.Board()
    root = mcts.run(board, 200)
    our = best_move(root)
    after = board.copy(); after.push(our)
    reply = best_move(root.children[our])
    new_board = after.copy(); new_board.push(reply)

    reused = mcts.advance_root(root, board, new_board)
    n_before = reused.N
    # Reprendre la recherche sur la racine réutilisée doit AJOUTER des visites,
    # pas repartir de zéro.
    root2 = mcts.run(new_board, n_before + 50, root=reused)
    assert root2 is reused
    assert root2.N >= n_before + 40


def test_advance_root_none_when_position_absent():
    mcts = _mcts()
    board = chess.Board()
    root = mcts.run(board, 60)
    # Une position d'une AUTRE partie, hors de l'arbre -> pas de réutilisation.
    foreign = chess.Board()
    foreign.push_san("e4"); foreign.push_san("c5"); foreign.push_san("Nf3")
    foreign.push_san("d6"); foreign.push_san("d4")  # 5 demi-coups : hors profondeur 2
    assert mcts.advance_root(root, board, foreign) is None


def test_advance_root_none_for_unexpanded_or_missing():
    mcts = _mcts()
    board = chess.Board()
    assert mcts.advance_root(None, board, board) is None
    fresh = mcts.run(board, 0)  # racine éventuellement non développée
    # Position identique : profondeur 0 -> renvoie la racine si développée, sinon None.
    res = mcts.advance_root(fresh, board, board)
    assert res is None or res is fresh
