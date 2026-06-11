"""MCTS PUCT guidé par le réseau, avec évaluation batchée sur GPU.

Convention de valeur : chaque noeud stocke W et N dans le repère du joueur AU
TRAIT à ce noeud. La valeur d'un fils, vue du parent, est donc -Q(fils).
Backup : on remonte le chemin en alternant le signe de la valeur à chaque ply.
"""

from __future__ import annotations

import math
import time

import chess
import numpy as np
import torch

from ..encoding import encode_board, move_to_index


class _Node:
    __slots__ = ("prior", "N", "W", "vloss", "children", "expanded", "terminal")

    def __init__(self, prior: float):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.vloss = 0            # pertes virtuelles en vol (batch GPU), hors stats
        self.children: dict[chess.Move, _Node] = {}
        self.expanded = False
        self.terminal = False

    def q(self) -> float:
        # Q « propre » (sans pertes virtuelles) : sert au FPU et aux stats finales.
        return self.W / self.N if self.N > 0 else 0.0


class Evaluator:
    """Encapsule le réseau pour évaluer un lot de positions."""

    def __init__(self, model, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device
        # NB : on n'active PAS torch.backends.cudnn.benchmark. Les formes d'entrée
        # NE sont PAS fixes (batch 1 à la racine, jusqu'à eval_batch_size, plus un
        # dernier lot partiel de taille variable) -> cuDNN ré-autotune à chaque
        # nouvelle forme, avec un pic de latence non interruptible par le deadline.
        # En cadence rapide ça peut faire tomber le drapeau ; le gain est nul ici.

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]):
        """Retourne une liste de (priors {move: prob}, value [-1,1] côté trait)."""
        x = np.stack([encode_board(b) for b in boards])
        t = torch.from_numpy(x).to(self.device)
        use_cuda = str(self.device).startswith("cuda")
        with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                            enabled=use_cuda):
            logits, values = self.model(t)
        logits = logits.float().cpu().numpy()
        # Tête WDL (B,3) -> scalaire E[résultat] = P(victoire) - P(défaite) ∈ [-1,1].
        # Tête scalaire (B,1) -> tel quel. (Détection par la forme : robuste.)
        if values.dim() == 2 and values.shape[1] == 3:
            p = torch.softmax(values.float(), dim=1).cpu().numpy()
            values = p[:, 2] - p[:, 0]
        else:
            values = values.float().cpu().numpy().reshape(-1)

        out = []
        for i, board in enumerate(boards):
            moves = list(board.legal_moves)
            if not moves:
                out.append(({}, float(values[i])))
                continue
            idx = np.fromiter((move_to_index(m, board.turn) for m in moves),
                              dtype=np.int64, count=len(moves))
            ml = logits[i, idx]
            ml -= ml.max()                       # stabilité numérique
            probs = np.exp(ml)
            probs /= probs.sum()
            out.append((dict(zip(moves, probs)), float(values[i])))
        return out


def _terminal_value(board: chess.Board) -> float:
    """Valeur du point de vue du joueur au trait pour une position terminée."""
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


class MCTS:
    def __init__(self, evaluator: Evaluator, cfg: dict):
        m = cfg.get("mcts", {})
        self.ev = evaluator
        self.c_puct = m.get("c_puct", 1.5)
        self.batch_size = m.get("eval_batch_size", 16)
        self.fpu = m.get("fpu", 0.2)
        self.dir_alpha = m.get("dirichlet_alpha", 0.3)
        self.dir_eps = m.get("dirichlet_eps", 0.0)

    # --- Sélection PUCT -------------------------------------------------------
    def _select_child(self, node: _Node) -> tuple[chess.Move, _Node]:
        total = node.N + node.vloss
        sqrt_n = math.sqrt(total) if total > 0 else 1.0
        # FPU : un fils non encore visité vaut ~ la valeur du parent (côté parent =
        # node.q()) diminuée de `fpu`. Base PROPRE (sans pertes virtuelles) pour ne
        # pas biaiser l'exploration quand un chemin est en vol dans le lot GPU.
        fpu_base = node.q() - self.fpu
        best_score, best = -1e9, None
        for move, child in node.children.items():
            n_eff = child.N + child.vloss
            if n_eff > 0:
                # Q du point de vue du parent = -Q(fils). Chaque perte virtuelle
                # compte comme une victoire du fils (donc une perte vue du parent)
                # -> décourage de re-descendre le même chemin, ce qui REMPLIT le lot
                # GPU (sinon le batch se vide à 1 feuille et la recherche rampe).
                q = -(child.W + child.vloss) / n_eff
            else:
                q = fpu_base
            u = self.c_puct * child.prior * sqrt_n / (1 + n_eff)
            score = q + u
            if score > best_score:
                best_score, best = score, (move, child)
        return best

    def _expand(self, node: _Node, priors: dict):
        for move, p in priors.items():
            node.children[move] = _Node(p)
        node.expanded = True

    def _add_dirichlet(self, root: _Node):
        if self.dir_eps <= 0 or not root.children:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dir_alpha] * len(moves))
        for move, n in zip(moves, noise):
            c = root.children[move]
            c.prior = (1 - self.dir_eps) * c.prior + self.dir_eps * n

    # --- Recherche ------------------------------------------------------------
    def run(self, board: chess.Board, num_nodes: int,
            max_seconds: float | None = None) -> _Node:
        root = _Node(0.0)
        root_priors, _ = self.ev.evaluate([board])[0]
        self._expand(root, root_priors)
        if not root.children:        # position terminale : aucun coup à chercher
            return root
        self._add_dirichlet(root)

        deadline = time.monotonic() + max_seconds if max_seconds else None
        simulations = 0
        while simulations < num_nodes and (deadline is None or time.monotonic() < deadline):
            leaves = []          # (leaf_node, path, leaf_board)
            pending = set()

            while len(leaves) < self.batch_size and simulations + len(leaves) < num_nodes:
                node = root
                path = [node]
                b = board.copy()
                # Descente PUCT jusqu'à une feuille (non expandée ou terminale).
                while node.expanded and not node.terminal:
                    move, node = self._select_child(node)
                    b.push(move)
                    path.append(node)

                if id(node) in pending:
                    break        # éviter les doublons dans le lot
                # Terminal ? backup immédiat sans réseau.
                if b.is_game_over(claim_draw=True):
                    node.terminal = True
                    self._backup(path, _terminal_value(b))
                    simulations += 1
                    continue

                self._apply_virtual_loss(path)
                pending.add(id(node))
                leaves.append((node, path, b))

            if not leaves:
                if simulations < num_nodes:
                    continue
                break

            results = self.ev.evaluate([lb for _, _, lb in leaves])
            for (node, path, _), (priors, value) in zip(leaves, results):
                self._revert_virtual_loss(path)
                self._expand(node, priors)
                self._backup(path, value)
                simulations += 1

        return root

    # --- Backup / virtual loss ------------------------------------------------
    @staticmethod
    def _backup(path: list[_Node], leaf_value: float):
        value = leaf_value
        for node in reversed(path):
            node.N += 1
            node.W += value
            value = -value

    @staticmethod
    def _apply_virtual_loss(path: list[_Node]):
        for node in path:
            node.vloss += 1

    @staticmethod
    def _revert_virtual_loss(path: list[_Node]):
        for node in path:
            node.vloss -= 1


def best_move(root: _Node) -> chess.Move | None:
    """Coup le plus visité à la racine."""
    if not root.children:
        return None
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]
