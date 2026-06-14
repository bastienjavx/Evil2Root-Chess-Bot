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


def select_child(node: "_Node", c_puct: float, fpu: float):
    """Sélection PUCT d'un fils (fonction libre, partagée par le MCTS séquentiel
    et le self-play batché GPU pour garantir une sélection IDENTIQUE).

    Q est exprimé du point de vue du PARENT (= -Q(fils)). Les pertes virtuelles
    (`vloss`) en vol comptent comme des défaites vues du parent -> découragent de
    re-descendre le même chemin, ce qui REMPLIT le lot GPU au lieu de le vider.

    Renvoie None UNIQUEMENT si le noeud n'a aucun fils (position sans coup légal :
    l'appelant doit alors traiter ce noeud comme terminal). Un noeud développé a
    toujours des fils -> on garantit de renvoyer un (move, child) même si tous les
    scores sont non finis (priors/valeurs NaN d'un checkpoint en cours d'écriture
    ou divergé) : on conserve un repli plutôt que de planter à l'unpack.
    """
    if not node.children:
        return None
    total = node.N + node.vloss
    sqrt_n = math.sqrt(total) if total > 0 else 1.0
    fpu_base = node.q() - fpu          # valeur optimiste d'un fils non visité
    best_score, best = -math.inf, None
    for move, child in node.children.items():
        n_eff = child.N + child.vloss
        if n_eff > 0:
            q = -(child.W + child.vloss) / n_eff
        else:
            q = fpu_base
        prior = child.prior if math.isfinite(child.prior) else 0.0
        u = c_puct * prior * sqrt_n / (1 + n_eff)
        score = q + u
        if not math.isfinite(score):
            score = -math.inf       # NaN/inf : repli (les fils sains gagnent)
        if best is None or score > best_score:
            best_score, best = score, (move, child)
    return best


class Evaluator:
    """Encapsule le réseau pour évaluer un lot de positions."""

    def __init__(self, model, device: str = "cuda"):
        self.device = device
        self._cuda = str(device).startswith("cuda")
        # channels_last (NHWC) : sur tensor cores (Ampere/Ada/Hopper) les conv 2D
        # tournent nettement plus vite dans ce format. Mesuré H100 PCIe sur ce
        # réseau (24x320, SE) : +46 % de débit forward (26k -> 38k pos/s). C'est le
        # plafond GPU qui borne le self-play batché dès qu'on a plusieurs workers,
        # donc le gain se répercute directement sur les parties/seconde.
        if self._cuda:
            model = model.to(memory_format=torch.channels_last)
            # TF32 pour les matmuls résiduelles hors autocast (gratuit sur Hopper).
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = model.to(device).eval()
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
        use_cuda = self._cuda
        if use_cuda:
            t = t.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                            enabled=use_cuda):
            logits, values = self.model(t)
        # Un checkpoint en cours d'écriture / divergé peut sortir des NaN/inf :
        # on les neutralise ici (logits -> 0 => priors uniformes, value -> 0)
        # pour que le self-play continue au lieu de propager des NaN dans l'arbre.
        logits = np.nan_to_num(logits.float().cpu().numpy(), nan=0.0,
                               posinf=0.0, neginf=0.0)
        # Tête WDL (B,3) -> scalaire E[résultat] = P(victoire) - P(défaite) ∈ [-1,1].
        # Tête scalaire (B,1) -> tel quel. (Détection par la forme : robuste.)
        if values.dim() == 2 and values.shape[1] == 3:
            p = torch.softmax(values.float(), dim=1).cpu().numpy()
            values = p[:, 2] - p[:, 0]
        else:
            values = values.float().cpu().numpy().reshape(-1)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

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
        return select_child(node, self.c_puct, self.fpu)

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

    # --- Réutilisation d'arbre entre coups ------------------------------------
    def advance_root(self, root: "_Node | None", board_before: chess.Board,
                     board_after: chess.Board, max_depth: int = 2) -> "_Node | None":
        """Retrouve, dans l'arbre `root` calculé à `board_before`, le sous-arbre
        correspondant à la position `board_after` (atteinte en `max_depth` demi-coups
        au plus : notre coup + la réponse adverse). Le renvoie pour servir de
        racine à la recherche suivante -> on capitalise les visites déjà faites,
        au lieu de repartir d'un arbre vide (gain de force GRATUIT à temps égal).

        Renvoie None si l'arbre est absent/non développé ou si la position cible
        n'a pas été explorée (on repartira alors d'une racine fraîche). On compare
        l'EPD (FEN sans compteurs) pour tolérer les transpositions de compteurs
        tout en garantissant une position strictement identique.
        """
        if root is None or not root.expanded:
            return None
        target = board_after._transposition_key()
        # DFS borné sur les fils RÉELLEMENT visités (les autres n'ont pas de stats
        # à réutiliser). Profondeur 0 = board_before lui-même (cas sans coup joué).
        stack = [(root, board_before, 0)]
        while stack:
            node, b, depth = stack.pop()
            if b._transposition_key() == target:
                return node if node.expanded else None
            if depth >= max_depth:
                continue
            for move, child in node.children.items():
                if child.N == 0 and not child.expanded:
                    continue                      # fils jamais visité : rien à reprendre
                nb = b.copy(stack=False)
                nb.push(move)
                stack.append((child, nb, depth + 1))
        return None

    # --- Recherche ------------------------------------------------------------
    def run(self, board: chess.Board, num_nodes: int,
            max_seconds: float | None = None, *,
            stop_event=None, root: "_Node | None" = None) -> _Node:
        """Lance la recherche et renvoie la racine.

        - `stop_event` (threading.Event) : si fourni, la recherche s'arrête
          proprement dès qu'il est positionné (sert au pondering : on interrompt
          la réflexion de fond dès que l'adversaire a joué pour son vrai coup).
        - `root` : si fourni et déjà développé, on REPREND cet arbre au lieu d'en
          repartir d'un neuf (ponder hit : on capitalise les visites déjà faites
          pendant le tour de l'adversaire). Sinon on construit une racine fraîche.
        """
        if root is None or not root.expanded:
            root = _Node(0.0)
            root_priors, _ = self.ev.evaluate([board])[0]
            self._expand(root, root_priors)
            if not root.children:    # position terminale : aucun coup à chercher
                return root
            self._add_dirichlet(root)
        # else : on reprend une racine déjà évaluée (ses stats N/W sont conservées).

        deadline = time.monotonic() + max_seconds if max_seconds else None
        stopped = (lambda: stop_event is not None and stop_event.is_set())
        simulations = 0
        while (simulations < num_nodes
               and (deadline is None or time.monotonic() < deadline)
               and not stopped()):
            leaves = []          # (leaf_node, path, leaf_board)
            pending = set()

            while (len(leaves) < self.batch_size
                   and simulations + len(leaves) < num_nodes
                   and not stopped()):
                node = root
                path = [node]
                b = board.copy()
                # Descente PUCT jusqu'à une feuille (non expandée ou terminale).
                while node.expanded and not node.terminal:
                    sel = self._select_child(node)
                    if sel is None:              # noeud développé sans fils légal
                        node.terminal = True     # -> feuille terminale
                        break
                    move, node = sel
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
