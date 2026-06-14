"""Self-play BATCHÉ sur GPU : la voie efficace pour générer des données RL.

Pourquoi ce module en plus de `selfplay.py` (CPU) ?
  Le self-play CPU joue UNE partie par process et évalue le réseau feuille par
  feuille : sur un gros GPU cloud (L40S/H100) le GPU passe son temps à attendre,
  on n'exploite quasiment rien. Ici on joue **N parties EN PARALLÈLE dans un seul
  process** et on regroupe TOUTES les feuilles à évaluer (de toutes les parties)
  dans UN SEUL forward GPU. Avec n_games × leaves_per_game ≈ quelques milliers de
  positions par batch, le GPU tourne à plein -> des dizaines de fois plus de
  parties/seconde. C'est ce débit qui alimente une vraie boucle d'amélioration RL.

Boucle d'auto-amélioration (cloud) :
    selfplay_gpu  ──écrit──>  data/replay_buffer/*.txt.gz  ──ingéré par──>  online.py
         ▲                                                                     │
         └──────────────── hot-reload checkpoints/latest.pt ◄─────────────────┘
Le réseau qui JOUE se recharge à chaud depuis latest.pt : le self-play profite
immédiatement des poids améliorés -> qualité des données qui monte avec le temps.

Format de samples IDENTIQUE à selfplay.py / stream.py :
    (fen, coup_joué_uci, valeur ∈ {-1,0,1} côté trait, pi = {uci: visites})
La cible politique DENSE (distribution de visites π) est déjà exploitée par
`losses.dense_policy_target` -> aucune modification du pipeline d'entraînement.

Algorithme = MCTS PUCT à PARALLÉLISATION DE FEUILLES inter-parties, avec pertes
virtuelles (mêmes conventions que search/mcts.py, dont `select_child` est partagé
pour garantir une sélection identique). Une feuille terminale est remontée sans
réseau ; les feuilles non terminales d'un même tour sont regroupées dans le batch.

Usage :
    python -m sanchess.train.selfplay_gpu --games 256 --nodes 200
    python -m sanchess.train.selfplay_gpu --config config.cloud.yaml
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import chess
import numpy as np

from ..data.samples import write_samples
from ..model import build_model
from ..search.mcts import _Node, _terminal_value, select_child, Evaluator
from ..utils import (load_checkpoint, load_config, load_model_state,
                     resolve_device)


def _gcfg(cfg: dict) -> dict:
    """Section selfplay_gpu (avec repli sur selfplay pour les clés communes)."""
    base = dict(cfg.get("selfplay", {}))
    base.update(cfg.get("selfplay_gpu", {}) or {})
    return base


# --- État d'une partie en cours ----------------------------------------------

class _Game:
    """Une partie self-play et son arbre MCTS, pilotée par le scheduler batché.

    Cycle par coup : (1) évaluer la racine (NEEDS_ROOT), (2) accumuler des
    simulations en soumettant des feuilles au batch GPU (SEARCHING), (3) une fois
    le budget de nœuds atteint, jouer un coup et boucler. Fin de partie -> samples
    finalisés (valeur = résultat réel du point de vue du trait à chaque position).
    """

    __slots__ = ("board", "root", "sims", "needs_root", "pending",
                 "_pending_ids", "history", "plies", "rng",
                 "c_puct", "fpu", "nodes", "max_plies", "temp_moves",
                 "temperature", "dir_eps", "dir_alpha", "done")

    def __init__(self, cfg_search: dict, rng: np.random.Generator):
        self.board = chess.Board()
        self.root: _Node | None = None
        self.sims = 0
        self.needs_root = True
        self.pending: list[tuple[_Node, list]] = []   # (feuille, chemin)
        self._pending_ids: set[int] = set()
        self.history: list[tuple[str, str, bool, dict]] = []  # (fen, uci, trait, pi)
        self.plies = 0
        self.done = False
        self.rng = rng
        self.c_puct = cfg_search["c_puct"]
        self.fpu = cfg_search["fpu"]
        self.nodes = cfg_search["nodes"]
        self.max_plies = cfg_search["max_plies"]
        self.temp_moves = cfg_search["temp_moves"]
        self.temperature = cfg_search["temperature"]
        self.dir_eps = cfg_search["dirichlet_eps"]
        self.dir_alpha = cfg_search["dirichlet_alpha"]

    # --- Phase de collecte (produit des positions à évaluer) -----------------
    def collect(self, max_leaves: int) -> list[chess.Board]:
        """Renvoie les positions à évaluer ce tour (la racine si NEEDS_ROOT, sinon
        jusqu'à `max_leaves` feuilles). Applique les pertes virtuelles ; remonte
        immédiatement les feuilles terminales (sans réseau)."""
        if self.done:
            return []
        if self.needs_root:
            return [self.board]                      # une seule requête : la racine

        self.pending.clear()
        self._pending_ids.clear()
        boards: list[chess.Board] = []
        while (len(self.pending) < max_leaves
               and self.sims + len(self.pending) < self.nodes):
            node = self.root
            path = [node]
            b = self.board.copy(stack=False)
            while node.expanded and not node.terminal:
                sel = select_child(node, self.c_puct, self.fpu)
                if sel is None:                  # noeud développé sans fils légal
                    node.terminal = True         # -> traité comme feuille terminale
                    break
                move, node = sel
                b.push(move)
                path.append(node)

            if id(node) in self._pending_ids:
                break                                # doublon dans ce lot -> on stoppe
            if b.is_game_over(claim_draw=True):
                node.terminal = True
                _backup(path, _terminal_value(b))
                self.sims += 1
                continue
            _apply_vloss(path)
            self._pending_ids.add(id(node))
            self.pending.append((node, path))
            boards.append(b)
        return boards

    # --- Phase d'application (consomme les résultats du réseau) ---------------
    def apply_root(self, priors: dict, _value: float) -> None:
        root = _Node(0.0)
        _expand(root, priors)
        self._add_dirichlet(root)
        self.root = root
        self.needs_root = False
        self.sims = 0
        if not root.children:                        # position terminale d'emblée
            self._finish_if_terminal(force=True)

    def apply_leaves(self, results: list[tuple[dict, float]]) -> None:
        for (node, path), (priors, value) in zip(self.pending, results):
            _revert_vloss(path)
            _expand(node, priors)
            _backup(path, value)
            self.sims += 1
        self.pending.clear()
        self._pending_ids.clear()

    # --- Avance : joue un coup quand le budget de nœuds est atteint -----------
    def maybe_play(self) -> None:
        if self.done or self.needs_root or self.sims < self.nodes:
            return
        root = self.root
        if not root.children:
            self._finish_if_terminal(force=True)
            return
        pi = {m.uci(): c.N for m, c in root.children.items() if c.N > 0}
        temp = self.temperature if self.plies < self.temp_moves else 0.0
        move = self._select_move(root, temp)
        self.history.append((self.board.fen(), move.uci(), self.board.turn, pi))
        self.board.push(move)
        self.plies += 1
        if self.board.is_game_over(claim_draw=True) or self.plies >= self.max_plies:
            self._finalize()
        else:
            self.root = None
            self.needs_root = True
            self.sims = 0

    # --- Helpers -------------------------------------------------------------
    def _select_move(self, root: _Node, temperature: float) -> chess.Move:
        moves = list(root.children.keys())
        visits = np.fromiter((root.children[m].N for m in moves),
                             dtype=np.float64, count=len(moves))
        if temperature <= 1e-6 or visits.sum() <= 0:
            return moves[int(visits.argmax())]
        p = visits ** (1.0 / temperature)
        p /= p.sum()
        return moves[int(self.rng.choice(len(moves), p=p))]

    def _add_dirichlet(self, root: _Node) -> None:
        if self.dir_eps <= 0 or not root.children:
            return
        moves = list(root.children.keys())
        noise = self.rng.dirichlet([self.dir_alpha] * len(moves))
        for move, n in zip(moves, noise):
            c = root.children[move]
            c.prior = (1 - self.dir_eps) * c.prior + self.dir_eps * float(n)

    def _finish_if_terminal(self, force: bool = False) -> None:
        if force or self.board.is_game_over(claim_draw=True):
            self._finalize()

    def _finalize(self) -> None:
        """Assigne le résultat de la partie à chaque position (du point de vue du
        joueur au trait) et marque la partie terminée."""
        if self.board.is_game_over(claim_draw=True):
            res = self.board.result(claim_draw=True)
            result_white = 1 if res == "1-0" else (-1 if res == "0-1" else 0)
        else:
            result_white = 0                          # partie tronquée -> nulle
        self.history = [
            (fen, uci, (result_white if turn == chess.WHITE else -result_white), pi)
            for fen, uci, turn, pi in self.history
        ]
        self.done = True

    def samples(self) -> list[tuple]:
        """Samples (fen, uci, valeur, pi) une fois la partie finalisée."""
        return list(self.history)


# --- Primitives d'arbre (mêmes conventions que search/mcts.py) ----------------

def _expand(node: _Node, priors: dict) -> None:
    for move, p in priors.items():
        node.children[move] = _Node(p)
    node.expanded = True


def _backup(path: list[_Node], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.N += 1
        node.W += value
        value = -value


def _apply_vloss(path: list[_Node]) -> None:
    for node in path:
        node.vloss += 1


def _revert_vloss(path: list[_Node]) -> None:
    for node in path:
        node.vloss -= 1


# --- Scheduler : N parties, un gros batch GPU par tour -------------------------

class BatchedSelfPlay:
    """Joue `n_games` parties en parallèle en regroupant toutes les évaluations
    réseau (racines + feuilles) dans un seul forward GPU par tour."""

    def __init__(self, evaluator: Evaluator, cfg_search: dict, n_games: int,
                 leaves_per_game: int, seed: int = 0):
        self.ev = evaluator
        self.cfg_search = cfg_search
        self.n_games = n_games
        self.leaves_per_game = max(1, leaves_per_game)
        self._rng = np.random.default_rng(seed)
        self.games = [self._new_game() for _ in range(n_games)]
        self.finished_games = 0

    def _new_game(self) -> _Game:
        # Graine propre par partie : diversité d'ouvertures reproductible.
        return _Game(self.cfg_search, np.random.default_rng(self._rng.integers(2**63)))

    def step(self) -> list[list[tuple]]:
        """Un tour : collecte -> un forward GPU -> application -> coups joués.
        Renvoie la liste des samples des parties TERMINÉES pendant ce tour
        (chacune remplacée par une nouvelle partie)."""
        boards: list[chess.Board] = []
        tags: list[tuple[_Game, bool, int]] = []     # (game, is_root, count)
        for g in self.games:
            bs = g.collect(self.leaves_per_game)
            if not bs:
                continue
            is_root = g.needs_root
            tags.append((g, is_root, len(bs)))
            boards.extend(bs)

        if boards:
            results = self.ev.evaluate(boards)
            off = 0
            for g, is_root, count in tags:
                chunk = results[off:off + count]
                off += count
                if is_root:
                    g.apply_root(*chunk[0])
                else:
                    g.apply_leaves(chunk)

        finished: list[list[tuple]] = []
        for i, g in enumerate(self.games):
            g.maybe_play()
            if g.done:
                finished.append(g.samples())
                self.finished_games += 1
                self.games[i] = self._new_game()      # relance immédiatement
        return finished


# --- Pilotage / écriture des shards ------------------------------------------

def _maybe_reload(model, latest: Path, mtime, device):
    """Recharge les poids si latest.pt a changé. Retourne le nouveau mtime."""
    if not latest.exists():
        return mtime
    try:
        m = latest.stat().st_mtime
        if m != mtime:
            ck = load_checkpoint(latest, device)
            load_model_state(model, ck.get("raw_state", ck["model_state"]))
            return m
    except (OSError, KeyError, RuntimeError):
        pass                                          # checkpoint en cours d'écriture
    return mtime


def _flush(buffer_dir: Path, rows: list) -> None:
    """Écrit un shard de façon atomique (rename)."""
    name = f"selfplay_gpu_{int(time.time()*1000)}.txt.gz"
    tmp = buffer_dir / f".{name}.partial"
    write_samples(tmp, rows)
    import os
    os.replace(tmp, buffer_dir / name)


def _build_search_cfg(cfg: dict, args) -> dict:
    m = cfg.get("mcts", {})
    s = _gcfg(cfg)
    return {
        "c_puct": float(m.get("c_puct", 1.5)),
        "fpu": float(m.get("fpu", 0.2)),
        "nodes": int(args.nodes if args.nodes is not None else s.get("nodes", 200)),
        "max_plies": int(args.max_plies if args.max_plies is not None
                         else s.get("max_plies", 250)),
        "temp_moves": int(s.get("temp_moves", 20)),
        "temperature": float(s.get("temperature", 1.0)),
        "dirichlet_eps": float(s.get("dirichlet_eps", 0.25)),
        "dirichlet_alpha": float(s.get("dirichlet_alpha", 0.3)),
    }


def main() -> None:
    # Pré-lecture de --config AVANT de figer les défauts : sinon argparse lie les
    # valeurs par défaut (games, leaves…) à config.yaml et --config est ignoré
    # pour tout ce qui a un défaut tiré de la config.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config) if pre_args.config else load_config()
    s = _gcfg(cfg)
    ap = argparse.ArgumentParser(description="Self-play batché GPU -> replay buffer.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=s.get("device", None),
                    help="cuda / cuda:0 / cpu (défaut: auto)")
    ap.add_argument("--games", type=int, default=int(s.get("games", 256)),
                    help="parties jouées EN PARALLÈLE (gros batch GPU)")
    ap.add_argument("--leaves-per-game", type=int,
                    default=int(s.get("leaves_per_game", 8)), dest="leaves_per_game",
                    help="feuilles soumises par partie et par tour (batch ≈ games×ce nombre)")
    ap.add_argument("--nodes", type=int, default=None,
                    help="simulations MCTS par coup (défaut: selfplay_gpu.nodes)")
    ap.add_argument("--max-plies", type=int, default=None, dest="max_plies")
    ap.add_argument("--flush-games", type=int, default=int(s.get("flush_games", 32)),
                    dest="flush_games", help="écrire un shard tous les N parties finies")
    ap.add_argument("--max-games", type=int, default=int(s.get("max_games", 0)),
                    dest="max_games", help="arrêter après ce nb de parties (0 = infini)")
    ap.add_argument("--seed", type=int, default=int(s.get("seed", 0)))
    ap.add_argument("--compile", action="store_true",
                    default=bool(s.get("compile", False)),
                    help="torch.compile l'évaluateur (reduce-overhead)")
    args = ap.parse_args()

    # run_rl_cloud.sh / run_selfplay_gpu.sh arrêtent les workers avec SIGTERM (kill,
    # docker stop, fin de groupe de session). Sans ça, SIGTERM tue le process AVANT
    # le `finally` -> le dernier lot de samples non flushé est PERDU. On le traduit en
    # KeyboardInterrupt : même chemin de sortie propre que Ctrl-C (flush final).
    def _on_sigterm(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)

    device = resolve_device(args.device or cfg.get("train", {}).get("device", "auto"))
    search_cfg = _build_search_cfg(cfg, args)

    model = build_model(cfg)
    if args.compile:
        try:
            import torch
            model = torch.compile(model, mode="reduce-overhead")
            print("[selfplay_gpu] torch.compile activé (reduce-overhead)", flush=True)
        except Exception as e:
            print(f"[selfplay_gpu] torch.compile ignoré : {e}", flush=True)
    ev = Evaluator(model, device)
    latest = Path(cfg["paths"]["latest"])
    mtime = _maybe_reload(model, latest, None, device)

    buffer_dir = Path(cfg["data"]["buffer_dir"])
    buffer_dir.mkdir(parents=True, exist_ok=True)

    batch = args.games * args.leaves_per_game
    print(f"Self-play GPU : {args.games} parties // device={device} "
          f"| {args.nodes or search_cfg['nodes']} nœuds/coup "
          f"| batch GPU ≈ {batch} | -> {buffer_dir}", flush=True)

    sp = BatchedSelfPlay(ev, search_cfg, args.games, args.leaves_per_game, args.seed)
    pending: list[tuple] = []
    games_done = 0
    games_since_flush = 0
    t0 = time.time()
    last_log = t0
    try:
        while True:
            mtime = _maybe_reload(model, latest, mtime, device)
            for rows in sp.step():
                pending.extend(rows)
                games_done += 1
                games_since_flush += 1
            if games_since_flush >= args.flush_games and pending:
                _flush(buffer_dir, pending)
                pending = []
                games_since_flush = 0
            now = time.time()
            if now - last_log >= 10.0:
                rate = games_done / max(1e-9, now - t0)
                print(f"[selfplay_gpu] {games_done} parties | "
                      f"{rate:.1f} parties/s | +{len(pending)} samples en attente",
                      flush=True)
                last_log = now
            if args.max_games and games_done >= args.max_games:
                break
    except KeyboardInterrupt:
        print("\nArrêt self-play GPU…", flush=True)
    finally:
        if pending:
            _flush(buffer_dir, pending)
            print(f"[selfplay_gpu] flush final : {len(pending)} samples", flush=True)


if __name__ == "__main__":
    main()
