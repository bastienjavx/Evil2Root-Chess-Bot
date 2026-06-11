"""Self-play CPU : génère des parties via MCTS et alimente le replay buffer.

Conçu pour tourner EN PARALLÈLE de l'entraînement GPU, sans le bloquer : il n'y a
aucune synchronisation entre les deux. Chaque partie que le CPU joue contre
lui-même est convertie en samples `(fen, coup, valeur)` écrits comme shards dans
`data/replay_buffer/`, exactement au format de `data/stream.py`. L'entraîneur
(`train/online.py`, ou le rang 0 GPU du distribué qui consomme aussi le buffer)
les ingère à son rythme. Le GPU apprend, le CPU produit des données : chacun va
à sa vitesse.

Le réseau utilisé pour jouer est rechargé à chaud depuis `checkpoints/latest.pt`
à chaque nouvelle partie : le self-play profite donc des poids qui s'améliorent.

Garde-fous (cette machine a déjà figé par saturation CPU) :
  - nombre de workers borné (laisser des cœurs au bureau / au DataLoader GPU) ;
  - `nice` + `torch.set_num_threads` bas par worker -> jamais 100 % sur 12 cœurs.

Convention valeur (cf. samples.py) : résultat de la partie DU POINT DE VUE DU
JOUEUR AU TRAIT dans la position. Cible politique = coup MCTS le plus visité
(un seul coup par position, comme les samples humains -> pipeline inchangé).

Usage :
  python -m sanchess.train.selfplay                 # paramètres du config.yaml
  python -m sanchess.train.selfplay --workers 4 --nodes 160
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.multiprocessing as mp

from ..data.samples import write_samples
from ..model import build_model
from ..search.mcts import MCTS, Evaluator
from ..utils import (load_checkpoint, load_config, load_model_state,
                     resolve_device)


def _scfg(cfg: dict) -> dict:
    return cfg.get("selfplay", {})


# --- Une partie ---------------------------------------------------------------

def _select_move(root, temperature: float) -> chess.Move:
    """Coup joué : échantillonné ∝ visites^(1/T) (diversité), ou argmax si T≈0."""
    moves = list(root.children.keys())
    visits = np.fromiter((root.children[m].N for m in moves),
                         dtype=np.float64, count=len(moves))
    if temperature <= 1e-6 or visits.sum() <= 0:
        return moves[int(visits.argmax())]
    p = visits ** (1.0 / temperature)
    p /= p.sum()
    return moves[int(np.random.choice(len(moves), p=p))]


def play_game(mcts: MCTS, nodes: int, max_plies: int,
              temp_moves: int, temperature: float):
    """Joue une partie complète contre soi-même. Retourne des samples AlphaZero
    (fen, coup_joué, valeur, pi) où pi = distribution de visites MCTS {uci: N}."""
    board = chess.Board()
    # (fen, coup_uci, trait, pi)
    history: list[tuple[str, str, bool, dict]] = []
    while not board.is_game_over(claim_draw=True) and len(history) < max_plies:
        root = mcts.run(board, nodes)
        if not root.children:                    # position terminale inattendue
            break
        pi = {m.uci(): c.N for m, c in root.children.items() if c.N > 0}
        t = temperature if len(history) < temp_moves else 0.0
        move = _select_move(root, t)
        history.append((board.fen(), move.uci(), board.turn, pi))
        board.push(move)

    if board.is_game_over(claim_draw=True):
        res = board.result(claim_draw=True)
        result_white = 1 if res == "1-0" else (-1 if res == "0-1" else 0)
    else:
        result_white = 0                          # partie tronquée -> nulle

    return [(fen, mv, result_white if turn == chess.WHITE else -result_white, pi)
            for fen, mv, turn, pi in history]


# --- Worker -------------------------------------------------------------------

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
        pass                                       # checkpoint en cours d'écriture
    return mtime


def run_worker(wid: int, cfg: dict, device: str, args):
    s = _scfg(cfg)
    torch.set_num_threads(max(1, int(args.threads)))   # ne pas saturer les cœurs
    try:
        os.nice(int(args.nice))                        # priorité basse (anti-freeze)
    except (OSError, AttributeError):
        pass

    model = build_model(cfg)
    ev = Evaluator(model, device)
    mcts = MCTS(ev, cfg)
    mcts.dir_eps = float(args.dirichlet_eps)           # bruit racine pour explorer
    mcts.dir_alpha = float(s.get("dirichlet_alpha", mcts.dir_alpha))

    latest = Path(cfg["paths"]["latest"])
    mtime = None
    buffer_dir = Path(cfg["data"]["buffer_dir"])
    buffer_dir.mkdir(parents=True, exist_ok=True)

    print(f"[selfplay w{wid}] device={device} threads={args.threads} "
          f"nodes={args.nodes} -> {buffer_dir}", flush=True)

    pending: list[tuple] = []
    games = 0
    while True:
        mtime = _maybe_reload(model, latest, mtime, device)
        rows = play_game(mcts, args.nodes, args.max_plies,
                         args.temp_moves, args.temperature)
        pending.extend(rows)
        games += 1
        if games % max(1, int(args.flush_games)) == 0 and pending:
            _flush(buffer_dir, wid, pending)
            print(f"[selfplay w{wid}] {games} parties | +{len(pending)} samples",
                  flush=True)
            pending = []


def _flush(buffer_dir: Path, wid: int, rows: list) -> None:
    """Écrit un shard de façon atomique (rename) pour que l'ingest ne lise
    jamais un fichier à moitié écrit."""
    name = f"selfplay_w{wid}_{int(time.time()*1000)}.txt.gz"
    tmp = buffer_dir / f".{name}.partial"
    write_samples(tmp, rows)
    os.replace(tmp, buffer_dir / name)


# --- Lancement ----------------------------------------------------------------

def main():
    cfg = load_config()
    s = _scfg(cfg)
    ap = argparse.ArgumentParser(description="Self-play CPU -> replay buffer.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=s.get("device", "cpu"),
                    help="appareil de jeu (cpu recommandé : laisse le GPU à l'entraînement)")
    ap.add_argument("--workers", type=int, default=int(s.get("workers", 4)),
                    help="parties jouées en parallèle (1 process/cœur env.)")
    ap.add_argument("--threads", type=int, default=int(s.get("threads_per_worker", 1)),
                    help="threads torch PAR worker (1 = un cœur par worker)")
    ap.add_argument("--nodes", type=int, default=int(s.get("nodes", 160)),
                    help="simulations MCTS par coup (↓ = parties plus rapides)")
    ap.add_argument("--max-plies", type=int, default=int(s.get("max_plies", 250)),
                    dest="max_plies")
    ap.add_argument("--temp-moves", type=int, default=int(s.get("temp_moves", 20)),
                    dest="temp_moves", help="nb de coups joués avec température (diversité)")
    ap.add_argument("--temperature", type=float, default=float(s.get("temperature", 1.0)))
    ap.add_argument("--dirichlet-eps", type=float,
                    default=float(s.get("dirichlet_eps", 0.25)), dest="dirichlet_eps")
    ap.add_argument("--flush-games", type=int, default=int(s.get("flush_games", 4)),
                    dest="flush_games", help="écrire un shard tous les N parties")
    ap.add_argument("--nice", type=int, default=int(s.get("nice", 10)))
    args = ap.parse_args()
    if args.config:
        cfg = load_config(args.config)

    device = resolve_device(args.device)
    print(f"Self-play : {args.workers} worker(s) sur {device} "
          f"(nice={args.nice}, {args.threads} thread(s)/worker).")

    if args.workers <= 1:
        run_worker(0, cfg, device, args)
        return

    ctx = mp.get_context("spawn")          # process frais : pas de fork post-torch
    procs = [ctx.Process(target=run_worker, args=(i, cfg, device, args), daemon=False)
             for i in range(args.workers)]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\nArrêt self-play…")
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
