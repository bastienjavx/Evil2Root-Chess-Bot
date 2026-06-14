#!/usr/bin/env python
"""Match ENGINE vs ENGINE entre deux checkpoints -> mesure de FORCE RELATIVE.

À la différence de bench_eval.py (débit GPU pur, poids aléatoires), ce script
fait JOUER deux réseaux l'un contre l'autre via le MÊME MCTS que le bot, et en
déduit un écart d'Elo. C'est le bon signal pour décider quand arrêter la boucle
RL : on mesure latest.pt contre un checkpoint de RÉFÉRENCE figé, et on coupe
quand le gain d'Elo plafonne (cf. scripts/track_strength.sh qui boucle dessus).

  python scripts/arena.py --a checkpoints/latest.pt --b checkpoints/ref.pt \
      --games 40 --nodes 200

Diversité des parties : les `--rand-plies` premiers demi-coups sont tirés au
hasard (sinon, jeu déterministe -> 40 parties identiques, aucun signal). Les
couleurs sont alternées à chaque partie pour neutraliser l'avantage des Blancs.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanchess.model import build_model_from_checkpoint
from sanchess.search.mcts import MCTS, Evaluator, best_move
from sanchess.utils import (load_checkpoint, load_config, load_model_state,
                            resolve_device)


def load_engine(path: str, cfg: dict, device: str) -> tuple[MCTS, int]:
    """Charge un checkpoint et renvoie (MCTS prêt à jouer, step du checkpoint).

    On force dirichlet_eps=0 : en MATCH on veut le meilleur coup, pas de bruit
    d'exploration racine (réservé au self-play).

    GARDE-FOU : les checkpoints sauvegardés pendant un entraînement torch.compile
    portent un préfixe `_orig_mod.` sur leurs clés. Sans le retirer, le chargement
    tolérant (strict=False) jette SILENCIEUSEMENT tous les poids -> le réseau joue
    avec des poids ALÉATOIRES et l'Elo mesuré ne veut rien dire. On dé-préfixe puis
    on EXIGE un chargement complet (sinon on lève) pour ne jamais publier un faux Elo."""
    ck = load_checkpoint(path, map_location=device)
    state = ck.get("raw_state") or ck.get("model_state", {})
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
             for k, v in state.items()}
    model = build_model_from_checkpoint(ck, fallback_cfg=cfg).to(device).eval()
    rep = load_model_state(model, state)
    if rep["missing"] or rep["mismatched"]:
        raise SystemExit(
            f"{path}: chargement INCOMPLET (manquants={rep['missing']}, "
            f"formes incompatibles={rep['mismatched']}) -> le réseau jouerait avec "
            f"des poids partiels/aléatoires. Match annulé.")
    match_cfg = {**cfg, "mcts": {**cfg.get("mcts", {}), "dirichlet_eps": 0.0}}
    return MCTS(Evaluator(model, device), match_cfg), int(ck.get("step", 0))


def play_game(white: MCTS, black: MCTS, nodes: int, rand_plies: int,
              max_plies: int) -> float:
    """Renvoie le résultat du point de vue des BLANCS : 1.0 / 0.5 / 0.0."""
    board = chess.Board()
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if ply < rand_plies:                       # ouverture aléatoire (diversité)
            move = random.choice(list(board.legal_moves))
        else:
            engine = white if board.turn == chess.WHITE else black
            move = best_move(engine.run(board, nodes))
            if move is None:                        # position sans coup légal
                break
        board.push(move)
        ply += 1
    res = board.result(claim_draw=True)            # "1-0" / "0-1" / "1/2-1/2" / "*"
    return {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)  # "*" (coupé) = nulle


def elo_diff(score: float, games: int) -> float:
    """Écart d'Elo de A par rapport à B à partir du score moyen de A."""
    eps = 1.0 / (2 * games)                          # évite log(0) à 0 % / 100 %
    s = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint A (ex. latest.pt)")
    ap.add_argument("--b", required=True, help="checkpoint B = référence figée")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--nodes", type=int, default=200, help="simulations MCTS/coup")
    ap.add_argument("--rand-plies", type=int, default=6,
                    help="demi-coups d'ouverture aléatoires (diversité)")
    ap.add_argument("--max-plies", type=int, default=250)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    cfg = load_config(args.config)
    device = args.device or resolve_device(cfg.get("train", {}).get("device", "auto"))

    a, step_a = load_engine(args.a, cfg, device)
    b, step_b = load_engine(args.b, cfg, device)
    print(f"A = {args.a} (step {step_a})", file=sys.stderr)
    print(f"B = {args.b} (step {step_b})", file=sys.stderr)
    print(f"device={device}  games={args.games}  nodes={args.nodes}", file=sys.stderr)

    a_wins = draws = b_wins = 0
    for g in range(args.games):
        a_white = (g % 2 == 0)                       # alterne les couleurs
        white, black = (a, b) if a_white else (b, a)
        white_score = play_game(white, black, args.nodes, args.rand_plies,
                                args.max_plies)
        a_score = white_score if a_white else 1.0 - white_score
        if a_score == 1.0:
            a_wins += 1
        elif a_score == 0.0:
            b_wins += 1
        else:
            draws += 1
        done = g + 1
        sc = (a_wins + 0.5 * draws) / done
        print(f"  partie {done}/{args.games}  A={a_wins} N={draws} B={b_wins}  "
              f"score={sc:.3f}  elo={elo_diff(sc, done):+.0f}", file=sys.stderr)

    score = (a_wins + 0.5 * draws) / args.games
    elo = elo_diff(score, args.games)
    # Ligne FINALE machine-parsable (track_strength.sh la grep).
    print(f"RESULT games={args.games} A_wins={a_wins} draws={draws} "
          f"B_wins={b_wins} score={score:.4f} elo_diff={elo:+.1f} "
          f"stepA={step_a} stepB={step_b}")


if __name__ == "__main__":
    main()
