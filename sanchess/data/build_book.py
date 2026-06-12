"""Construit un livre d'ouvertures Polyglot (.bin) depuis un dump PGN.

Aucune dépendance externe : on agrège les coups joués dans les `max_ply`
premiers demi-coups des parties (filtrées par le même critère qualité que
`pgn_to_samples` : Elo mini, pas de bullet, résultat valide), puis on écrit un
fichier Polyglot 100 % standard, relisible par n'importe quel moteur.

Pondération : chaque coup gagne `win + draw/2` point par partie (du point de
vue du camp au trait). Un coup souvent gagnant pèse donc plus lourd, ce qui
oriente la sélection pondérée du livre vers la bonne théorie.

Usage :
  python -m sanchess.data.build_book data/lichess_raw/dump.pgn.zst \\
      --out checkpoints/book.bin --max-games 200000 --max-ply 24

  # Directement depuis une URL Lichess (streaming, pas de téléchargement complet) :
  python -m sanchess.data.build_book \\
      https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst \\
      --out checkpoints/book.bin --max-games 100000
"""

from __future__ import annotations

import argparse
import struct
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot

from ..utils import load_config
from .pgn_to_samples import RESULT_VALUE, _keep_game, _open_pgn

# Codes de promotion Polyglot (aucune=0, C=1, F=2, T=3, D=4).
_POLY_PROMO = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
# Entrée Polyglot : clé u64, coup u16, poids u16, learn u32, big-endian (16 o).
_ENTRY = struct.Struct(">QHHI")
_MAX_WEIGHT = 0xFFFF


def _encode_move(board: chess.Board, move: chess.Move) -> int:
    """Encode un coup au format Polyglot 16 bits (roque = roi vers SA tour)."""
    to_sq = move.to_square
    from_sq = move.from_square
    if board.is_castling(move):
        # Polyglot stocke le roque comme « roi prend sa propre tour ».
        rank = chess.square_rank(from_sq)
        to_file = 7 if chess.square_file(to_sq) > chess.square_file(from_sq) else 0
        to_sq = chess.square(to_file, rank)
    promo = _POLY_PROMO.get(move.promotion, 0)
    return ((promo & 0x7) << 12
            | (chess.square_rank(from_sq) & 0x7) << 9
            | (chess.square_file(from_sq) & 0x7) << 6
            | (chess.square_rank(to_sq) & 0x7) << 3
            | (chess.square_file(to_sq) & 0x7))


def build(pgn_src, out_path: Path, cfg: dict, max_games: int | None,
          max_ply: int, min_count: int) -> None:
    cfg_data = cfg["data"]
    # weights[(zobrist_key, poly_move)] -> score cumulé ; counts -> nb d'occurrences.
    weights: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)

    handle = _open_pgn(pgn_src)
    games_done = games_kept = 0
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        games_done += 1
        if not _keep_game(game.headers, cfg_data):
            continue
        result_white = RESULT_VALUE[game.headers["Result"]]  # 1 / 0 / -1
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= max_ply:
                break
            # Score du point de vue du camp AU TRAIT : victoire=1, nulle=0.5.
            side = result_white if board.turn == chess.WHITE else -result_white
            score = (side + 1) / 2.0
            key = (chess.polyglot.zobrist_hash(board), _encode_move(board, move))
            weights[key] += score
            counts[key] += 1
            board.push(move)
        games_kept += 1
        if games_kept % 20000 == 0:
            print(f"  ... {games_kept} parties retenues ({games_done} lues)")
        if max_games and games_kept >= max_games:
            break

    # Construit les entrées : poids = score moyen * occurrences, borné à u16.
    entries = []
    for (zkey, pmove), total in weights.items():
        n = counts[(zkey, pmove)]
        if n < min_count:
            continue
        w = int(round(total))            # somme des (gain + nulle/2) -> entier
        w = max(1, min(_MAX_WEIGHT, w))
        entries.append((zkey, pmove, w))

    # Polyglot exige un tri par clé (puis poids décroissant, par convention).
    entries.sort(key=lambda e: (e[0], -e[2]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        for zkey, pmove, w in entries:
            fh.write(_ENTRY.pack(zkey, pmove, w, 0))

    positions = len({e[0] for e in entries})
    print(f"Terminé : {games_kept} parties / {games_done} lues -> "
          f"{len(entries)} entrées, {positions} positions, {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Construit un livre Polyglot (.bin)")
    ap.add_argument("pgn", help="chemin/URL d'un .pgn ou .pgn.zst")
    ap.add_argument("--out", default="checkpoints/book.bin", help="fichier .bin de sortie")
    ap.add_argument("--max-games", type=int, default=None, help="parties retenues max")
    ap.add_argument("--max-ply", type=int, default=24,
                    help="profondeur en demi-coups indexée (def. 24 = 12 coups)")
    ap.add_argument("--min-count", type=int, default=3,
                    help="occurrences min d'un (position,coup) pour l'inclure")
    args = ap.parse_args()

    cfg = load_config()
    build(args.pgn, Path(args.out), cfg, args.max_games, args.max_ply, args.min_count)


if __name__ == "__main__":
    main()
