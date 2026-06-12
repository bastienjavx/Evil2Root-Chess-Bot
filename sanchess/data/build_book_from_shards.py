"""Construit un livre d'ouvertures Polyglot (.bin) depuis les shards locaux.

Variante de `build_book` qui lit `data/shards/*.txt.gz` au lieu d'un PGN. Les
shards contiennent déjà tout le nécessaire : `fen <TAB> coup_uci <TAB> valeur`,
où `valeur` ∈ {-1,0,1} est le résultat DU POINT DE VUE DU CAMP AU TRAIT. On
agrège donc les coups des premiers `max_ply` demi-coups (le ply est déduit du
compteur de coups de la FEN), pondérés par `(valeur+1)/2` (gain=1, nulle=0.5),
et on écrit un fichier Polyglot 100 % standard.

Usage :
  python -m sanchess.data.build_book_from_shards \\
      --out checkpoints/book.bin --max-ply 24 --min-count 3
  # dossier de shards explicite :
  python -m sanchess.data.build_book_from_shards data/shards --out checkpoints/book.bin
"""

from __future__ import annotations

import argparse
import glob
import struct
from collections import defaultdict
from pathlib import Path

import chess
import chess.polyglot

from ..utils import load_config
from .build_book import _ENTRY, _MAX_WEIGHT, _encode_move
from .samples import iter_samples


def _ply_from_fen(fen: str) -> int:
    """Demi-coup (0 = position de départ) déduit du compteur de coups FEN."""
    parts = fen.split()
    turn, fullmove = parts[1], int(parts[5])
    return (fullmove - 1) * 2 + (0 if turn == "w" else 1)


def build(shard_paths, out_path: Path, max_ply: int, min_count: int) -> None:
    weights: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)

    samples_seen = samples_used = 0
    for shard in shard_paths:
        for fen, move_uci, value in iter_samples(shard):
            samples_seen += 1
            # Filtre PLY d'abord (ops chaîne) : évite de construire un Board
            # pour les millions de positions de milieu/fin de partie.
            if _ply_from_fen(fen) >= max_ply:
                continue
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            key = (chess.polyglot.zobrist_hash(board), _encode_move(board, move))
            weights[key] += (value + 1) / 2.0      # gain=1, nulle=0.5, perte=0
            counts[key] += 1
            samples_used += 1
        print(f"  ... {Path(shard).name} ({samples_used} retenus / {samples_seen} lus)")

    # Regroupe par position pour NORMALISER les poids à l'intérieur de chaque
    # position : Polyglot ne compare les poids qu'entre coups d'une même clé. On
    # met le meilleur coup de chaque position à _MAX_WEIGHT et les autres à
    # l'échelle -> on récupère le vrai ratio (ex. e4 vs d4) sans saturer, tout en
    # préservant les positions profondes au faible total cumulé.
    by_key: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (zkey, pmove), total in weights.items():
        if counts[(zkey, pmove)] < min_count:
            continue
        by_key[zkey].append((pmove, total))

    entries = []
    for zkey, moves in by_key.items():
        top = max(total for _, total in moves)
        if top <= 0:
            # Position où TOUS les coups ont perdu (score 0) : pas de signal de
            # résultat -> on retombe sur la popularité (occurrences) comme base.
            base = {pmove: counts[(zkey, pmove)] for pmove, _ in moves}
        else:
            base = {pmove: total for pmove, total in moves}
        top = max(base.values())
        for pmove, val in base.items():
            w = max(1, min(_MAX_WEIGHT, int(round(val / top * _MAX_WEIGHT))))
            entries.append((zkey, pmove, w))

    # Polyglot exige un tri par clé (puis poids décroissant, par convention).
    entries.sort(key=lambda e: (e[0], -e[2]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        for zkey, pmove, w in entries:
            fh.write(_ENTRY.pack(zkey, pmove, w, 0))

    positions = len({e[0] for e in entries})
    print(f"Terminé : {samples_used} samples retenus / {samples_seen} lus -> "
          f"{len(entries)} entrées, {positions} positions, {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Livre Polyglot (.bin) depuis les shards")
    ap.add_argument("shards_dir", nargs="?", default=None,
                    help="dossier de shards (def. data.shards_dir de la config)")
    ap.add_argument("--out", default="checkpoints/book.bin", help="fichier .bin de sortie")
    ap.add_argument("--max-ply", type=int, default=24,
                    help="profondeur en demi-coups indexée (def. 24 = 12 coups)")
    ap.add_argument("--min-count", type=int, default=3,
                    help="occurrences min d'un (position,coup) pour l'inclure")
    args = ap.parse_args()

    cfg = load_config()
    shards_dir = Path(args.shards_dir or cfg["data"]["shards_dir"])
    shard_paths = sorted(glob.glob(str(shards_dir / "*.txt.gz")))
    if not shard_paths:
        raise SystemExit(f"Aucun shard *.txt.gz dans {shards_dir}")
    print(f"{len(shard_paths)} shards depuis {shards_dir}")
    build(shard_paths, Path(args.out), args.max_ply, args.min_count)


if __name__ == "__main__":
    main()
