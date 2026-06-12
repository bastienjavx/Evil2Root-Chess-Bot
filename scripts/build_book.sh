#!/usr/bin/env bash
# Construit un livre d'ouvertures Polyglot (checkpoints/book.bin) depuis un PGN.
# Améliore le jeu en ouverture SANS réentraîner le réseau.
#
# Exemples :
#   scripts/build_book.sh data/lichess_raw/dump.pgn.zst
#   scripts/build_book.sh <url-lichess.pgn.zst> --max-games 100000 --max-ply 24
#
# Puis activer dans config.yaml :  book.enabled: true
cd "$(dirname "$0")/.." || exit 1
PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <fichier-ou-url.pgn[.zst]> [--out f.bin] [--max-games N] [--max-ply N]" >&2
  exit 1
fi

SRC="$1"; shift
exec "$PY" -m sanchess.data.build_book "$SRC" --out checkpoints/book.bin "$@"
