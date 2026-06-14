#!/usr/bin/env bash
# Lance le moteur UCI San-o1 (à brancher dans Nibbler/Cutechess/Arena).
cd "$(dirname "$0")/.." || exit 1
exec python3 -m sanchess.uci
