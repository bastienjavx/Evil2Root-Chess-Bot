#!/usr/bin/env bash
# Lance le bot Lichess San-o1 (nécessite un compte BOT + LICHESS_TOKEN dans .env).
cd "$(dirname "$0")/.." || exit 1
exec python -u -m sanchess.lichess_bot "$@"
