#!/usr/bin/env bash
# Lance le bot Lichess San-o1 (nécessite un compte BOT + LICHESS_TOKEN dans .env).
cd "$(dirname "$0")/.." || exit 1
# Utilise l'interpréteur du venv (les deps — chess, torch… — y sont installées).
PY="$(dirname "$0")/../.venv/bin/python"
[ -x "$PY" ] || PY="python3"
# Allocateur en segments extensibles : limite la fragmentation pour cohabiter
# avec le pré-entraînement/online sur le même GPU 8 Go (évite l'OOM CUDA).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PY" -u -m sanchess.lichess_bot "$@"
