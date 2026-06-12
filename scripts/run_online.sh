#!/usr/bin/env bash
# Apprentissage continu : stream Lichess + entraînement de fond.
# Lance les deux processus ; Ctrl-C arrête tout.
cd "$(dirname "$0")/.." || exit 1
# Allocateur en segments extensibles : limite la fragmentation pour cohabiter
# avec le bot sur le même GPU 8 Go (évite l'OOM CUDA).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sanchess.data.stream --perf "${PERF:-blitz}" &
STREAM_PID=$!
trap "kill $STREAM_PID 2>/dev/null" EXIT

exec python3 -m sanchess.train.online --seed-shards data/shards "$@"
