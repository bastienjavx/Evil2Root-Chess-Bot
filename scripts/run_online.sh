#!/usr/bin/env bash
# Apprentissage continu : stream Lichess + entraînement de fond.
# Lance les deux processus ; Ctrl-C arrête tout.
cd "$(dirname "$0")/.." || exit 1

python -m sanchess.data.stream --perf "${PERF:-blitz}" &
STREAM_PID=$!
trap "kill $STREAM_PID 2>/dev/null" EXIT

exec python -m sanchess.train.online --seed-shards data/shards "$@"
