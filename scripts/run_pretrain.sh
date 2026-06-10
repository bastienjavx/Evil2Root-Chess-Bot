#!/usr/bin/env bash
# Pré-entraînement supervisé sur les shards Lichess (data/shards par défaut).
cd "$(dirname "$0")/.." || exit 1
exec python -m sanchess.train.pretrain "$@"
