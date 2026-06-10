#!/usr/bin/env bash
# Collecte des parties Lichess (streaming, filtre Elo) puis pré-entraîne le réseau.
# Variables : MONTH (défaut 2019-01), MAXG (parties gardées à collecter).
set -u
cd "$(dirname "$0")/.." || exit 1

MONTH="${MONTH:-2019-01}"
MAXG="${MAXG:-50000}"
URL="https://database.lichess.org/standard/lichess_db_standard_rated_${MONTH}.pgn.zst"

# Utilise le python du venv si présent, sinon python3.
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "[1/2] Collecte de $MAXG parties (Elo>=2000) en streaming depuis $MONTH ..."
"$PY" -u -m sanchess.data.pgn_to_samples "$URL" --out data/shards --max-games "$MAXG" \
    || echo "(collecte interrompue — on entraîne sur ce qui a été récupéré)"

if ! ls data/shards/*.txt.gz >/dev/null 2>&1; then
    echo "Aucun shard produit, arrêt."; exit 1
fi

echo "[2/2] Pré-entraînement (checkpoints/latest.pt mis à jour régulièrement) ..."
exec "$PY" -u -m sanchess.train.pretrain --shards data/shards
