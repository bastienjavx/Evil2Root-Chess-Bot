#!/usr/bin/env bash
# Self-play CPU : génère des parties (MCTS) -> data/replay_buffer, en PARALLÈLE
# de l'entraînement GPU. Aucune synchro : le GPU consomme le buffer à son rythme.
#
# Variables (sinon valeurs du bloc `selfplay` de config.yaml) :
#   WORKERS  parties en parallèle (≈ 1 cœur chacune ; <nproc pour garder du bureau)
#   NODES    simulations MCTS par coup
#   DEVICE   cpu (défaut) | cuda | mps
#
# Exemple : faire tourner GPU (entraînement) + CPU (self-play) ensemble :
#   ./scripts/run_pretrain.sh &              # ou run_online.sh : le GPU entraîne
#   WORKERS=4 ./scripts/run_selfplay.sh      # le CPU produit des parties
set -u
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3

ARGS=()
[ -n "${WORKERS:-}" ] && ARGS+=(--workers "$WORKERS")
[ -n "${NODES:-}" ]   && ARGS+=(--nodes "$NODES")
[ -n "${DEVICE:-}" ]  && ARGS+=(--device "$DEVICE")

echo "[self-play] workers=${WORKERS:-config} nodes=${NODES:-config} device=${DEVICE:-config}"
exec "$PY" -u -m sanchess.train.selfplay "${ARGS[@]}"
