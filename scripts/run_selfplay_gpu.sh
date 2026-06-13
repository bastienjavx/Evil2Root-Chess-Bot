#!/usr/bin/env bash
# Self-play BATCHÉ GPU : joue des centaines de parties en parallèle et regroupe
# toutes les évaluations réseau dans un seul forward GPU -> alimente le replay
# buffer (data/replay_buffer) pour la boucle RL. À lancer SUR la machine GPU.
#
# Variables (sinon valeurs de la section selfplay_gpu de la config) :
#   CONFIG   fichier de config (défaut config.cloud.yaml sur le cloud)
#   GAMES    parties jouées en parallèle (monter pour saturer la VRAM)
#   NODES    simulations MCTS par coup
#   DEVICE   cuda (défaut) | cuda:0 | cpu
#
# Exemple cloud : entraînement + génération de données ensemble
#   ./scripts/run_rl_cloud.sh        # lance les deux (recommandé)
# ou séparément :
#   GAMES=512 NODES=400 ./scripts/run_selfplay_gpu.sh
set -u
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export NVIDIA_TF32_OVERRIDE=1   # matmuls plus rapides (Ampere/Ada/Hopper)

CONFIG="${CONFIG:-config.cloud.yaml}"
ARGS=(--config "$CONFIG")
[ -n "${GAMES:-}" ]  && ARGS+=(--games "$GAMES")
[ -n "${NODES:-}" ]  && ARGS+=(--nodes "$NODES")
[ -n "${DEVICE:-}" ] && ARGS+=(--device "$DEVICE")

echo "[selfplay-gpu] config=$CONFIG games=${GAMES:-config} nodes=${NODES:-config} device=${DEVICE:-config}"
exec "$PY" -u -m sanchess.train.selfplay_gpu "${ARGS[@]}"
