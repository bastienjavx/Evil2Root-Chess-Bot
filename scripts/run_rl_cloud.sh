#!/usr/bin/env bash
# Boucle RL self-play COMPLÈTE sur le cloud (à lancer APRÈS le pré-entraînement).
#
#   selfplay_gpu  ──écrit──>  data/replay_buffer  ──ingéré par──>  online.py
#        ▲                                                            │
#        └────────── hot-reload checkpoints/latest.pt ◄──────────────┘
#
# Deux process sur la même machine GPU :
#   1. le TRAINER (online.py) ingère le replay buffer, met à jour latest.pt
#      (verrou flock partagé : un seul écrivain de latest.pt) ;
#   2. le GÉNÉRATEUR (selfplay_gpu) joue en parallèle et recharge à chaud latest.pt
#      -> les données s'améliorent à mesure que le réseau progresse.
#
# Le réseau JOUE et S'ENTRAÎNE en même temps sur le GPU. Sur un GPU bien dimensionné
# (L40S/H100) les deux cohabitent ; si la VRAM est juste, baisser selfplay_gpu.games.
#
# Variables : CONFIG (défaut config.cloud.yaml), SEED_SHARDS (anti-oubli, défaut
# data/shards si présent : mélange des parties humaines de pretrain au self-play).
set -u
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export NVIDIA_TF32_OVERRIDE=1

CONFIG="${CONFIG:-config.cloud.yaml}"
SEED_SHARDS="${SEED_SHARDS:-data/shards}"
SEED_ARG=()
[ -d "$SEED_SHARDS" ] && SEED_ARG=(--seed-shards "$SEED_SHARDS")

echo "[rl-cloud] config=$CONFIG"
echo "[rl-cloud] 1/2 -> trainer online (écrit checkpoints/latest.pt)"
# Verrou « trainer » partagé avec pretrain/online : un seul écrivain de latest.pt.
flock -n /tmp/sano1-trainer.lock \
    "$PY" -u -m sanchess.train.online --config "$CONFIG" "${SEED_ARG[@]}" &
TRAINER_PID=$!

# Arrêt propre des deux process sur Ctrl-C / kill.
cleanup() {
    echo "[rl-cloud] arrêt…"
    kill "$TRAINER_PID" "${GEN_PID:-}" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup INT TERM

# Laisse le trainer s'initialiser (et écrire un premier latest.pt si besoin) avant
# que le générateur ne tente de le recharger à chaud.
sleep 5

echo "[rl-cloud] 2/2 -> générateur self-play GPU (alimente le replay buffer)"
"$PY" -u -m sanchess.train.selfplay_gpu --config "$CONFIG" &
GEN_PID=$!

# Si l'un des deux meurt, on arrête l'autre (évite un demi-pipeline silencieux).
wait -n "$TRAINER_PID" "$GEN_PID"
echo "[rl-cloud] un process s'est arrêté -> arrêt de la boucle."
cleanup
