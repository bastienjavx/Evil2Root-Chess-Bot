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
# POURQUOI LE GÉNÉRATEUR EST MULTI-WORKER : le MCTS (descente PUCT, board.copy/push,
# encodage, softmax par coup) est du Python pur MONO-THREAD ; un seul process sature
# 1 cœur et laisse le GPU à ~14 % (mesuré H100 PCIe). Il faut ~16 process pour
# saturer le H100. On lance donc plusieurs workers self-play, en gardant la moitié
# des cœurs au trainer (son dataloader + le pas de gradient).
#
# Variables : CONFIG (défaut config.cloud.yaml), SEED_SHARDS (anti-oubli, défaut
# data/shards si présent : mélange des parties humaines de pretrain au self-play).
#   SELFPLAY_WORKERS  process self-play parallèles (défaut nproc/2 ; le reste va au
#                     trainer). SELFPLAY_GAMES total réparti (défaut workers×32 ->
#                     batch GPU ~256/worker, zone efficace des tensor cores).
set -u
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export NVIDIA_TF32_OVERRIDE=1

CONFIG="${CONFIG:-config.cloud.yaml}"
SEED_SHARDS="${SEED_SHARDS:-data/shards}"
SEED_ARG=()
[ -d "$SEED_SHARDS" ] && SEED_ARG=(--seed-shards "$SEED_SHARDS")

# Partage des cœurs : moitié au self-play, moitié au trainer (min 1 chacun).
NCPU="$(nproc 2>/dev/null || echo 2)"
SELFPLAY_WORKERS="${SELFPLAY_WORKERS:-$(( NCPU/2 > 0 ? NCPU/2 : 1 ))}"
SELFPLAY_GAMES="${SELFPLAY_GAMES:-$(( SELFPLAY_WORKERS * 32 ))}"

echo "[rl-cloud] config=$CONFIG"
echo "[rl-cloud] 1/2 -> trainer online (écrit checkpoints/latest.pt)"
# Verrou « trainer » partagé avec pretrain/online : un seul écrivain de latest.pt.
flock -n /tmp/sano1-trainer.lock \
    "$PY" -u -m sanchess.train.online --config "$CONFIG" "${SEED_ARG[@]}" &
TRAINER_PID=$!

# Arrêt propre des deux process sur Ctrl-C / kill.
cleanup() {
    echo "[rl-cloud] arrêt…"
    kill "$TRAINER_PID" 2>/dev/null
    # GEN_PID est le leader de session (setsid) -> tuer tout son groupe (les workers).
    [ -n "${GEN_PID:-}" ] && kill -TERM -- "-$GEN_PID" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup INT TERM

# Laisse le trainer s'initialiser (et écrire un premier latest.pt si besoin) avant
# que le générateur ne tente de le recharger à chaud.
sleep 5

echo "[rl-cloud] 2/2 -> générateur self-play GPU multi-worker (alimente le replay buffer)"
echo "[rl-cloud]      workers=$SELFPLAY_WORKERS games_total=$SELFPLAY_GAMES (trainer garde $((NCPU-SELFPLAY_WORKERS)) cœurs)"
# Délègue à run_selfplay_gpu.sh : il gère le fan-out N process + l'arrêt propre
# (trap TERM -> kill de ses workers). On le lance dans une nouvelle session pour
# pouvoir tuer tout son groupe d'un coup au cleanup.
setsid env WORKERS="$SELFPLAY_WORKERS" GAMES="$SELFPLAY_GAMES" CONFIG="$CONFIG" \
    bash scripts/run_selfplay_gpu.sh &
GEN_PID=$!

# Si l'un des deux meurt, on arrête l'autre (évite un demi-pipeline silencieux).
wait -n "$TRAINER_PID" "$GEN_PID"
echo "[rl-cloud] un process s'est arrêté -> arrêt de la boucle."
cleanup
