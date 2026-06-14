#!/usr/bin/env bash
# Self-play BATCHÉ GPU : joue des centaines de parties en parallèle et regroupe
# toutes les évaluations réseau dans un seul forward GPU -> alimente le replay
# buffer (data/replay_buffer) pour la boucle RL. À lancer SUR la machine GPU.
#
# POURQUOI PLUSIEURS WORKERS ? Le MCTS (descente PUCT, board.copy/push, encodage)
# est du Python pur, MONO-THREAD : un seul process sature UN cœur et laisse le GPU
# quasi inactif (mesuré sur L40S : GPU 0 %, 1 cœur à 100 %). On lance donc N process
# indépendants (un par cœur) qui écrivent TOUS dans le même data/replay_buffer
# (noms de shards horodatés + rename atomique -> aucune coordination nécessaire) et
# rechargent latest.pt à chaud chacun de leur côté. Débit ≈ ×N (scaling cœurs).
#
# Variables (sinon valeurs de la section selfplay_gpu de la config) :
#   CONFIG   fichier de config (défaut config.cloud.yaml sur le cloud)
#   WORKERS  process parallèles (défaut nproc-1 ; 1 = ancien comportement mono)
#   GAMES    parties en parallèle au TOTAL (réparties sur les workers)
#   NODES    simulations MCTS par coup
#   DEVICE   cuda (défaut) | cuda:0 | cpu
#
# Exemple cloud : entraînement + génération de données ensemble
#   ./scripts/run_rl_cloud.sh        # lance les deux (recommandé)
# ou séparément :
#   WORKERS=7 GAMES=512 NODES=400 ./scripts/run_selfplay_gpu.sh
set -u
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export NVIDIA_TF32_OVERRIDE=1   # matmuls plus rapides (Ampere/Ada/Hopper)
# Un cœur dédié par worker : on EMPÊCHE chaque process de lancer ses propres
# threads BLAS/OpenMP (le forward est sur GPU, le reste est du Python séquentiel).
# Sans ça, N workers × plusieurs threads = sur-souscription -> context-switching pur.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CONFIG="${CONFIG:-config.cloud.yaml}"

# Nb de workers : défaut = nproc-1 (un cœur laissé à l'OS / au process principal).
NCPU="$(nproc 2>/dev/null || echo 1)"
WORKERS="${WORKERS:-$(( NCPU > 1 ? NCPU - 1 : 1 ))}"
[ "$WORKERS" -ge 1 ] 2>/dev/null || WORKERS=1

# Total de parties à répartir sur les workers (défaut = défaut config: 256).
TOTAL_GAMES="${GAMES:-256}"
PER_WORKER=$(( (TOTAL_GAMES + WORKERS - 1) / WORKERS ))   # ceil
[ "$PER_WORKER" -ge 1 ] || PER_WORKER=1

ARGS=(--config "$CONFIG")
[ -n "${NODES:-}" ]   && ARGS+=(--nodes "$NODES")
[ -n "${DEVICE:-}" ]  && ARGS+=(--device "$DEVICE")
[ "${COMPILE:-}" = "1" ] && ARGS+=(--compile)

echo "[selfplay-gpu] config=$CONFIG workers=$WORKERS games_total≈$((PER_WORKER*WORKERS)) (${PER_WORKER}/worker) nodes=${NODES:-config} device=${DEVICE:-config} compile=${COMPILE:-config}"

# --- Un seul worker : exec direct (pas de couche de gestion) ------------------
if [ "$WORKERS" -eq 1 ]; then
  exec "$PY" -u -m sanchess.train.selfplay_gpu "${ARGS[@]}" --games "$PER_WORKER"
fi

# --- Plusieurs workers : lancer en parallèle, arrêt propre de tous -------------
PIDS=()
cleanup() {
  trap - INT TERM
  echo "[selfplay-gpu] arrêt des $WORKERS workers…"
  kill -TERM "${PIDS[@]}" 2>/dev/null
  wait
  exit 0
}
trap cleanup INT TERM

SEED_BASE="${SEED:-0}"
for i in $(seq 0 $((WORKERS - 1))); do
  seed=$(( SEED_BASE + i ))
  # Préfixe [w<i>] sur chaque ligne pour distinguer les workers dans le log commun.
  "$PY" -u -m sanchess.train.selfplay_gpu "${ARGS[@]}" --games "$PER_WORKER" --seed "$seed" 2>&1 \
    | while IFS= read -r line; do printf '[w%d] %s\n' "$i" "$line"; done &
  PIDS+=("$!")
done
wait
