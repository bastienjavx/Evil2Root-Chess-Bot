#!/usr/bin/env bash
# RL EN LOCAL sur RTX 2070 SUPER (8 Go) : le moteur joue contre lui-même
# (self-play CPU) et apprend de SES parties (entraînement online sur GPU).
#
#   self-play CPU ──► data/replay_buffer ──► online GPU ──► checkpoints/latest.pt
#                     (ses propres parties)                 (rechargé à chaud par
#                                                            l'UCI / le bot / le web)
#
# Pourquoi le self-play sur CPU ? Sur 8 Go on évite la contention VRAM : le
# self-play tourne sur les cœurs CPU (~0 Go VRAM, laisse le GPU 100 % à
# l'entraînement) et l'online entraîne le 24x320 en bf16/fp16. C'est plus lent
# que sur GPU, mais ça cohabite sans OOM pendant que tu utilises la machine.
# (Baisse NODES pour des parties plus rapides — donc plus nombreuses — quitte à
#  ce qu'elles soient un peu plus faibles.)
#
# Verrou trainer PARTAGÉ avec le pretrain -> un SEUL process écrit latest.pt
# (jamais pretrain + online en même temps qui se battraient sur le fichier).
#
# Variables :
#   WORKERS  parties self-play en parallèle (défaut nproc-2 ; baisse si ça rame)
#   NODES    simulations MCTS/coup en self-play (défaut config ; +fort mais +lent)
#   BATCH    batch de l'online (défaut 128 pour tenir sur 8 Go ; 256 = défaut config)
#
# Ctrl-C arrête proprement le self-play ET l'online.
set -u
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3

# Segments extensibles : cohabitation online (+ éventuel bot) sur 8 Go sans OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1 thread BLAS/worker : le forward self-play est séquentiel (pas de sur-souscription).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NCPU="$(nproc 2>/dev/null || echo 2)"
WORKERS="${WORKERS:-$(( NCPU > 2 ? NCPU - 2 : 1 ))}"
[ "$WORKERS" -ge 1 ] 2>/dev/null || WORKERS=1
BATCH="${BATCH:-128}"

SP_ARGS=(--device cpu --workers "$WORKERS")
[ -n "${NODES:-}" ] && SP_ARGS+=(--nodes "$NODES")

echo "[rl-local] self-play CPU (workers=$WORKERS, nodes=${NODES:-config}) + online GPU (batch=$BATCH)"
echo "[rl-local] Ctrl-C pour tout arrêter."

# 1) Self-play CPU en arrière-plan : alimente data/replay_buffer de SES parties.
"$PY" -u -m sanchess.train.selfplay "${SP_ARGS[@]}" &
SP_PID=$!

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "[rl-local] arrêt du self-play (pid $SP_PID)…"
  kill -TERM "$SP_PID" 2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap cleanup INT TERM EXIT

# 2) Online GPU au premier plan, sous verrou trainer (un seul écrivain de latest.pt).
#    --seed-shards : réinjecte des parties de pretrain pour limiter l'oubli (no-op
#    si data/shards est vide ; l'online attend alors que le self-play remplisse
#    le buffer jusqu'à online.min_buffer avant le premier pas de gradient).
flock -n /tmp/sano1-trainer.lock \
  "$PY" -u -m sanchess.train.online --seed-shards data/shards --batch-size "$BATCH" \
  || echo "[rl-local] online arrêté (ou verrou trainer déjà pris : pretrain/online déjà actif ?)." >&2
