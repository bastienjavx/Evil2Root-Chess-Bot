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
#   EVAL_GAMES / EVAL_NODES  match candidat vs latest avant promotion
#   EVAL_DEVICE   appareil de l'arène (auto: cuda si dispo, sinon cpu)
#   EVAL_MIN_STEPS pas de gradient minimaux entre deux candidats (défaut 200)
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
PROMOTE_ONLY_BETTER="${PROMOTE_ONLY_BETTER:-1}"
EVAL_GAMES="${EVAL_GAMES:-12}"
EVAL_NODES="${EVAL_NODES:-${NODES:-80}}"
EVAL_MIN_SCORE="${EVAL_MIN_SCORE:-0.5}"
EVAL_MIN_STEPS="${EVAL_MIN_STEPS:-200}"
# Éval de promotion sur GPU PAR DÉFAUT : l'online ne fait qu'un pas de gradient
# toutes les ~2 s (online.step_every_sec) -> le GPU est libre ~99 % du temps.
# L'arène CPU d'un gros réseau (24x320) à 80 nœuds × 12 parties était ULTRA lente
# (elle ne finissait jamais avant le candidat suivant). Sur GPU elle prend des
# secondes. Repli automatique sur CPU si CUDA indisponible. Forcer : EVAL_DEVICE=cpu.
if [ -z "${EVAL_DEVICE:-}" ]; then
  if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    EVAL_DEVICE=cuda
  else
    EVAL_DEVICE=cpu
  fi
fi

SP_ARGS=(--device cpu --workers "$WORKERS")
[ -n "${NODES:-}" ] && SP_ARGS+=(--nodes "$NODES")

ONLINE_ARGS=(--seed-shards data/shards --batch-size "$BATCH")
if [ "$PROMOTE_ONLY_BETTER" != "0" ]; then
  ONLINE_ARGS+=(
    --promote-only-better
    --promotion-games "$EVAL_GAMES"
    --promotion-nodes "$EVAL_NODES"
    --promotion-device "$EVAL_DEVICE"
    --promotion-min-score "$EVAL_MIN_SCORE"
    --promotion-min-steps "$EVAL_MIN_STEPS"
  )
fi

echo "[rl-local] self-play CPU (workers=$WORKERS, nodes=${NODES:-config}) + online GPU (batch=$BATCH)"
if [ "$PROMOTE_ONLY_BETTER" != "0" ]; then
  echo "[rl-local] promotion: candidat vs latest (arène EN PARALLÈLE de l'entraînement),"
  echo "[rl-local]            games=$EVAL_GAMES nodes=$EVAL_NODES device=$EVAL_DEVICE min_score>$EVAL_MIN_SCORE min_steps=$EVAL_MIN_STEPS"
fi
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
  "$PY" -u -m sanchess.train.online "${ONLINE_ARGS[@]}" \
  || echo "[rl-local] online arrêté (ou verrou trainer déjà pris : pretrain/online déjà actif ?)." >&2
