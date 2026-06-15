#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# On-start Vast.ai — rejoint le cluster San-o1 comme TRAINER (federated averaging).
#
# Contrairement au worker (self-play, accès ouvert), le trainer ENTRAÎNE le modèle
# et PUBLIE des poids : il exige donc `TRAINER_TOKEN` (le même que le coordinateur).
# Plusieurs instances Vast lancées avec ce script et le MÊME token entraînent le
# même modèle en parallèle ; leurs poids sont moyennés à chaque round (cf. CLUSTER.md
# §2bis et `sanchess/cluster/fedavg.py`).
#
# Colle ce script dans le champ "On-start Script" d'une template Vast.ai (voir VAST.md).
#
# VARIABLES D'ENVIRONNEMENT de la template Vast (onglet "Env") :
#   TRAINER_TOKEN        (REQUIS) secret partagé avec le coordinateur — publie le modèle
#   SANO1_SERVER         URL du coordinateur            (défaut: coordinateur officiel)
#   SANO1_TRAINER_ID     identité stable de ce trainer  (défaut: vast-$HOSTNAME)
#   SANO1_FED_LOCAL_STEPS  pas de gradient locaux/round (défaut: 400)
#   SANO1_FED_BATCH_SIZE   surcharge online.batch_size  (défaut: celui du config)
#   SANO1_CONFIG         fichier de config              (défaut: config.cloud.yaml)
#   SANO1_SEED_SHARDS    dossier de shards d'amorçage   (défaut: aucun)
#   SANO1_REPO/BRANCH/DIR/LOG   dépôt git / chemins      (défauts officiels)
#   SANO1_ALLOW_CPU      "1" pour autoriser le repli CPU si pas de CUDA (défaut: refuse)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="${SANO1_REPO:-https://github.com/bastienjavx/Evil2Root-Chess-Bot.git}"
BRANCH="${SANO1_BRANCH:-main}"
DIR="${SANO1_DIR:-/workspace/San-o1}"
LOG="${SANO1_LOG:-/workspace/sano1_trainer.log}"
SANO1_SERVER="${SANO1_SERVER:-https://evil2root-chess-bot-production.up.railway.app}"
SANO1_CONFIG="${SANO1_CONFIG:-config.cloud.yaml}"

# Anti-fragmentation VRAM (cohérent avec run_pretrain.sh).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Le token est OBLIGATOIRE pour un trainer : sans lui, aucune publication possible.
if [ -z "${TRAINER_TOKEN:-}" ]; then
  echo "[vast_trainer] ❌ TRAINER_TOKEN manquant : renseigne-le dans l'onglet Env de la" >&2
  echo "    template (le MÊME secret que le coordinateur). Trainer NON lancé." >&2
  exit 1
fi
export TRAINER_TOKEN          # lu par trainer.py (jamais passé en argv -> pas dans ps)

export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }

# Clone (ou met à jour si l'instance redémarre sur le même volume).
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch --depth 1 origin "$BRANCH" && git -C "$DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"

# Dépendances SANS toucher à torch (déjà dans l'image PyTorch de Vast).
python -m pip install --no-cache-dir -q \
  "python-chess>=1.999" "pyyaml>=6.0.3" "requests>=2.31" "numpy>=1.26" "zstandard>=0.22" "tqdm>=4.68.2"

TRAINER_ID="${SANO1_TRAINER_ID:-vast-$(hostname)}"

# ── Préflight CUDA ────────────────────────────────────────────────────────────
# Entraîner sur CPU un GPU Vast payant est encore plus gaspilleur que du self-play.
echo "[vast_trainer] préflight CUDA…" | tee -a "$LOG"
echo "── nvidia-smi ──" | tee -a "$LOG"
(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || echo "nvidia-smi indisponible") | tee -a "$LOG"
CUDA_OK="$(python - <<'PY'
try:
    import torch
    print("yes" if torch.cuda.is_available() else "no")
    print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
except Exception as e:
    print("no"); print(f"import torch a échoué: {e}")
PY
)"
echo "$CUDA_OK" | tee -a "$LOG"

if ! echo "$CUDA_OK" | head -1 | grep -q "^yes$"; then
  {
    echo "[vast_trainer] ❌ CUDA INDISPONIBLE pour torch sur cette image."
    echo "    Utilise une image PyTorch CUDA, p.ex. : pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
    echo "    (driver hôte requis : compatible avec la version CUDA de l'image)."
    echo "    Pour forcer quand même le CPU (déconseillé sur GPU payant) : SANO1_ALLOW_CPU=1"
  } | tee -a "$LOG"
  if [ "${SANO1_ALLOW_CPU:-0}" != "1" ]; then
    echo "[vast_trainer] trainer NON lancé (évite de payer du GPU pour du CPU)." | tee -a "$LOG"
    exit 1
  fi
  echo "[vast_trainer] SANO1_ALLOW_CPU=1 → entraînement sur CPU malgré tout." | tee -a "$LOG"
fi

# Arguments optionnels (uniquement si la variable est renseignée).
EXTRA=()
[ -n "${SANO1_CONFIG:-}" ]         && EXTRA+=(--config "$SANO1_CONFIG")
[ -n "${SANO1_FED_BATCH_SIZE:-}" ] && EXTRA+=(--fed-batch-size "$SANO1_FED_BATCH_SIZE")
[ -n "${SANO1_SEED_SHARDS:-}" ]    && EXTRA+=(--seed-shards "$SANO1_SEED_SHARDS")

echo "[vast_trainer] San-o1 trainer FedAvg → $SANO1_SERVER" | tee -a "$LOG"
echo "[vast_trainer] trainer_id=$TRAINER_ID local_steps=${SANO1_FED_LOCAL_STEPS:-400} config=$SANO1_CONFIG" | tee -a "$LOG"
echo "[vast_trainer] logs: $LOG  | suivi: tail -f $LOG"

# Lancement en tâche de fond (token via l'environnement, pas en argv).
nohup python -m sanchess.cluster.trainer \
  --server "$SANO1_SERVER" \
  --fedavg \
  --trainer-id "$TRAINER_ID" \
  --local-steps "${SANO1_FED_LOCAL_STEPS:-400}" \
  "${EXTRA[@]}" \
  >> "$LOG" 2>&1 &

echo "[vast_trainer] trainer lancé (PID $!)."
