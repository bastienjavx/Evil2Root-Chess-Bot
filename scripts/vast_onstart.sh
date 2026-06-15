#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# On-start Vast.ai — rejoint le cluster San-o1 comme worker de self-play (GPU).
#
# Colle ce script dans le champ "On-start Script" d'une template Vast.ai (voir
# VAST.md). Il clone le dépôt, installe les dépendances manquantes (torch+CUDA
# sont déjà dans l'image de base) puis lance le worker en tâche de fond.
#
# Réglage via VARIABLES D'ENVIRONNEMENT de la template Vast (onglet "Env") :
#   SANO1_SERVER         URL du coordinateur (défaut: coordinateur officiel ci-dessous)
#   SANO1_WORKER_NAME    pseudo affiché au leaderboard          (défaut: vast-$HOSTNAME)
#   SANO1_REPO           URL git du dépôt                        (défaut: dépôt officiel)
#   SANO1_BRANCH         branche à cloner                        (défaut: main)
#   SANO1_WORKER_GPU_GAMES   parties GPU batchées                (défaut: auto)
#   SANO1_WORKER_WORKERS     process de self-play en parallèle   (défaut: auto)
#   SANO1_ALLOW_CPU      "1" pour autoriser le repli CPU si pas de CUDA (défaut: refuse)
#   ... + toutes les SANO1_WORKER_* reconnues par worker.py (device, threads, etc.)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="${SANO1_REPO:-https://github.com/bastienjavx/Evil2Root-Chess-Bot.git}"
BRANCH="${SANO1_BRANCH:-main}"
DIR="${SANO1_DIR:-/workspace/San-o1}"
LOG="${SANO1_LOG:-/workspace/sano1_worker.log}"
SANO1_SERVER="${SANO1_SERVER:-https://evil2root-chess-bot-production.up.railway.app}"

export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }

# Clone (ou met à jour si l'instance redémarre sur le même volume).
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch --depth 1 origin "$BRANCH" && git -C "$DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"

# Dépendances du worker SANS toucher à torch (déjà fourni par l'image PyTorch de
# Vast). On installe juste le reste pour ne pas réinstaller CUDA inutilement.
python -m pip install --no-cache-dir -q \
  "python-chess>=1.999" "pyyaml>=6.0.3" "requests>=2.31" "numpy>=1.26" "zstandard>=0.22" "tqdm>=4.68.2"

NAME="${SANO1_WORKER_NAME:-vast-$(hostname)}"

# ── Préflight CUDA ────────────────────────────────────────────────────────────
# Sur un GPU Vast payant, lancer du self-play sur CPU est du gaspillage : si torch
# ne voit pas le GPU, le worker en --device auto retombe SILENCIEUSEMENT sur CPU.
# On vérifie ici et on s'arrête net (sauf SANO1_ALLOW_CPU=1) avec un diagnostic.
echo "[vast_onstart] préflight CUDA…" | tee -a "$LOG"
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
    echo "[vast_onstart] ❌ CUDA INDISPONIBLE pour torch sur cette image."
    echo "    Cause probable : image sans torch+CUDA, ou driver hôte trop ancien."
    echo "    Utilise une image PyTorch CUDA, p.ex. : pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
    echo "    (driver hôte requis : compatible avec la version CUDA de l'image)."
    echo "    Pour forcer quand même le CPU (déconseillé sur GPU payant) : SANO1_ALLOW_CPU=1"
  } | tee -a "$LOG"
  if [ "${SANO1_ALLOW_CPU:-0}" != "1" ]; then
    echo "[vast_onstart] worker NON lancé (évite de payer du GPU pour du CPU)." | tee -a "$LOG"
    exit 1
  fi
  echo "[vast_onstart] SANO1_ALLOW_CPU=1 → lancement sur CPU malgré tout." | tee -a "$LOG"
fi

echo "[vast_onstart] San-o1 worker → $SANO1_SERVER (name=$NAME, device=${SANO1_WORKER_DEVICE:-auto})" | tee -a "$LOG"
echo "[vast_onstart] logs: $LOG  | suivi: tail -f $LOG"

# Lancement en tâche de fond : l'onstart se termine, le worker continue.
# nice par défaut (worker.py: --nice 10) ; reconnexion + backoff automatiques.
nohup python -m sanchess.cluster.worker \
  --server "$SANO1_SERVER" \
  --name "$NAME" \
  --device "${SANO1_WORKER_DEVICE:-auto}" \
  >> "$LOG" 2>&1 &

echo "[vast_onstart] worker lancé (PID $!)."
