#!/usr/bin/env bash
# Entraînement distribué multi-appareils (Mac M1 + Linux), SGD synchrone via gloo.
#
# À lancer SUR CHAQUE MACHINE, avec le même WORLD_SIZE et le même MASTER_ADDR/PORT.
# Le rang 0 est le "master" : c'est lui qui écrit checkpoints/latest.pt.
#
# Variables d'environnement :
#   MASTER_ADDR  IP du rang 0 sur le réseau local      (défaut 127.0.0.1)
#   MASTER_PORT  port de rendez-vous                   (défaut 29500)
#   WORLD_SIZE   nombre total de machines              (défaut 1)
#   RANK         rang de CETTE machine (0..WORLD_SIZE-1)
#   BACKEND      gloo (défaut, cross-platform) | nccl  (Linux multi-GPU)
#   DEVICE       appareil de CE rang (cuda/mps/cpu)    (défaut: train.device)
#
# Exemples (cluster de 2 machines, master = Linux 192.168.1.10) :
#   # Linux (rang 0) :
#   MASTER_ADDR=192.168.1.10 WORLD_SIZE=2 RANK=0 ./scripts/run_distributed.sh
#   # Mac M1 (rang 1) :
#   MASTER_ADDR=192.168.1.10 WORLD_SIZE=2 RANK=1 ./scripts/run_distributed.sh
#
# GPU + CPU sur UNE SEULE machine (2 terminaux, world_size=2, master local) :
#   WORLD_SIZE=2 RANK=0 DEVICE=cuda ./scripts/run_distributed.sh   # rang GPU (master)
#   WORLD_SIZE=2 RANK=1 DEVICE=cpu  ./scripts/run_distributed.sh   # rang CPU
set -u
cd "$(dirname "$0")/.." || exit 1

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export DIST_BACKEND="${BACKEND:-gloo}"
# Appareil de CE rang (cuda/mps/cpu). Vide => train.device du config.yaml.
export DEVICE="${DEVICE:-}"

# Python du venv si présent (torch/chess n'y sont QUE dans le venv).
PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3

echo "[distribué] rang $RANK/$WORLD_SIZE | master=$MASTER_ADDR:$MASTER_PORT | backend=$DIST_BACKEND | device=${DEVICE:-config}"

# Verrou « trainer » : seul le rang 0 écrit checkpoints/latest.pt, donc lui seul
# prend le verrou exclusif (distribué XOR pretrain XOR online). Les autres rangs
# LOCAUX (ex. le rang CPU sur la même machine) ne l'écrivent pas -> pas de verrou,
# sinon le `flock -n` les ferait échouer immédiatement.
if [ "$RANK" = "0" ]; then
    exec flock -n /tmp/sano1-trainer.lock \
        "$PY" -u -m sanchess.train.distributed "$@"
else
    exec "$PY" -u -m sanchess.train.distributed "$@"
fi
