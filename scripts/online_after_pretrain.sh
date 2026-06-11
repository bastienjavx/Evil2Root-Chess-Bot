#!/usr/bin/env bash
# Démarre l'entraînement continu (online.py) UNIQUEMENT quand le pré-entraînement
# est terminé, pour éviter que deux processus écrivent checkpoints/latest.pt en
# même temps. À utiliser avec stream.py qui alimente le replay buffer en parallèle.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "[online-différé] attente de la fin du pré-entraînement…"
while pgrep -f "sanchess.train.pretrain" >/dev/null 2>&1; do
    sleep 30
done
echo "[online-différé] pré-entraînement terminé -> démarrage de l'apprentissage continu"

# Seed avec les shards de pretrain (anti-oubli) + ingestion du replay buffer live.
# Verrou « trainer » PARTAGÉ avec le pretrain : garantit qu'un seul process écrit
# checkpoints/latest.pt (couvre la course où systemd relancerait le pretrain).
exec flock -n /tmp/sano1-trainer.lock "$PY" -u -m sanchess.train.online --seed-shards data/shards
