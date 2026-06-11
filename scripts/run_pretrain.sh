#!/usr/bin/env bash
# Pré-entraînement supervisé sur les shards Lichess (data/shards par défaut).
cd "$(dirname "$0")/.." || exit 1
# Interpréteur du venv (chess/torch n'y sont QUE dans le venv).
PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3
# Verrou « trainer » PARTAGÉ avec l'online : un seul process écrit
# checkpoints/latest.pt à la fois (pretrain XOR online), sinon corruption.
# flock -n échoue tout de suite si un autre entraînement tient déjà le verrou.
exec flock -n /tmp/sano1-trainer.lock "$PY" -m sanchess.train.pretrain "$@"
