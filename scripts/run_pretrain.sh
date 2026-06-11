#!/usr/bin/env bash
# Pré-entraînement supervisé sur les shards Lichess (data/shards par défaut).
cd "$(dirname "$0")/.." || exit 1
# Interpréteur du venv (chess/torch n'y sont QUE dans le venv).
PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3
# Verrou anti-double-instance : deux entraînements écriraient le même
# checkpoints/latest.pt et se corrompraient. flock -n échoue tout de suite si
# un autre run (manuel ou service) tient déjà le verrou.
exec flock -n /tmp/sano1-pretrain.lock "$PY" -m sanchess.train.pretrain "$@"
