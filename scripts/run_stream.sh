#!/usr/bin/env bash
# Collecte continue de parties Lichess -> replay buffer (alimente l'online).
cd "$(dirname "$0")/.." || exit 1
# Interpréteur du venv (chess/requests n'y sont QUE dans le venv).
PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3
# Verrou anti-double-instance (deux streams dupliqueraient les parties).
exec flock -n /tmp/sano1-stream.lock "$PY" -u -m sanchess.data.stream --perf "${PERF:-blitz}"
