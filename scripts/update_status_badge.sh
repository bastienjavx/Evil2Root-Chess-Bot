#!/usr/bin/env bash
# Pousse l'état d'entraînement (training + machine) vers le gist GitHub
# qui alimente les badges shields.io du README. Best-effort, jamais fatal.
set -Eeuo pipefail

cd "$(dirname "$0")/.." || exit 1

LOCK_PATH="${LOCK_PATH:-/tmp/sano1-status-badge.lock}"

log() { printf '[status-badge] %s\n' "$*"; }

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    log "skip: another update is already running"
    exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
    log "error: GitHub CLI 'gh' is required"
    exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
    log "error: GitHub CLI is not authenticated; run 'gh auth login'"
    exit 1
fi

PY="$(pwd)/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

exec "$PY" scripts/update_status_badge.py "$@"
