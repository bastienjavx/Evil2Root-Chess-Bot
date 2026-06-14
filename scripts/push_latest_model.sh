#!/usr/bin/env bash
# Upload checkpoints/latest.pt to a GitHub Release when it changes.
set -Eeuo pipefail

cd "$(dirname "$0")/.." || exit 1

MODEL_PATH="${MODEL_PATH:-checkpoints/latest.pt}"
TAG="${GITHUB_MODEL_TAG:-model-latest}"
TITLE="${GITHUB_MODEL_TITLE:-San-o1 latest model}"
NOTES="${GITHUB_MODEL_NOTES:-Latest San-o1 checkpoint uploaded automatically.}"
ASSET_NAME="${GITHUB_MODEL_ASSET_NAME:-latest.pt}"
STATE_PATH="${STATE_PATH:-checkpoints/.github_latest.sha256}"
LOCK_PATH="${LOCK_PATH:-/tmp/sano1-github-model-push.lock}"
STABILITY_DELAY="${STABILITY_DELAY:-15}"

log() {
    printf '[model-push] %s\n' "$*"
}

if [ ! -f "$MODEL_PATH" ]; then
    log "skip: missing $MODEL_PATH"
    exit 0
fi

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    log "skip: another push is already running"
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

size_before="$(stat -c '%s' "$MODEL_PATH")"
mtime_before="$(stat -c '%Y' "$MODEL_PATH")"
sleep "$STABILITY_DELAY"
size_after="$(stat -c '%s' "$MODEL_PATH")"
mtime_after="$(stat -c '%Y' "$MODEL_PATH")"

if [ "$size_before" != "$size_after" ] || [ "$mtime_before" != "$mtime_after" ]; then
    log "skip: $MODEL_PATH changed while checking stability"
    exit 0
fi

sha="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
old_sha=""
[ -f "$STATE_PATH" ] && old_sha="$(tr -d '[:space:]' < "$STATE_PATH")"

if [ "$sha" = "$old_sha" ]; then
    log "skip: unchanged $MODEL_PATH ($sha)"
    exit 0
fi

if ! gh release view "$TAG" >/dev/null 2>&1; then
    log "creating release $TAG"
    gh release create "$TAG" --title "$TITLE" --notes "$NOTES" --latest
fi

log "uploading $MODEL_PATH as release asset $ASSET_NAME on tag $TAG"
gh release upload "$TAG" "$MODEL_PATH#$ASSET_NAME" --clobber
printf '%s\n' "$sha" > "$STATE_PATH"
log "done: uploaded $ASSET_NAME ($sha)"
