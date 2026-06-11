#!/usr/bin/env bash
# Lance l'interface web San-o1 (API REST + WebSocket + frontend) et, si
# cloudflared est installé, l'expose publiquement via un tunnel Cloudflare.
#
#   ./scripts/run_web.sh                 # local : http://localhost:8000
#   CLOUDFLARED=1 ./scripts/run_web.sh   # + tunnel public (URL trycloudflare.com)
#   PORT=8080 ./scripts/run_web.sh       # autre port
#
# Variables : HOST (def 0.0.0.0), PORT (def 8000), CLOUDFLARED (1 = tunnel).
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Python du venv (les deps torch/chess/fastapi n'y sont QUE) sinon fallback système.
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi

# Verrou mono-instance : deux serveurs sur le même port échoueraient de toute façon.
exec 9>/tmp/sano1-web.lock
if ! flock -n 9; then
  echo "Une interface web San-o1 tourne déjà (verrou /tmp/sano1-web.lock)." >&2
  exit 1
fi

CF_PID=""
cleanup(){ [ -n "$CF_PID" ] && kill "$CF_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [ "${CLOUDFLARED:-0}" = "1" ]; then
  if command -v cloudflared >/dev/null 2>&1; then
    echo ">> Démarrage du tunnel Cloudflare vers http://localhost:${PORT} ..."
    cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT}" 2>&1 \
      | grep --line-buffered -Eo 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' &
    CF_PID=$!
    echo ">> (l'URL publique s'affichera ci-dessus dès qu'elle est prête)"
  else
    echo "!! cloudflared introuvable. Installez-le :" >&2
    echo "   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared" >&2
  fi
fi

echo ">> Interface San-o1 sur http://${HOST}:${PORT}"
exec env SANO1_WEB_HOST="$HOST" SANO1_WEB_PORT="$PORT" "$PY" -m sanchess.web
