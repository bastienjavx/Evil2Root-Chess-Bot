#!/usr/bin/env bash
# Boucle de SUIVI DE FORCE pour décider OBJECTIVEMENT quand arrêter le RL.
#
# Au démarrage on FIGE une copie de latest.pt comme référence (checkpoints/
# ref_strength.pt). Puis, toutes les $INTERVAL secondes, on fait jouer le
# latest.pt courant contre cette référence figée (scripts/arena.py) et on logge
# l'écart d'Elo + le step. Tant que l'Elo MONTE, le réseau progresse encore ;
# quand il PLAFONNE sur 2-3 mesures, c'est le moment de couper la boucle RL.
#
#   bash scripts/track_strength.sh
#   INTERVAL=1800 GAMES=60 NODES=200 bash scripts/track_strength.sh
#
# Variables : INTERVAL (s, défaut 3600), GAMES, NODES, LATEST, REF, LOG.
set -u
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3

INTERVAL="${INTERVAL:-3600}"
GAMES="${GAMES:-40}"
NODES="${NODES:-200}"
LATEST="${LATEST:-checkpoints/latest.pt}"
REF="${REF:-checkpoints/ref_strength.pt}"
LOG="${LOG:-checkpoints/strength.log}"

if [ ! -f "$LATEST" ]; then
    echo "[track] introuvable: $LATEST (le trainer a-t-il déjà écrit un checkpoint ?)" >&2
    exit 1
fi

# Fige la référence UNE fois (ne pas l'écraser si on relance le suivi).
if [ ! -f "$REF" ]; then
    cp "$LATEST" "$REF"
    echo "[track] référence figée -> $REF" | tee -a "$LOG"
fi

echo "[track] latest=$LATEST vs ref=$REF  interval=${INTERVAL}s games=$GAMES nodes=$NODES" \
    | tee -a "$LOG"
echo "[track] log -> $LOG (Elo POSITIF = latest plus fort que la référence)" | tee -a "$LOG"

while true; do
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    # On capte UNIQUEMENT la ligne RESULT (stdout) ; la progression va sur stderr.
    line="$("$PY" scripts/arena.py --a "$LATEST" --b "$REF" \
            --games "$GAMES" --nodes "$NODES" 2>/dev/null \
            | grep '^RESULT')"
    if [ -n "$line" ]; then
        echo "[$ts] $line" | tee -a "$LOG"
    else
        echo "[$ts] (arena: aucun RESULT — checkpoint en cours d'écriture ?)" | tee -a "$LOG"
    fi
    sleep "$INTERVAL"
done
