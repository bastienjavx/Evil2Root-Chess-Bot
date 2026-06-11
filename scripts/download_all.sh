#!/usr/bin/env bash
# Téléchargement massif et REPRENABLE des parties Lichess (Elo moyen >= 2000).
#
# Stream chaque dump mensuel database.lichess.org du plus récent au plus ancien,
# conversion à la volée en shards (data/shards). Chaque mois terminé pose un
# marqueur .done_<mois> : relancer le script reprend là où il s'était arrêté.
#
# Usage :  scripts/download_all.sh            # tout, jusqu'à arrêt manuel
#          START=2024-12 END=2024-01 scripts/download_all.sh   # plage précise
#
# Arrêt : Ctrl-C, ou  pkill -f sanchess.data.pgn_to_samples
set -u
cd "$(dirname "$0")/.." || exit 1

# Verrou anti-double-instance : deux téléchargements du même mois écriraient des
# shards en collision. On se ré-exécute une seule fois sous flock (échec immédiat
# si un autre run, manuel ou service, tient déjà le verrou).
if [[ -z "${_SANO1_DL_FLOCKED:-}" ]]; then
  exec env _SANO1_DL_FLOCKED=1 flock -n /tmp/sano1-download.lock "$0" "$@"
fi

# Interpréteur du venv (zstandard/chess n'y sont QUE dans le venv).
PY="$(pwd)/.venv/bin/python"; [[ -x "$PY" ]] || PY=python3

OUT="data/shards"
mkdir -p "$OUT"

# Plage de mois (du plus récent au plus ancien). Lichess publie depuis 2013-01.
START="${START:-2026-05}"   # mois le plus récent à tenter (404 => ignoré)
END="${END:-2013-01}"       # mois le plus ancien

# Itère AAAA-MM de START à END en descendant.
y=${START%-*}; m=${START#*-}; m=$((10#$m))
ey=${END%-*}; em=${END#*-}; em=$((10#$em))

while (( y > ey || (y == ey && m >= em) )); do
  MONTH=$(printf "%04d-%02d" "$y" "$m")
  MARK="$OUT/.done_${MONTH}"
  if [[ -f "$MARK" ]]; then
    echo "[$(date +%H:%M:%S)] $MONTH déjà fait, skip."
  else
    URL="https://database.lichess.org/standard/lichess_db_standard_rated_${MONTH}.pgn.zst"
    echo "[$(date +%H:%M:%S)] === $MONTH : stream $URL ==="
    if "$PY" -m sanchess.data.pgn_to_samples "$URL" --out "$OUT"; then
      touch "$MARK"
      echo "[$(date +%H:%M:%S)] $MONTH TERMINÉ."
    else
      echo "[$(date +%H:%M:%S)] $MONTH échoué/indisponible (peut-être pas encore publié), on continue."
    fi
  fi
  # mois précédent
  m=$((m-1)); if (( m < 1 )); then m=12; y=$((y-1)); fi
done
echo "[$(date +%H:%M:%S)] === TOUS LES MOIS PARCOURUS ==="
