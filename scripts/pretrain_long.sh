#!/usr/bin/env bash
# ============================================================================
# Pré-entraînement supervisé ÉTENDU (~1 jour) — phase 1 du plan multi-jours.
#
# Le pretrain v3 a été coupé au step ~42000 (≈0,14 epoch) sur un plateau à LR
# plein : le réseau n'a vu qu'une fraction des données et n'a jamais eu sa
# descente de LR. Ce script reprend depuis le net de pretrain (step_42000.pt) et
# pousse jusqu'à HORIZON steps avec le cosinus du config calé sur TOUT l'horizon
# -> entraînement à LR ~1e-3 pendant ~1 epoch puis annealing naturel jusqu'au
# plancher 5e-5. Total ≈ 2,3 epochs, ≈ 24 h sur la 2070S (~7,5 steps/s).
#
# Phase 2 (à lancer À LA MAIN une fois CE pretrain terminé) : RL self-play via le
# cluster, re-seedé depuis le latest.pt annealé. Cf. le bloc final affiché.
#
# IMPORTANT : ce script NE LANCE PAS l'entraînement. Il prépare l'état (sauvegarde
# + bascule de latest.pt + config dédié) puis AFFICHE la commande à lancer.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3

HORIZON=700000                      # pretrain_steps : ~24 h / ~2,3 epochs depuis 42k.
                                    #   600000 ≈ 20 h / ~2 ep | 800000 ≈ 28 h / ~2,6 ep
SEED=checkpoints/step_42000.pt      # net de pretrain à prolonger (step embarqué = 42000)
LATEST=checkpoints/latest.pt
CFG_SRC=config.yaml
CFG_OUT=config.pretrain-long.yaml

# --- 1. Aucun autre writer de latest.pt ne doit tourner (sinon corruption) ----
# pretrain/online partagent ce flock ; le trainer du cluster, lui, NON -> check à part.
if ! flock -n /tmp/sano1-trainer.lock true 2>/dev/null; then
  echo "ABANDON : /tmp/sano1-trainer.lock tenu -> un pretrain/online tourne déjà." >&2
  exit 1
fi
if pgrep -af "sanchess\.cluster\.traine[r]" >/dev/null; then
  echo "ABANDON : un sanchess.cluster.trainer tourne et écrit latest.pt. Stoppe-le d'abord." >&2
  exit 1
fi

# --- 2. Le net de pretrain doit exister --------------------------------------
[ -f "$SEED" ] || { echo "ABANDON : $SEED introuvable." >&2; exit 1; }

# --- 3. Sauvegarder le latest.pt courant (gains RL du cluster) avant bascule --
if [ -f "$LATEST" ]; then
  CUR_STEP=$("$PY" - "$LATEST" <<'EOF'
import sys, torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))
EOF
)
  BAK="checkpoints/cluster_rl_step${CUR_STEP}.pt"
  cp -f "$LATEST" "$BAK"
  echo "Sauvegarde du latest.pt courant (step $CUR_STEP, RL cluster) -> $BAK"
fi

# --- 4. Préserver la graine de pretrain hors du motif de rotation step_*.pt ----
# (_prune_checkpoints ne garde que les 20 derniers step_*.pt : ce nom-ci y échappe.)
cp -f "$SEED" checkpoints/pretrain_seed_step42000.pt
echo "Graine préservée -> checkpoints/pretrain_seed_step42000.pt"

# --- 5. Repositionner latest.pt sur le net de pretrain (step 42000) -----------
# Le LR ne dépend QUE du step embarqué : on repart ainsi à ~1e-3 et on anneale.
cp -f "$SEED" "$LATEST"
echo "Bascule : $SEED -> $LATEST (step 42000, optimiseur inclus)"

# --- 6. Config étendu : copie de config.yaml, seul pretrain_steps change -------
cp -f "$CFG_SRC" "$CFG_OUT"
sed -i -E "s/^([[:space:]]*pretrain_steps:)[[:space:]]*[0-9_]+/\1 ${HORIZON}/" "$CFG_OUT"
echo "Config étendu écrit : $CFG_OUT (pretrain_steps=${HORIZON})"
grep -nE "^[[:space:]]*pretrain_steps:" "$CFG_OUT" | sed 's/^/  /'

cat <<EOF

=============================================================================
Prêt. État préparé, RIEN n'a été lancé.

PHASE 1 — pretrain étendu (~24 h, survit à la fermeture du terminal) :

    nohup scripts/run_pretrain.sh --config ${CFG_OUT} \\
          > data/pretrain_long.log 2>&1 &
    tail -f data/pretrain_long.log          # suivi LR / loss

  Le bot peut rester allumé en parallèle (batch 256 + bot tiennent sur le 8 Go).
  NE relance PAS le cluster.trainer pendant cette phase (2 writers = corruption).

PHASE 2 — RL cluster (quand la phase 1 atteint le step ${HORIZON} et SORT) :

    set -a; . ./.env; set +a    # exporte TRAINER_TOKEN depuis .env (git-ignoré)
    .venv/bin/python -m sanchess.cluster.trainer \\
        --server https://evil2root-chess-bot-production.up.railway.app/
    # (re-seede tout seul depuis le latest.pt annealé ; trainer.py lit TRAINER_TOKEN
    #  dans l'env -> NE colle PLUS le token en clair sur la ligne de commande)

Annuler la prépa / revenir au net du cluster :
    cp ${BAK:-<backup>} ${LATEST}
=============================================================================
EOF
