#!/usr/bin/env bash
# Pré-entraînement CLOUD (L40S / H100) : grosse archi 24x320, batch 1024, via
# config.cloud.yaml. À lancer sur l'instance Scaleway, PAS sur la 2070S locale.
# Le checkpoint produit (checkpoints/latest.pt) embarque son archi -> se recharge
# tel quel sur la machine locale (cf. CLOUD.md).
cd "$(dirname "$0")/.." || exit 1
PY="$(pwd)/.venv/bin/python"; [ -x "$PY" ] || PY=python3
# Pas de cohabitation bot/bureau sur le cloud : on laisse l'allocateur par défaut
# (expandable_segments servait à limiter le pic sur les 8 Go locaux).
# TF32 = matmuls plus rapides sur Ampere/Ada/Hopper sans perte sensible.
export NVIDIA_TF32_OVERRIDE=1
# Verrou anti-double-instance (un seul process écrit latest.pt à la fois).
exec flock -n /tmp/sano1-trainer.lock "$PY" -m sanchess.train.pretrain \
  --config config.cloud.yaml "$@"
