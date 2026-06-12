# Entraîner sur le cloud (L40S / H100), faire tourner en local (RTX 2070S 8 Go)

Objectif : pretrain rapide sur un gros GPU Scaleway, puis ramener le checkpoint
qui **tourne ici** sans surprise. Le réseau cloud est `24x320` (sweet spot validé
par `scripts/bench_eval.py` : encore jouable en blitz sur la 2070S).

## Pourquoi ça « remarche » tel quel en local

Le checkpoint **embarque son archi** (`model_cfg`), et `build_model_from_checkpoint`
(sanchess/model.py) la relit pour reconstruire le réseau. Donc le bot/online/web
local rechargent à chaud un `latest.pt` 24x320 **sans toucher au config.yaml
local**. La seule vraie contrainte est la **compat des versions** (ci-dessous).

## 0. Reproductibilité (le point qui casse silencieusement)

Aligne l'environnement cloud sur le local pour qu'un `torch.load` ne coince pas :

| | Local (mesuré) | Cloud (à reproduire) |
|---|---|---|
| Python | 3.12.3 | 3.12.x |
| PyTorch | `2.12.0+cu130` | `>=2.12` (même majeure) |
| CUDA (wheel) | 13.0 | 12.x/13.x selon le driver de l'instance |

Un `state_dict` reste lisible across versions mineures de torch, mais reste sur la
**même majeure** pour éviter les surprises (formats d'optimiseur, etc.).

```bash
# sur l'instance cloud
git clone <repo> && cd San-o1
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu130   # ou cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi   # vérifie L40S / H100 vu
```

## 1. Données

Deux options :

- **Rapatrier tes shards locaux** (déjà filtrés Elo≥2000) — le plus rapide pour
  reprendre l'entraînement existant :
  ```bash
  rsync -avP /media/evil2root/8TB/sano1_data/shards/  user@INSTANCE:San-o1/data/shards/
  ```
- **Re-télécharger sur le cloud** (bande passante datacenter) :
  ```bash
  ./scripts/download_all.sh        # dumps mensuels lichess.org, filtre Elo≥2000
  ```

## 2. Pretrain cloud

```bash
./scripts/run_pretrain_cloud.sh
# = python -m sanchess.train.pretrain --config config.cloud.yaml
# 24x320, batch 1024, LR 0.002, num_workers 16, amp fp16, 1M steps.
```

Sur **H100 80 Go** tu peux pousser dans `config.cloud.yaml` : `train.batch_size`
2048-4096 + `num_workers` 24+ (sinon le DataLoader affame le GPU). Surveille
`nvidia-smi` : si util GPU < 90 %, le goulot est le CPU/DataLoader, pas le GPU.

**Multi-GPU** (8x H100 sur une instance) : `distributed.backend: nccl` déjà mis ;
lance via `sanchess.train.distributed` (un rang par GPU, `--device cuda:N`).

## 3. Rapatrier le checkpoint et le faire tourner ici

```bash
# depuis la machine locale
rsync -avP user@INSTANCE:San-o1/checkpoints/latest.pt  checkpoints/latest.pt
```

Le bot le recharge **à chaud** (mtime de `latest.pt`) — pas besoin de redémarrer
pour les poids. Mais le 24x320 est plus lourd : redémarre quand même les services
pour repartir propre et re-mesurer :

```bash
sudo systemctl restart sano1-bot
.venv/bin/python scripts/bench_eval.py --sizes 24x320   # confirme le débit local
```

Optionnel mais conseillé pour la cohérence (re-build d'un réseau neuf en local) :
recopie la section `model:` de `config.cloud.yaml` dans `config.yaml` local.

## 4. Garde-fous

- **Ne pas dépasser 24x320 pour le temps-réel.** 30x384/40x512 = node-starved en
  blitz sur la 2070S (cf. table de `bench_eval.py`). Réserve-les à l'analyse/web.
- **Coût** : le pretrain est le gros poste. La boucle RL/self-play/online continue
  ensuite EN LOCAL gratuitement (le cloud ne sert qu'à l'amorçage fort).
- **VRAM locale** : 24x320 ~0,4 Go en éval, aucun risque sur 8 Go même avec
  bot + online + bureau.
