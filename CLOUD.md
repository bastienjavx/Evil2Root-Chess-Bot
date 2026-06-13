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

---

# 5. Aller BEAUCOUP plus fort : la boucle RL self-play (cloud)

Le pretrain supervisé plafonne près de l'**imitation du fort humain** (Lichess
Elo≥2000) : c'est un mur de **données**, pas de compute. Pour dépasser ce plafond,
on enchaîne sur une **boucle RL self-play** (style AlphaZero) amorcée sur le réseau
supervisé. C'est là que le GPU cloud paie vraiment : il faut générer BEAUCOUP de
parties à haut nombre de nœuds.

```
   selfplay_gpu  ──écrit──>  data/replay_buffer  ──ingéré par──>  online.py
        ▲                                                            │
        └────────── hot-reload checkpoints/latest.pt ◄──────────────┘
```

Le `selfplay_gpu` joue **des centaines de parties EN PARALLÈLE** et regroupe toutes
les évaluations réseau dans **un seul forward GPU par tour** (parallélisation de
feuilles inter-parties). Sur L40S/H100 il produit des dizaines de fois plus de
parties/s que le self-play CPU — c'est ce débit qui fait progresser le réseau.

```bash
# Tout-en-un (trainer online + générateur self-play GPU, après le pretrain) :
./scripts/run_rl_cloud.sh

# Ou séparément, pour régler le débit :
GAMES=512 NODES=400 ./scripts/run_selfplay_gpu.sh   # sature la VRAM (cf. config)
./scripts/online_after_pretrain.sh                   # le trainer ingère le buffer
```

Réglages dans la section `selfplay_gpu:` de `config.cloud.yaml` :

| Clé | L40S 48 Go | H100 80 Go | Effet |
|---|---|---|---|
| `games` | 384–512 | 768–1024 | parties parallèles (= taille du batch GPU) |
| `leaves_per_game` | 8 | 8 | batch effectif ≈ `games × leaves_per_game` |
| `nodes` | 300–600 | 400–800 | qualité des cibles π (force RL) vs vitesse |

> Surveille `nvidia-smi` : si util GPU < 90 %, le goulot est l'encodage CPU des
> positions (`encode_board`) — augmente `games` ou lance le générateur sur une
> instance CPU costaude séparée qui écrit dans le même `data/replay_buffer`.

Le format de samples est **identique** au reste du pipeline (`fen, coup, valeur,
π=visites`) : `online.py` l'ingère sans modification (la distribution de visites π
est déjà la cible politique via `dense_policy_target`). On rapatrie `latest.pt`
comme au §3 ; le bot le recharge à chaud.

# 6. Convertir le réseau en force RÉELLE sur la 2070S

En blitz, l'Elo ≈ **nœuds/coup** ≈ débit d'éval. Deux accélérateurs, gratuits ou
presque, déjà intégrés :

**Réutilisation d'arbre** (`mcts.tree_reuse: true`, défaut) — entre deux coups, on
reprend le sous-arbre déjà calculé (notre coup + réponse adverse) au lieu de
repartir de zéro. Rien à faire, c'est actif partout (UCI, bot, web). Invalidé
automatiquement au hot-reload des poids.

**Moteur TensorRT fp16** — sur Turing, ~2-3× le débit d'éval -> ~2× de nœuds.
À compiler **sur la 2070S** (un moteur TRT est spécifique au GPU/driver, pas
transférable depuis le cloud) :

```bash
# Option A — TorchScript+TensorRT (le plus simple, branché tel quel) :
python -m sanchess.export --ckpt checkpoints/latest.pt --trt checkpoints/sano1.ts --fp16
# puis dans config.yaml :  model.trt_engine: checkpoints/sano1.ts

# Option B — ONNX + trtexec (si tu préfères onnxruntime-gpu / un .plan natif) :
python -m sanchess.export --ckpt checkpoints/latest.pt --onnx checkpoints/sano1.onnx
# la commande trtexec exacte est affichée par l'export.

# Re-mesurer le gain de débit :
.venv/bin/python scripts/bench_eval.py --sizes 24x320
```

Repli automatique sur PyTorch si `torch_tensorrt`/le fichier sont absents : aucune
régression possible. **Recompiler le moteur à chaque changement d'archi** (un
moteur 20x256 ne convient pas à un 24x320).
