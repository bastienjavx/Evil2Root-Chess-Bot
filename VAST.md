# San-o1 — Template Vast.ai (rejoindre le cluster en 1 clic)

Cette template loue un GPU sur [Vast.ai](https://vast.ai) et le branche
automatiquement comme **worker de self-play** sur le cluster San-o1
(voir [CLUSTER.md](CLUSTER.md)). Aucun token requis : l'accès worker est ouvert.

## Créer la template

Vast.ai → **Templates → New Template**, puis remplis les champs ci-dessous.

### Image Docker

```
pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
```

> N'importe quelle image PyTorch+CUDA récente convient (torch est déjà dedans,
> l'on-start n'y touche pas). Vérifie la compatibilité CUDA du GPU loué.

### Launch mode

- Coche **Run interactive shell server, SSH** (pratique pour `tail -f` les logs).
- Pas besoin de Jupyter.

Le coordinateur officiel est déjà la valeur par défaut du script
(`https://evil2root-chess-bot-production.up.railway.app`) : tu n'as donc **rien**
à configurer pour rejoindre. Renseigne juste `SANO1_WORKER_NAME` pour apparaître
sous ton pseudo au leaderboard.

### On-start Script

Colle le contenu de [`scripts/vast_onstart.sh`](scripts/vast_onstart.sh).
Il clone le dépôt, installe les dépendances manquantes (sans réinstaller torch)
et lance le worker en tâche de fond.

### Variables d'environnement (onglet Env de la template)

| Variable | Requis | Rôle | Exemple |
|---|:---:|---|---|
| `SANO1_SERVER` | | URL du coordinateur (défaut: coordinateur officiel) | `https://evil2root-chess-bot-production.up.railway.app` |
| `SANO1_WORKER_NAME` | | Pseudo au leaderboard | `bastien-vast-3090` |
| `SANO1_WORKER_GPU_GAMES` | | Budget parties GPU batchées (total, réparti entre process) | `512` |
| `SANO1_WORKER_WORKERS` | | Process self-play parallèles (`auto` recommandé) | `auto` |
| `SANO1_WORKER_VRAM_PER_PROC_GB` | | VRAM estimée par process pour l'auto-sizing | `2.5` |
| `SANO1_BRANCH` | | Branche git à utiliser | `main` |

**Auto-sizing GPU (`SANO1_WORKER_WORKERS=auto`, défaut).** Le MCTS est CPU-bound,
donc un seul process laisse le GPU quasi inactif. En `auto`, le worker **détecte
tous les GPU** et lance autant de process que la **VRAM libre** le permet
(≈ `SANO1_WORKER_VRAM_PER_PROC_GB` par process), borné par les cœurs CPU, et les
**répartit sur tous les GPU**. Ex. un GPU 12 Go → ~4 process ; deux GPU 12 Go →
jusqu'à ~8, étalés. Baisse `SANO1_WORKER_VRAM_PER_PROC_GB` pour en lancer plus,
ou fixe un nombre explicite avec `SANO1_WORKER_WORKERS=N`.

Toutes les options de `worker.py` sont pilotables par `SANO1_WORKER_*`
(`DEVICE`, `THREADS`, `RESERVE_CORES`, `GPU_LEAVES`, `NICE`) — cf. CLUSTER.md §3.

## Utilisation

1. (Optionnel) renseigne `SANO1_WORKER_NAME` ; `SANO1_SERVER` est déjà par défaut.
2. **Save**, puis loue une offre GPU avec cette template (un GPU récent + bonne
   bande passante = plus de parties/s).
3. L'instance démarre, clone, installe, et commence à jouer en ~1–2 min.
4. Vérifie sur le **dashboard du coordinateur** (`$SANO1_SERVER/`) : ton pseudo
   apparaît au leaderboard et le compteur de parties grimpe.

## Suivi & dépannage

```bash
# en SSH sur l'instance Vast :
tail -f /workspace/sano1_worker.log     # logs du worker
nvidia-smi                              # le GPU doit être utilisé
```

- **« tout tourne sur le CPU » / GPU inactif** → torch ne voit pas le GPU
  (`torch.cuda.is_available()` = False) et `--device auto` retombe en silence sur
  CPU. L'on-start fait désormais un **préflight CUDA** : il affiche `nvidia-smi` +
  la version torch/CUDA et **refuse de lancer** sur CPU (gaspillage sur GPU
  payant). Corrige l'**image Docker** : prends-en une avec torch+CUDA, p.ex.
  `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`, et vérifie que le driver de
  l'hôte est compatible avec la version CUDA de l'image. Pour forcer le CPU malgré
  tout : `SANO1_ALLOW_CPU=1`.
- **Le worker ne démarre pas** → `SANO1_SERVER` manquant/mal orthographié, ou
  préflight CUDA en échec (voir ci-dessus) ; l'on-start sort en erreur, visible
  dans les logs Vast de l'instance et dans `/workspace/sano1_worker.log`.
- **0 partie créditée** → vérifie que l'URL du coordinateur est joignable
  (`curl $SANO1_SERVER/cluster/stats`).
- **Redémarrage d'instance** → l'on-start fait un `git reset --hard` sur le
  volume existant, donc le code est remis à jour automatiquement.

---

# Template TRAINER (federated averaging)

La template ci-dessus loue un GPU comme **worker** (self-play, accès ouvert). Cette
seconde template loue un GPU comme **trainer** : il **entraîne** le modèle et **publie**
les poids. Lance-en **plusieurs** avec le **même `TRAINER_TOKEN`** pour entraîner le
même modèle en parallèle (data-parallel asynchrone façon FedAvg) — leurs poids sont
moyennés à chaque round, lignée unique préservée. Voir [CLUSTER.md](CLUSTER.md) §2bis.

> Crée une template **distincte** de celle du worker (image identique, mais on-start et
> variables différents). Un trainer **exige un token** ; un worker non.

### Image Docker

Identique au worker : `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` (ou toute image
PyTorch+CUDA récente). Vise un GPU costaud (L40S/A100/H100…) : le débit d'entraînement
est l'objectif.

### On-start Script

Colle le contenu de [`scripts/vast_trainer_onstart.sh`](scripts/vast_trainer_onstart.sh).
Il clone le dépôt, installe les dépendances (sans toucher à torch), fait un **préflight
CUDA**, puis lance `sanchess.cluster.trainer --fedavg` en tâche de fond. Le token est lu
depuis l'environnement (jamais passé en argv → invisible dans `ps`).

### Variables d'environnement (onglet Env de la template)

| Variable | Requis | Rôle | Exemple |
|---|:---:|---|---|
| `TRAINER_TOKEN` | ✅ | secret partagé avec le coordinateur (autorise la publication) | `un-secret-long-aleatoire` |
| `SANO1_SERVER` | | URL du coordinateur (défaut: coordinateur officiel) | `https://…up.railway.app` |
| `SANO1_TRAINER_ID` | | identité stable de ce trainer (défaut: `vast-$HOSTNAME`) | `vast-a100-1` |
| `SANO1_FED_LOCAL_STEPS` | | pas de gradient locaux par round (défaut: 400) | `400` |
| `SANO1_FED_BATCH_SIZE` | | surcharge `online.batch_size` (gros GPU) | `512` |
| `SANO1_CONFIG` | | fichier de config (défaut: `config.cloud.yaml`) | `config.cloud.yaml` |
| `SANO1_SEED_SHARDS` | | dossier de shards d'amorçage du buffer | `data/shards` |
| `SANO1_BRANCH` | | branche git | `main` |

- Mets le **même `TRAINER_TOKEN`** sur toutes tes instances trainer **et** sur le
  coordinateur Railway. Donne à chacune un `SANO1_TRAINER_ID` **unique** (le défaut
  `vast-$HOSTNAME` l'est déjà en pratique).
- `--local-steps` règle la granularité FedAvg : plus haut = moins de synchros (moins de
  réseau) mais risque de divergence ; reste modéré (~200–800).
- Le rythme des rounds se règle **côté coordinateur** (`FED_ROUND_TIMEOUT`, `FED_QUORUM`,
  `FED_FINALIZE_GRACE` — cf. CLUSTER.md §2bis).

### Utilisation & suivi

1. Renseigne `TRAINER_TOKEN` (+ `SANO1_TRAINER_ID` si tu veux un nom lisible), **Save**.
2. Loue 1..N offres GPU avec cette template (chacune devient un trainer).
3. Sur le dashboard du coordinateur, `model_version` doit s'incrémenter et `model_round`
   avancer. En SSH : `tail -f /workspace/sano1_trainer.log` (les logs montrent les
   contributions, qui finalise chaque round, et les `409 round déjà finalisé` normaux
   côté non-finalizer).

> ⚠️ Vast.ai est facturé à l'heure tant que l'instance tourne. Pense à
> **détruire** l'instance quand tu arrêtes de contribuer.
