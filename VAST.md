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
| `SANO1_WORKER_GPU_GAMES` | | Budget parties GPU batchées | `512` |
| `SANO1_WORKER_WORKERS` | | Process self-play parallèles | `auto` |
| `SANO1_BRANCH` | | Branche git à utiliser | `main` |

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

> ⚠️ Vast.ai est facturé à l'heure tant que l'instance tourne. Pense à
> **détruire** l'instance quand tu arrêtes de contribuer.
