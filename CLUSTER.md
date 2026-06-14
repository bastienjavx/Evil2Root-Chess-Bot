# San-o1 — Cluster de calcul distribué bénévole

Mettre en commun la puissance de calcul de plusieurs machines (CPU, GPU NVIDIA, Mac M1)
pour entraîner le réseau San-o1 **en continu**, sur le modèle de **Leela Chess Zero** :
des bénévoles génèrent des parties de self-play, une machine GPU centralise
l'entraînement, et le modèle amélioré redescend vers tout le monde.

```
        Railway (coordinateur, CPU)              Ta machine GPU (trainer)
   ┌──────────────────────────────┐         ┌──────────────────────────────┐
   │ distribue les jobs           │ ◄────── │ publie le modèle (publish)   │
   │ collecte/valide les parties  │ ──────► │ entraîne (online.py, GPU)    │
   │ sert le modèle + dashboard   │         └──────────────────────────────┘
   └──────────────────────────────┘
        ▲ download modèle   ▲ upload parties
   ┌────┴────┐  ┌───────────┴────┐  ┌──────────────┐
   │ CPU     │  │ GPU NVIDIA     │  │ Mac M1 (MPS) │  … N bénévoles (workers)
   └─────────┘  └────────────────┘  └──────────────┘
```

**Pourquoi pas l'all-reduce gloo existant ?** `train/distributed.py` suppose un LAN fiable
et des nœuds qui restent connectés (SGD synchrone). Sur Internet, avec des bénévoles qui
vont et viennent, le modèle asynchrone (self-play réparti + entraînement central) est le
seul robuste — c'est exactement celui de Lc0.

---

## 1. Déployer le coordinateur sur Railway

Le coordinateur est **sans GPU et sans torch** : il route des octets et valide les parties.

1. Pousse le dépôt San-o1 sur GitHub.
2. Sur [railway.app](https://railway.app) : **New Project → Deploy from GitHub repo**.
   Railway détecte `railway.json` et build `Dockerfile.cluster` (image légère).
3. **Ajoute un Volume** au service, monté sur `/data` (Settings → Volumes). Il persiste le
   modèle, le buffer de parties et la base sqlite entre les redéploiements.
4. **Variables d'environnement** (Settings → Variables) :

   | Variable | Rôle | Exemple |
   |---|---|---|
   | `POOL_DIR` | dossier d'état (= point de montage du Volume) | `/data` |
   | `TRAINER_TOKEN` | secret qui autorise **uniquement ton trainer** à publier le modèle | `un-secret-long-aleatoire` |
   | `JOB_NODES` | simulations MCTS/coup demandées aux workers | `160` |
   | `JOB_TARGET_SAMPLES` | positions par envoi de chaque worker | `2000` |
   | `JOB_GPU_GAMES` | parties batchées par worker GPU | `64` |
   | `POOL_MAX_SHARDS` | taille max du buffer de parties (rotation) | `4000` |

5. Railway expose une URL publique : `https://<app>.up.railway.app`. Ouvre-la → **dashboard**
   (stats live + leaderboard + commande pour rejoindre). Healthcheck : `/cluster/stats`.

> ⚠️ Railway ne fournit pas de GPU : l'**entraînement** (pas de gradient) tourne sur **ta
> machine**, pas ici. Le coordinateur ne fait que la mise en commun.

---

## 2. Lancer le trainer (sur ta RTX 2070S)

Le trainer tire les parties, lance `train/online.py` (inchangé) pour entraîner, puis
republie le modèle amélioré. C'est la seule machine qui détient `TRAINER_TOKEN`.

```bash
cd San-o1
source .venv/bin/activate            # ton environnement habituel (torch + CUDA)

TRAINER_TOKEN=le-meme-secret-que-railway \
  python -m sanchess.cluster.trainer --server https://<app>.up.railway.app
```

- Au démarrage il publie ton `checkpoints/latest.pt` actuel (les workers ont un modèle tout de suite).
- Ensuite : synchronise les parties → `data/replay_buffer/` → `online.py` entraîne →
  `latest.pt` se met à jour → republié automatiquement (version incrémentée).
- `--no-train` : ne fait que synchroniser + publier (si tu lances `online.py` toi-même).
- `--seed-shards data/shards` : amorce `online.py` avec des données existantes.

---

## 3. Rejoindre comme worker (n'importe qui)

Aucun token nécessaire (accès ouvert). Le matériel est détecté automatiquement.

```bash
git clone <repo San-o1> && cd San-o1
pip install -r requirements.txt      # inclut torch ; pour CUDA voir le commentaire du fichier

python -m sanchess.cluster.worker \
  --server https://<app>.up.railway.app \
  --name "MonPseudo"
```

Options utiles :

| Option | Effet |
|---|---|
| `--device auto` | `cuda` > `mps` (Mac M1) > `cpu` (défaut) |
| `--workers 4` | (CPU) lance N processus de self-play en parallèle |
| `--threads 1` | threads torch par worker CPU (1 = un cœur/worker) |
| `--nice 10` | priorité basse : ne fige pas le poste |
| `--once` | un seul lot puis arrêt (test) |

- **GPU CUDA** : self-play batché (des dizaines de parties par forward) — laisse `--workers 1`.
- **Mac M1 / CPU** : `play_game` séquentiel ; augmente `--workers` pour utiliser plus de cœurs.

Le worker télécharge le modèle courant (revérifié par sha256), joue jusqu'à
`JOB_TARGET_SAMPLES` positions, renvoie le shard, et recommence. Reconnexion + backoff
automatiques si le réseau tombe.

---

## Sécurité & robustesse

- **Validation des parties** côté serveur : FEN parsable + coups légaux (échantillonnés),
  bornes de taille/lignes, garde-fou anti zip-bomb. Shard invalide → rejeté (422), non crédité.
- **Rate-limit** par IP (`POOL_UPLOADS_PER_MIN`).
- **Publication du modèle protégée** par `TRAINER_TOKEN` : même en accès ouvert, personne
  d'autre que ton trainer ne peut écraser le modèle.
- **Écritures atomiques** (tmp + rename) partout : un lecteur ne voit jamais un fichier à moitié écrit.
- **Qualité des données** : `online.py` garde un LR faible + fenêtre glissante → l'impact
  d'éventuelles parties bruitées reste borné. Évolutions possibles : réputation par worker,
  arène de promotion (`scripts/arena.py` existe déjà).

---

## Vérification rapide en local (sans Railway)

```bash
# Terminal 1 — coordinateur
POOL_DIR=/tmp/pool TRAINER_TOKEN=dev \
  uvicorn sanchess.cluster.server:app --port 8001

# Terminal 2 — trainer (publie latest.pt + entraîne)
python -m sanchess.cluster.trainer --server http://localhost:8001 --token dev

# Terminal 3 — worker
python -m sanchess.cluster.worker --server http://localhost:8001 --device cpu --name moi --once
```

Ouvre http://localhost:8001/ : la version du modèle, le total de parties et le leaderboard
doivent évoluer. Tests automatisés : `pytest tests/test_cluster.py`.
