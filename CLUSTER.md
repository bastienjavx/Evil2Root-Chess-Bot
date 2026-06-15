# San-o1 — Cluster de calcul distribué bénévole

Mettre en commun la puissance de calcul de plusieurs machines (CPU, GPU NVIDIA, Mac M1)
pour entraîner le réseau San-o1 **en continu**, sur le modèle de **Leela Chess Zero** :
des bénévoles génèrent des parties de self-play, une machine GPU centralise
l'entraînement, et le modèle amélioré redescend vers tout le monde.

```
        Railway (coordinateur, CPU)          1+ machine(s) GPU (trainer·s)
   ┌──────────────────────────────┐      ┌──────────────────────────────────┐
   │ distribue les jobs           │ ◄─── │ contribue/publie les poids        │
   │ collecte/valide les parties  │ ───► │ entraîne (online.py ou FedAvg)    │
   │ sert le modèle + dashboard   │      │ §2 : 1 trainer · §2bis : N (FedAvg)│
   │ arbitre les rounds FedAvg    │      └──────────────────────────────────┘
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

## 2 bis. Plusieurs trainers (federated averaging)

Pour augmenter le **débit d'entraînement**, tu peux faire tourner **plusieurs machines
GPU de train principal**, réparties sur Internet, partageant le **même `TRAINER_TOKEN`**.
Elles entraînent toutes le *même* modèle en parallèle (data-parallel asynchrone, façon
FedAvg) et leurs poids sont **moyennés** périodiquement : une **lignée unique** est
préservée, sans qu'aucune machine n'écrase le travail d'une autre.

> Le mode mono-trainer du §2 reste inchangé. Et comme `train/distributed.py` (all-reduce
> gloo) suppose un **LAN fiable**, c'est ce mode FedAvg qui convient à des machines
> distantes.

Sur **chaque** machine de train (même token) :

```bash
TRAINER_TOKEN=le-meme-secret-que-railway \
  python -m sanchess.cluster.trainer --server https://<app>.up.railway.app \
    --fedavg --trainer-id <nom-unique> --local-steps 400
```

Comment ça marche (un **round** = base `v` → version `v+1`) :

1. chaque trainer télécharge le modèle global `v` (écrit aussi dans `latest.pt` local) ;
2. il entraîne `--local-steps` pas sur les parties synchronisées ;
3. il envoie ses poids (`/cluster/trainer/contribute`, blob opaque + `num_samples`) ;
4. à la fermeture du round (quorum ou deadline), **un seul** trainer (le *finalizer* =
   premier contributeur) télécharge toutes les contributions, les **moyenne** (pondérées
   par `num_samples`) et publie → `v+1`, round suivant ; les autres re-téléchargent.

Robustesse : la publication est **idempotente par round** (un seul publish accepté), donc
même si deux trainers finalisent en même temps, un seul gagne. Si le finalizer meurt, un
autre reprend après le délai de grâce. **Le coordinateur reste sans torch** : il ne fait
qu'arbitrer le cycle des rounds et router les blobs — la moyenne se fait côté trainers.

Options utiles : `--local-steps N` (pas par round, défaut 400 ; rester modéré pour éviter
la divergence FedAvg), `--trainer-id NOM` (identité stable, défaut = nom d'hôte),
`--fed-batch-size N`, `--seed-shards data/shards` (amorçage). Mêmes réglages en env :
`SANO1_FEDAVG=1`, `SANO1_FED_LOCAL_STEPS`, `SANO1_TRAINER_ID`.

Variables d'environnement **côté coordinateur** (Railway) pour régler le rythme :

| Variable | Rôle | Défaut |
|---|---|---|
| `FED_ROUND_TIMEOUT` | s avant de fermer un round même incomplet | `180` |
| `FED_QUORUM` | contributions min pour fermer tôt (1 = mono-trainer OK) | `1` |
| `FED_FINALIZE_GRACE` | s après deadline avant qu'un autre trainer reprenne la finalisation | `60` |

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
| `--workers auto` | utilise presque tous les cœurs (`cpu_count - reserve_cores`) ; défaut agressif |
| `--workers 4` | force N processus de self-play en parallèle |
| `--threads auto` | threads torch par worker (`auto` = 1, recommandé pour éviter la sur-souscription) |
| `--reserve-cores 2` | garde N cœurs libres quand `--workers auto` |
| `--gpu-games 512` | budget total de parties GPU batchées sur cette machine, réparti entre les workers |
| `--gpu-leaves 8` | feuilles GPU par partie (surcharge locale de `JOB_GPU_LEAVES`) |
| `--nice 10` | priorité basse : ne fige pas le poste |
| `--once` | un seul lot puis arrêt (test) |

Les mêmes réglages existent en variables d'environnement locales :
`SANO1_WORKER_WORKERS`, `SANO1_WORKER_THREADS`, `SANO1_WORKER_RESERVE_CORES`,
`SANO1_WORKER_GPU_GAMES`, `SANO1_WORKER_GPU_LEAVES`, `SANO1_WORKER_DEVICE`,
`SANO1_WORKER_NICE`.

- **GPU CUDA** : le worker peut lancer plusieurs process pour saturer le MCTS Python ;
  le budget `gpu_games` est réparti entre eux pour éviter de multiplier la VRAM par accident.
- **Mac M1 / CPU** : `play_game` séquentiel par process ; le mode `auto` utilise les cœurs
  disponibles avec un cœur réservé par défaut.

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
