# San-o1 — IA d'échecs neuronale (AlphaZero/Lc0-like)

Moteur d'échecs à base de **réseau de neurones** (ResNet politique + valeur) avec
**recherche MCTS PUCT**, pré-entraîné en **supervisé sur les parties Lichess**, et
capable d'**apprendre en continu** à partir des parties Lichess en temps réel —
les nouveaux poids sont **rechargés à chaud** par le moteur pendant que tu joues.

Le moteur parle le protocole **UCI** : tu peux le brancher dans n'importe quelle
interface d'échecs (Nibbler, Cutechess, Arena) ou en faire un bot Lichess.

> ⚠️ **Réalisme** : atteindre un niveau « super-GM » type AlphaZero *from scratch*
> demande des milliers de TPU. Sur un seul GPU grand public (ici une RTX 2070
> SUPER), l'objectif réaliste est un niveau **club fort → expert**, atteint
> rapidement par le pré-entraînement supervisé, qui continue de progresser avec
> le fine-tuning en ligne. L'architecture (ResNet + MCTS) *scale* si tu ajoutes
> du calcul plus tard (plus de blocs/canaux, plus de données, self-play).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (PyTorch s'installe avec CUDA par défaut ; vérifie : python -c "import torch;print(torch.cuda.is_available())")
```

## Démarrage rapide (pipeline complet)

```bash
# 1. Télécharger un dump mensuel Lichess (plusieurs Go)
python -m sanchess.data.download 2024-01

# 2. Convertir en samples (positions, coups, résultats), filtré par Elo
python -m sanchess.data.pgn_to_samples data/lichess_raw/lichess_db_standard_rated_2024-01.pgn.zst \
    --out data/shards --max-games 200000

# 3. Pré-entraîner le réseau (écrit checkpoints/latest.pt)
python -m sanchess.train.pretrain --shards data/shards

# 4. Jouer : brancher le moteur UCI dans une GUI
python -m sanchess.uci          # ou ./scripts/run_uci.sh

# 5. (optionnel) Apprentissage continu : stream Lichess + entraînement de fond
export LICHESS_TOKEN=ton_token  # optionnel, augmente les limites API
./scripts/run_online.sh
```

Pendant que `run_online.sh` tourne, il met à jour `checkpoints/latest.pt` ; le
moteur UCI recharge ces poids automatiquement à chaque nouvelle partie
(`ucinewgame`) → l'IA s'améliore en continu.

## Utiliser le moteur dans une interface de jeu

Le binaire UCI est `python -m sanchess.uci`. Dans la GUI, ajoute un « nouveau
moteur UCI » pointant vers `scripts/run_uci.sh`.

- **Nibbler** (recommandé) : affiche l'arbre MCTS comme Leela — idéal pour ce moteur.
- **Cutechess** : pour organiser des matchs (ex. estimer l'Elo contre Stockfish bridé).
- **Arena** : GUI Windows classique.
- **lichess-bot** : pour le faire jouer en ligne sur Lichess (compte BOT requis).

Commandes UCI supportées : `uci`, `isready`, `ucinewgame`, `position`,
`go nodes N` / `go movetime ms` / `go wtime .. btime ..`, `stop`, `quit`.

## Structure

```
sanchess/
  encoding.py      plans 8x8 (style AlphaZero) + mapping coups<->index (4672)
  model.py         ResNet : têtes politique (4672) & valeur (tanh)
  search/mcts.py   MCTS PUCT, évaluation batchée GPU, hot-reload-friendly
  uci.py           moteur UCI + rechargement à chaud des poids
  data/
    download.py        dumps mensuels Lichess (.pgn.zst)
    pgn_to_samples.py  PGN -> shards (fen, coup, valeur), filtre Elo
    stream.py          parties Lichess en temps réel -> replay buffer
    samples.py         format de samples partagé (gzip texte)
  train/
    dataset.py     Dataset PyTorch (encodage à la volée)
    pretrain.py    pré-entraînement supervisé (+ AMP)
    online.py      entraînement continu + checkpoints hot-reload
config.yaml        tous les hyperparamètres (réseau, MCTS, entraînement)
tests/             tests d'encodage (round-trip coup<->index)
```

## Réglages clés (`config.yaml`)

- `model.channels` / `model.blocks` : taille du réseau (8 Go tiennent bien plus
  large que le défaut 128×10 — augmente pour plus de force, au prix de la vitesse).
- `mcts.default_nodes`, `mcts.c_puct`, `mcts.eval_batch_size` : force/vitesse de la recherche.
- `data.min_elo`, `data.exclude_bullet` : qualité des données d'entraînement.
- `online.lr`, `online.buffer_capacity` : agressivité de l'apprentissage continu
  (LR faible + gros buffer = moins d'oubli catastrophique).

## Tests

```bash
python -m tests.test_encoding     # round-trip coup<->index, formes des plans
```

## Évaluer la force

Organise un match dans Cutechess contre Stockfish à niveau/Elo limité
(`UCI_LimitStrength`) pour estimer l'Elo et suivre la progression au fil de
l'entraînement.

## Pistes pour aller plus loin
- Réseau plus gros (20×256) + plus de données → plus fort.
- Tête **WDL** (3 classes) au lieu d'une valeur scalaire.
- **Self-play** + apprentissage par renforcement (AlphaZero) une fois la base supervisée solide.
- Export **ONNX/TensorRT** pour accélérer l'inférence MCTS.
- Cache de transposition dans le MCTS.

## Respect de Lichess
`stream.py` respecte la ToS Lichess : throttling entre requêtes et gestion des
erreurs 429. Un token API (`LICHESS_TOKEN`) est optionnel mais recommandé.
