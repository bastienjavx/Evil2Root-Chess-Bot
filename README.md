<div align="center">

# San-o1 ♟️🧠

**A neural chess engine that learns in real time from Lichess.**

AlphaZero/Lc0-style architecture — ResNet policy + value network, PUCT Monte-Carlo Tree Search,
supervised pre-training on Lichess games, and **continuous online learning** with live hot-reload.
Ships as a **UCI engine** and a **native Lichess bot**.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white)
![Lichess](https://img.shields.io/badge/Lichess-BOT_API-000000?logo=lichess&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How it works](#how-it-works)
- [Hardware](#hardware)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Training pipeline](#training-pipeline)
- [Evaluating strength](#evaluating-strength)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Overview

San-o1 is a from-scratch chess engine built around a deep residual network and neural
Monte-Carlo Tree Search, in the spirit of **AlphaZero** and **Leela Chess Zero**.

It is **bootstrapped by supervised learning** on millions of Lichess games (fast path to a
strong baseline), then **keeps improving online** by streaming live high-rated games and
fine-tuning continuously. The playing engine **hot-reloads the latest weights between games**,
so the bot literally gets stronger while it runs.

Two ways to play it:

- **UCI engine** — plug it into any chess GUI (Nibbler, Cutechess, Arena).
- **Native Lichess bot** — connects to the Lichess Bot API and plays online directly.

## Features

- 🧠 **ResNet policy + value network** with **Squeeze-Excitation** blocks (configurable, default **20×256 + SE**).
- 🌲 **PUCT MCTS** with batched GPU leaf evaluation, virtual loss, and Dirichlet root noise (self-play).
- 📚 **Supervised pre-training** on Lichess monthly dumps — streamed and decompressed on the fly (no 30 GB download).
- 🔄 **Continuous online learning** from live Lichess games, with a sliding replay buffer to limit catastrophic forgetting.
- 🌐 **Distributed training across machines** — data-parallel over **Mac M1 (MPS) + Linux (CUDA/CPU)** via gloo all-reduce.
- 🎯 **Modern training recipe** — LR warmup + cosine schedule, gradient clipping, label smoothing, optional weight EMA.
- ♻️ **Hot-reload** — the engine picks up newly trained weights at the start of each game.
- 🔌 **UCI protocol** + 🤖 **native Lichess bot** (challenge handling, clock-aware time management).
- 🍎 **Multi-accelerator** — automatic device selection (CUDA → Apple Silicon MPS → CPU).
- 🧪 Unit-tested move ↔ index encoding and network forward pass.
- ⚡ Mixed-precision training (AMP) on CUDA — the default network uses ~1.3 GB VRAM at batch 512.

## How it works

```
                    ┌──────────────────────────────────────────────┐
   Lichess DB       │                  TRAINING                     │
  (monthly dumps) ──┼─► pgn_to_samples ─► shards ─┐                 │
                    │                              ├─► pretrain.py ──┼─► checkpoints/latest.pt
  Lichess API       │                              │                 │          │
  (live games)   ───┼─► stream.py ─► replay buffer ┴─► online.py ────┘          │ hot-reload
                    └──────────────────────────────────────────────┘          ▼
                                                                    ┌──────────────────────┐
                    Board ──► encoding (19×8×8 planes) ──► ResNet ──►│  policy(4672)+value  │
                                                                    └──────────┬───────────┘
                                                                               ▼
                                                                       PUCT MCTS search
                                                                               ▼
                                                      ┌────────────────────────┴───────────┐
                                                      │  UCI engine        Lichess bot      │
                                                      │  (Nibbler/...)     (@Evil2Root)      │
                                                      └────────────────────────────────────┘
```

**Board encoding** — 19 planes of 8×8 (12 piece planes, side-to-move, castling rights,
fifty-move counter, en-passant), canonicalised to the side-to-move perspective.

**Move encoding** — the AlphaZero `64 × 73 = 4672` policy head (56 "queen" moves, 8 knight
moves, 9 underpromotions per square); illegal moves are masked using `python-chess`.

**Search** — PUCT selection `Q + c_puct · P · √N / (1 + n)`, network priors for expansion,
value backup with alternating sign, batched leaf evaluation for GPU throughput.

## Hardware

Developed and tested on an **NVIDIA RTX 2070 SUPER (8 GB)**. The default `20×256` network
trains at ~5 steps/s (batch 512, AMP) using ~1.3 GB of VRAM, leaving plenty of headroom to
scale the network up. CPU-only inference works but is slow.

## Installation

```bash
git clone https://github.com/bastienjavx/Evil2Root-Chess-Bot.git
cd Evil2Root-Chess-Bot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# PyTorch ships with CUDA by default — verify:
python -c "import torch; print(torch.cuda.is_available())"
```

For the Lichess bot, set a token (scope `bot:play`) — put it in a local `.env` (git-ignored):

```bash
echo "LICHESS_TOKEN=lip_xxxxxxxxxxxxxxxx" > .env
```

## Quickstart

### 1 — Collect data & pre-train

```bash
# Stream a Lichess dump (Elo-filtered) and train, in one command:
MONTH=2019-01 MAXG=50000 ./scripts/collect_and_train.sh
```

This streams games, builds shards under `data/shards/`, then trains, writing
`checkpoints/latest.pt` every few minutes.

### 2 — Play it in a GUI (UCI)

```bash
./scripts/run_uci.sh        # python -m sanchess.uci
```

Add it as a UCI engine in **Nibbler** (shows the MCTS tree, like Leela), **Cutechess**, or **Arena**.

### 3 — Run the Lichess bot

```bash
python -m sanchess.lichess_bot --check     # show account status (read-only)
./scripts/run_bot.sh                        # go online and accept challenges
```

> ⚠️ Converting an account to a BOT (`--upgrade`) is **irreversible** and only works on an
> account that has never played a game.

### 4 — Continuous learning

```bash
./scripts/run_online.sh     # live stream + background fine-tuning (after pre-training converges)
```

The collector feeds a replay buffer from live games while the trainer fine-tunes and refreshes
`checkpoints/latest.pt`; the engine hot-reloads it between games.

### 4b — Opening book (stronger openings, no retraining)

A standard **Polyglot** opening book plays known theory for the first moves and only hands
over to the MCTS once out of book — an instant strength boost for both the UCI engine and the
Lichess bot, without touching the network's weights.

```bash
# Build a book from a Lichess PGN dump (local file or streamed URL):
./scripts/build_book.sh data/lichess_raw/dump.pgn.zst --max-games 100000 --max-ply 24
# -> writes checkpoints/book.bin
```

Then enable it in `config.yaml`:

```yaml
book:
  enabled: true                # off by default
  path: checkpoints/book.bin   # any standard Polyglot .bin works (gm2600, Perfect20xx…)
  max_ply: 16                  # use the book for the first 16 half-moves
  selection: weighted          # "weighted" (varied) or "best" (always the heaviest move)
```

Both `python -m sanchess.uci` and the Lichess bot consult the book automatically; book moves
are **not** used as learning targets, so continuous training stays clean.

### 5 — Web console (API + live play + stats)

A self-contained web UI (`sanchess.web`) to **play the model live**, watch it play **itself**,
browse the exposed **checkpoints**, and monitor **training / system** stats in real time.

```bash
pip install fastapi "uvicorn[standard]"   # one-time (already in requirements.txt)
./scripts/run_web.sh                        # -> http://localhost:8000
CLOUDFLARED=1 ./scripts/run_web.sh          # + public tunnel (*.trycloudflare.com)
```

Tabs: **Jouer** (human vs model, choose checkpoint + node/time budget, see the model's
candidate moves & eval), **Regarder** (model vs model, streamed move-by-move over WebSocket),
**Modèles** (every `checkpoints/*.pt` with step/arch/size), **Entraînement** (policy/value/loss
curves parsed from `data/*.log`), **Système** (GPU, systemd services, replay buffer, processes).

The engine **hot-reloads `latest.pt`**, so the live UI tracks ongoing training. All chess logic
(legality, SAN, game-over) runs server-side — the frontend has **zero external dependencies**.

API surface (full schema at `/docs`):

| Method | Route | Purpose |
|--------|-------|---------|
| `GET`  | `/api/models`   | checkpoints + metadata, active device |
| `GET`  | `/api/legal`    | legal moves of a position (FEN/UCI moves) |
| `POST` | `/api/analyze`  | MCTS analysis: bestmove, value, candidates |
| `POST` | `/api/move`     | model plays one move on a position |
| `GET`  | `/api/training` | policy/value/loss series from logs |
| `GET`  | `/api/system`   | GPU / services / buffer / processes |
| `WS`   | `/ws/play`      | human-vs-model game, one move per message |
| `WS`   | `/ws/selfplay`  | model-vs-model game, streamed |

Run it as a service (auto-start on boot): copy `scripts/systemd/sano1-web.service` to
`/etc/systemd/system/` then `sudo systemctl enable --now sano1-web`.

## Project structure

```
sanchess/
├── encoding.py            8×8 planes + move ↔ index (4672) mapping
├── model.py               ResNet: policy (4672) & value (tanh) heads
├── uci.py                 UCI engine + weight hot-reload
├── book.py                Polyglot opening book (theory before MCTS)
├── lichess_bot.py         native Lichess Bot API client
├── utils.py               config / checkpoints / .env loader
├── search/
│   └── mcts.py            batched PUCT Monte-Carlo Tree Search
├── data/
│   ├── download.py        Lichess monthly dumps (.pgn.zst)
│   ├── pgn_to_samples.py  PGN → shards (streamed, Elo-filtered)
│   ├── build_book.py      PGN → Polyglot .bin opening book
│   ├── stream.py          live Lichess games → replay buffer
│   └── samples.py         shared sample format (gzip text)
├── train/
│   ├── dataset.py         PyTorch dataset (on-the-fly encoding)
│   ├── pretrain.py        supervised pre-training (AMP, LR schedule, EMA)
│   ├── distributed.py     multi-machine data-parallel trainer (Mac M1 + Linux)
│   ├── selfplay.py        CPU self-play → replay buffer
│   └── online.py          continuous learning + hot-reload checkpoints
└── web/
    ├── server.py          FastAPI app (REST + WebSocket)
    ├── engine.py          model manager: load/hot-reload, MCTS analysis
    ├── stats.py           log parsing + GPU/services/data status
    └── static/            zero-dependency frontend (board, charts, live play)
scripts/                   one-command launchers
tests/                     encoding round-trip tests
config.yaml                all hyperparameters
```

## Configuration

All knobs live in [`config.yaml`](config.yaml):

| Section | Key | Purpose |
|---|---|---|
| `model` | `channels`, `blocks` | network size (default 256×20; scale up for strength) |
| `mcts` | `c_puct`, `default_nodes`, `eval_batch_size` | search strength / speed |
| `train` | `batch_size`, `lr`, `amp` | pre-training |
| `online` | `lr`, `buffer_capacity`, `min_buffer` | continuous-learning aggressiveness |
| `data` | `min_elo`, `exclude_bullet` | training-data quality filter |
| `bot` | `accept_variants`, `max_think_seconds` | Lichess bot behaviour |

## Training pipeline

1. **Supervised pre-training** — cross-entropy on the played move + MSE on the game result
   (from the side-to-move perspective). Reaches a solid baseline quickly.
2. **Online fine-tuning** — small learning rate over a sliding window of recent + seeded
   games, to keep improving without forgetting.
3. **Checkpoints** are written atomically (`tmp` + rename) so the engine can hot-reload safely.

To avoid two trainers fighting over `latest.pt`, run online learning **after** pre-training
finishes (see `scripts/online_after_pretrain.sh`).

## Distributed training (Mac M1 + Linux)

Train the same network across several machines at once — for example a **Linux box
with an NVIDIA GPU** and a **MacBook on Apple Silicon** — pooling their compute on
one shared model. Each node computes on its **local** accelerator (CUDA, MPS or CPU)
and gradients are averaged every step over the cross-platform **gloo** backend (CPU
transport), so heterogeneous CUDA + MPS clusters just work — no NCCL required.

```
        ┌── Linux (rank 0, CUDA) ──┐        averaged gradients (gloo / TCP)
        │   forward / backward     │◄──────────────────────────────────────┐
        └──────────┬───────────────┘                                        │
                   │ writes checkpoints/latest.pt (hot-reload)              │
        ┌──────────┴───────────────┐                                        │
        │   Mac M1 (rank 1, MPS)   │────────────────────────────────────────┘
        │   forward / backward     │
        └──────────────────────────┘
```

Run **on each machine** (same `WORLD_SIZE`, same master address/port). Rank 0 is the
master and is the only node that writes `checkpoints/latest.pt`:

```bash
# On the Linux master (its LAN IP is e.g. 192.168.1.10) — rank 0:
MASTER_ADDR=192.168.1.10 WORLD_SIZE=2 RANK=0 ./scripts/run_distributed.sh

# On the Mac M1 — rank 1:
MASTER_ADDR=192.168.1.10 WORLD_SIZE=2 RANK=1 ./scripts/run_distributed.sh
```

Each node needs its own copy of the training shards (`data/shards/`); a
`DistributedSampler` gives every rank a disjoint slice so no sample is seen twice
per step. Tune the backend, ports and timeout under the `distributed:` section of
[`config.yaml`](config.yaml). On a single machine with multiple NVIDIA GPUs, set
`BACKEND=nccl` for faster GPU-to-GPU communication.

> The master port must be reachable from the other nodes (open it on the LAN /
> firewall). Heterogeneous clusters should keep the default `gloo` backend.

## Evaluating strength

Run a match in **Cutechess** against Stockfish with `UCI_LimitStrength` / a fixed Elo to
estimate rating and track progress across checkpoints.

## Roadmap

- [x] Squeeze-Excitation residual blocks
- [x] Distributed multi-device training (Mac M1 + Linux)
- [x] Apple Silicon (MPS) support
- [ ] Larger network (e.g. 20×384) and more training data
- [ ] WDL (win/draw/loss) value head
- [ ] Self-play reinforcement learning on top of the supervised base
- [ ] Transposition table / tree reuse between moves
- [ ] ONNX / TensorRT export for faster MCTS inference

## Disclaimer

Reaching "super-GM / AlphaZero" strength from scratch requires thousands of TPUs. On a single
consumer GPU the realistic target is **strong club → expert** level, reached quickly via
supervised pre-training and improving continuously with online learning. The architecture
(ResNet + MCTS) scales cleanly if you add more compute and data.

This project uses the Lichess API in accordance with its
[Terms of Service](https://lichess.org/terms-of-service) (request throttling, 429 backoff).

## License

Released under the [MIT License](LICENSE).
