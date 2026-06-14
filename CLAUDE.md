# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

San-o1 is a from-scratch neural chess engine (AlphaZero/Lc0-style: ResNet policy+value net +
PUCT MCTS) that pre-trains on Lichess dumps, learns online from live games, and ships as a UCI
engine and a native Lichess bot (`@Evil2Root`). Code is Python; user-facing docs/comments are
mostly French.

## Environment & commands

- **Always use the venv interpreter `.venv/bin/python`.** `torch` / `python-chess` exist only in
  the venv, not in the system `python3`. The `scripts/*.sh` launchers handle this themselves
  (`PY=.venv/bin/python` with a `python3` fallback) — invoke modules through them, or activate the
  venv first (`source .venv/bin/activate`).
- Run all tests: `.venv/bin/python -m pytest tests/` — Run one: `.venv/bin/python -m pytest tests/test_model.py::test_name`. There is no Makefile, linter, or pytest config; tests are plain `pytest`.
- Common entry points (all are `python -m sanchess.<module>`, wrapped by a script):
  - `scripts/run_uci.sh` → `sanchess.uci` (UCI engine for Nibbler/Cutechess/Arena)
  - `scripts/run_bot.sh` → `sanchess.lichess_bot` (`--check` = read-only account status; `--upgrade` = irreversible BOT conversion)
  - `scripts/run_pretrain.sh` → `sanchess.train.pretrain` (supervised pre-training)
  - `scripts/run_online.sh` → `sanchess.train.online` (continuous learning)
  - `scripts/run_web.sh` → `sanchess.web` (FastAPI console at :8000; `CLOUDFLARED=1` adds a public tunnel)
  - `scripts/run_distributed.sh`, `run_selfplay*.sh`, `download_all.sh`, `bench_eval.py`, `monitor_train.py`
- `scripts/run_pretrain.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (anti-fragmentation so training and the bot can share the 8 GB GPU). Preserve this in any new training launcher.

## Config & secrets

- `config.yaml` = local **blitz** preset (v3 20×256). `config.cloud.yaml` = **classical** preset (v3 24×320, bigger batch/LR) for cloud pre-training. Most modules take `--config`.
- `sanchess/utils.py` is the central plumbing: `load_config`, `resolve_device` (CUDA → MPS → CPU), `save_checkpoint`/`load_checkpoint`, `load_model_state`, `load_dotenv`.
- `.env` (git-ignored) holds `LICHESS_TOKEN`, loaded via `utils.load_dotenv`.

## Architecture — the cross-cutting contracts

These are the invariants that span many files; understanding them is the fast path to being productive.

1. **`checkpoints/latest.pt` is the single shared artifact.** Trainers write it atomically (`tmp` + rename); all inference paths (`uci`, `lichess_bot`, `web/engine`, the self-play generators) **hot-reload it by watching its mtime** and rebuild the engine in place — no restart needed to pick up new weights. The checkpoint **embeds its `model_cfg`**, so `model.build_model_from_checkpoint` reconstructs the exact architecture (even one trained in the cloud with a different config) without touching the local `config.yaml`.

2. **Only one process may write `latest.pt` at a time.** `run_pretrain.sh` and `online_after_pretrain.sh` share a `flock /tmp/sano1-trainer.lock` (pretrain XOR online). Distributed training (rank 0) and the cluster trainer also write it. Before starting any new writer, stop the others, or two trainers will corrupt the file.

3. **`sanchess/data/samples.py` defines the one shared on-disk sample format** (gzip text shards). Every producer (`pgn_to_samples`, `stream`, `selfplay`, `selfplay_gpu`, `lichess_bot._record_game`) and every consumer (`train/dataset.py`, `online.py`) goes through it. Columns: `fen value [policy uci:visits…] [plies_to_end]`, the last two **optional and backward-compatible** (old shards just train fewer heads). Don't change the column layout without updating both ends and keeping old shards loadable.

4. **The net (`model.py`) has families `v1`/`v2`/`v3`** (default `v3`); old checkpoints still load via their embedded `model_cfg`. **`v3` forward returns a 3-tuple `(policy, value, moves_left|None)`** — all callers unpack three. Heads: conv policy (4672), WDL value (3 logits), auxiliary moves-left (training target only; **not** wired into MCTS). `mcts.Evaluator` keeps a 2-tuple contract (uses `out[0]`, `out[1]`).

5. **Strength on the RTX 2070S is throughput-bound, not VRAM-bound.** Elo ≈ MCTS nodes/move ≈ evals/s. Pick network size with `scripts/bench_eval.py`. Inference accelerators (all opt-in, all fall back to plain PyTorch): tree reuse between moves (`mcts.tree_reuse`), `model.compile_inference` (`torch.compile(dynamic=True)`), and TensorRT fp16 export (`sanchess.export`, which drops the moves-left head → stable 2-output graph).

6. **Two different multi-machine systems, don't confuse them:**
   - `train/distributed.py` — synchronous data-parallel SGD over **gloo** (Mac M1 MPS + Linux CUDA on a trusted LAN). Rank 0 writes `latest.pt`.
   - `sanchess/cluster/` — asynchronous **volunteer** compute (Lc0-style): a tiny torch-free coordinator (`cluster/server.py`, deployed on Railway via `Dockerfile.cluster`/`requirements-cluster.txt`) hands self-play jobs to workers; your GPU box runs `cluster/trainer.py` (the only holder of `TRAINER_TOKEN`, the only publisher of the model). See `CLUSTER.md`.

## Runtime services & gotchas

- **systemd services** (`scripts/systemd/`, installed to `/etc/systemd/system/`): `sano1-bot`, `sano1-train`, `sano1-online`, `sano1-stream`, `sano1-download`, `sano1-web`, plus `sano1-model-push.timer` and `sano1-status-badge.timer`. Logs: `journalctl -u <svc> -f`.
- **Hot-reload is weights-only, not code.** After any *code* change to the bot, `sudo systemctl restart sano1-bot` (likewise for other long-running services) for it to take effect.
- **Single-instance locks.** The bot (`/tmp/sanchess_lichess_bot.lock`), web, stream, download, and trainer each take a `flock`. Never run two bot instances with the same Lichess token — they steal each other's events and the bot goes silent. Reset with `pkill -f sanchess.lichess_bot`.
- **`pkill -f sanchess.train.pretrain` also kills the shell matching that string.** Use the bracket trick: `pgrep -af "sanchess.train.pretrai[n]"`. Self-play workers use multiprocessing `spawn` and won't match the parent's cmdline (`pgrep -P <main>` to see them).
- **GPU is 8 GB.** Don't run a `batch≥512` training at the same time as the bot. The configured `train.batch_size: 256` + `expandable_segments` keeps pretrain (~1.85 GB) and bot (~0.7 GB) co-resident.
- **`data/shards` is a symlink** to an external 8 TB ext4 disk (`/media/evil2root/8TB/...`). Pre-training loads all shards into RAM at startup.
