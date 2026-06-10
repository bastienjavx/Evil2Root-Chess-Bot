"""Convertit un dump PGN (.pgn ou .pgn.zst) Lichess en shards de samples.

Filtre qualité (config `data`) : Elo moyen minimum, exclusion du bullet, parties
avec résultat décisif/nul valides. Chaque position jouée devient un sample
`(fen, coup joué, valeur=résultat côté trait)`.

Usage :
  python -m sanchess.data.pgn_to_samples data/lichess_raw/xxx.pgn.zst \\
      --out data/shards --max-games 100000
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import chess
import chess.pgn

from ..utils import load_config
from .samples import write_samples

RESULT_VALUE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def _open_pgn(path):
    """Ouvre un .pgn / .pgn.zst local OU une URL http(s) (décompression à la volée).

    Le streaming HTTP évite de télécharger le dump entier (~30 Go) : on lit juste
    assez de parties (--max-games) puis on coupe la connexion.
    """
    src = str(path)
    if src.startswith("http://") or src.startswith("https://"):
        import requests
        import zstandard as zstd
        resp = requests.get(src, stream=True, timeout=60)
        resp.raise_for_status()
        raw = resp.raw
        raw.decode_content = True
        if src.endswith(".zst"):
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            return io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
        return io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")

    path = Path(src)
    if path.suffix == ".zst":
        import zstandard as zstd
        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def _keep_game(headers, cfg_data) -> bool:
    if headers.get("Result") not in RESULT_VALUE:
        return False
    if cfg_data.get("exclude_bullet", True):
        if "Bullet" in headers.get("Event", ""):
            return False
    try:
        we = int(headers.get("WhiteElo", "0"))
        be = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    if (we + be) / 2 < cfg_data.get("min_elo", 2000):
        return False
    return True


def convert(pgn_src, out_dir: Path, cfg: dict,
            max_games: int | None, games_per_shard: int = 5000) -> None:
    cfg_data = cfg["data"]
    out_dir.mkdir(parents=True, exist_ok=True)
    base = os.path.basename(str(pgn_src).rstrip("/"))
    stem = base.replace(".pgn.zst", "").replace(".pgn", "") or "dump"

    handle = _open_pgn(pgn_src)
    games_kept = games_done = shard_idx = 0
    buffer: list[tuple[str, str, int]] = []

    def flush():
        nonlocal shard_idx, buffer
        if not buffer:
            return
        shard = out_dir / f"{stem}_{shard_idx:04d}.txt.gz"
        write_samples(shard, buffer)
        print(f"  -> {shard.name} ({len(buffer)} samples)")
        shard_idx += 1
        buffer = []

    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        games_done += 1
        if not _keep_game(game.headers, cfg_data):
            continue

        result_white = RESULT_VALUE[game.headers["Result"]]
        board = game.board()
        for move in game.mainline_moves():
            # valeur côté trait : +result_white si Blancs au trait, sinon l'opposé.
            value = result_white if board.turn == chess.WHITE else -result_white
            buffer.append((board.fen(), move.uci(), value))
            board.push(move)

        games_kept += 1
        if games_kept % games_per_shard == 0:
            flush()
        if max_games and games_kept >= max_games:
            break

    flush()
    print(f"Terminé : {games_kept} parties gardées / {games_done} lues.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", help="chemin du .pgn ou .pgn.zst")
    ap.add_argument("--out", default=None, help="dossier de sortie des shards")
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(args.out or cfg["data"]["shards_dir"])
    convert(args.pgn, out_dir, cfg, args.max_games)


if __name__ == "__main__":
    main()
