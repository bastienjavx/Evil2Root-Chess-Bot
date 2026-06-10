"""Collecte temps réel de parties Lichess -> replay buffer.

Stratégie : on récupère périodiquement les dernières parties classées des
meilleurs joueurs (leaderboard Lichess), on les convertit en samples
`(fen, coup, valeur)` et on les ajoute au replay buffer que `online.py` consomme.

Respecte la ToS Lichess : throttling + gestion des 429. Un token API
(var d'env LICHESS_TOKEN) augmente les limites mais reste optionnel.

Usage :  LICHESS_TOKEN=xxx python -m sanchess.data.stream --perf blitz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import chess
import requests

from ..utils import load_config, load_dotenv
from .samples import write_samples

API = "https://lichess.org"


def _headers() -> dict:
    h = {"Accept": "application/x-ndjson"}
    token = os.environ.get("LICHESS_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_top_players(perf: str, count: int, timeout: int) -> list[str]:
    r = requests.get(f"{API}/api/player/top/{count}/{perf}",
                     headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    return [u["username"] for u in r.json().get("users", [])]


def _get_with_backoff(url, params, timeout):
    """GET en flux avec respect des 429 (backoff exponentiel)."""
    delay = 5
    while True:
        r = requests.get(url, params=params, headers=_headers(),
                         stream=True, timeout=timeout)
        if r.status_code == 429:
            print(f"  429 — pause {delay}s"); time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        r.raise_for_status()
        return r


def game_to_samples(game: dict, min_elo: int):
    """Convertit un objet partie (ndjson Lichess) en samples."""
    players = game.get("players", {})
    try:
        we = players["white"]["rating"]
        be = players["black"]["rating"]
    except (KeyError, TypeError):
        return
    if (we + be) / 2 < min_elo:
        return
    moves = game.get("moves")
    if not moves:
        return

    winner = game.get("winner")
    result_white = 1 if winner == "white" else (-1 if winner == "black" else 0)

    board = chess.Board()
    for san in moves.split():
        try:
            move = board.parse_san(san)
        except ValueError:
            return
        value = result_white if board.turn == chess.WHITE else -result_white
        yield board.fen(), move.uci(), value
        board.push(move)


def poll_once(users, cfg, seen: set, since_ms: int) -> int:
    cfg_data = cfg["data"]
    perf = cfg["_perf"]
    buffer_dir = Path(cfg_data["buffer_dir"])
    timeout = cfg["lichess"]["request_timeout"]
    min_elo = cfg_data.get("min_elo", 2000)

    rows = []
    new_games = 0
    for user in users:
        url = f"{API}/api/games/user/{user}"
        params = {"max": 20, "rated": "true", "perfType": perf,
                  "moves": "true", "since": since_ms}
        try:
            r = _get_with_backoff(url, params, timeout)
        except requests.RequestException as e:
            print(f"  erreur {user}: {e}"); continue
        for line in r.iter_lines():
            if not line:
                continue
            game = json.loads(line)
            if game.get("id") in seen:
                continue
            seen.add(game.get("id"))
            new_games += 1
            rows.extend(game_to_samples(game, min_elo))
        time.sleep(1)        # throttle entre joueurs

    if rows:
        shard = buffer_dir / f"stream_{int(time.time())}.txt.gz"
        write_samples(shard, rows)
        print(f"  +{new_games} parties / {len(rows)} samples -> {shard.name}")
    return new_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", default="blitz", choices=["blitz", "rapid", "classical"])
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--interval", type=int, default=120, help="secondes entre polls")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config()
    cfg["_perf"] = args.perf
    timeout = cfg["lichess"]["request_timeout"]

    print(f"Récupération du top {args.top} {args.perf}…")
    users = fetch_top_players(args.perf, args.top, timeout)
    print(f"{len(users)} joueurs suivis. Poll toutes les {args.interval}s.")

    seen: set = set()
    since_ms = int((time.time() - 3600) * 1000)   # 1h d'historique au démarrage
    while True:
        try:
            poll_once(users, cfg, seen, since_ms)
        except Exception as e:           # robustesse : on ne meurt pas sur une erreur réseau
            print(f"erreur poll: {e}")
        since_ms = int(time.time() * 1000)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
