"""Moteur UCI San-o1 : utilisable dans Nibbler, Cutechess, Arena, lichess-bot...

Lancement :  python -m sanchess.uci
Recharge à chaud `checkpoints/latest.pt` à chaque `ucinewgame` -> l'apprentissage
continu en arrière-plan se reflète automatiquement en jeu.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chess
import torch

from .book import OpeningBook
from .model import build_model, build_model_from_checkpoint
from .search.mcts import MCTS, Evaluator, best_move
from .utils import (load_checkpoint, load_config, load_model_state,
                    resolve_device)

NAME = "San-o1"
AUTHOR = "San-o1 project"


class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = resolve_device(cfg.get("train", {}).get("device", "auto"))
        self.ckpt_path = Path(cfg["paths"]["latest"])
        self._ckpt_mtime = None
        self.board = chess.Board()
        self.default_nodes = cfg.get("mcts", {}).get("default_nodes", 800)
        self.book = OpeningBook.from_config(cfg)
        self._build_engine()

    def _build_engine(self):
        if self.ckpt_path.exists():
            ck = load_checkpoint(self.ckpt_path, map_location=self.device)
            model = build_model_from_checkpoint(ck, self.cfg)
            load_model_state(model, ck["model_state"])
            self._ckpt_mtime = self.ckpt_path.stat().st_mtime
            sys.stderr.write(f"info string poids chargés depuis {self.ckpt_path} "
                             f"(step {ck.get('step', '?')})\n")
        else:
            model = build_model(self.cfg)
            sys.stderr.write("info string AUCUN checkpoint -> réseau aléatoire "
                             "(jeu faible, normal avant entraînement)\n")
        self.mcts = MCTS(Evaluator(model, self.device), self.cfg)

    def maybe_reload(self):
        """Recharge les poids si le checkpoint a changé (hot-reload)."""
        if self.ckpt_path.exists():
            mtime = self.ckpt_path.stat().st_mtime
            if mtime != self._ckpt_mtime:
                self._build_engine()

    def set_position(self, tokens: list[str]):
        if tokens[0] == "startpos":
            self.board = chess.Board()
            moves = tokens[2:] if len(tokens) > 1 and tokens[1] == "moves" else []
        elif tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            self.board = chess.Board(fen)
            moves = tokens[8:] if len(tokens) > 7 and tokens[7] == "moves" else []
        else:
            return
        for mv in moves:
            self.board.push_uci(mv)

    def go(self, tokens: list[str]) -> str:
        nodes, movetime, wtime, btime = None, None, None, None
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "nodes":
                nodes = int(tokens[i + 1]); i += 2
            elif t == "movetime":
                movetime = int(tokens[i + 1]) / 1000.0; i += 2
            elif t == "wtime":
                wtime = int(tokens[i + 1]) / 1000.0; i += 2
            elif t == "btime":
                btime = int(tokens[i + 1]) / 1000.0; i += 2
            else:
                i += 1

        max_seconds = None
        if movetime is not None:
            max_seconds = movetime
        elif wtime is not None or btime is not None:
            remaining = wtime if self.board.turn == chess.WHITE else btime
            if remaining is not None:
                max_seconds = max(0.05, remaining / 30.0)   # ~1/30 du temps restant

        # Livre d'ouvertures : réponse immédiate tant qu'on est dans la théorie.
        if self.book is not None:
            mv = self.book.lookup(self.board)
            if mv is not None:
                sys.stderr.write(f"info string livre {mv.uci()}\n")
                return f"bestmove {mv.uci()}"

        num_nodes = nodes if nodes is not None else (
            10_000_000 if max_seconds is not None else self.default_nodes)

        root = self.mcts.run(self.board, num_nodes, max_seconds=max_seconds)
        mv = best_move(root)
        if mv is None:
            return "bestmove (none)"
        child = root.children[mv]
        score_cp = int(-child.q() * 300)     # heuristique d'affichage
        sys.stderr.write(f"info string visites racine={root.N} "
                         f"meilleur={mv.uci()} N={child.N} score~{score_cp}cp\n")
        return f"bestmove {mv.uci()}"


def main():
    cfg = load_config(os.environ.get("SANCHESS_CONFIG"))
    engine = Engine(cfg)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            print(f"id name {NAME}")
            print(f"id author {AUTHOR}")
            print("option name WeightsFile type string default "
                  + str(engine.ckpt_path))
            print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            engine.maybe_reload()
            engine.board = chess.Board()
        elif cmd == "position":
            engine.set_position(parts[1:])
        elif cmd == "go":
            print(engine.go(parts[1:]), flush=True)
        elif cmd == "setoption":
            pass
        elif cmd in ("quit", "stop"):
            if cmd == "quit":
                break
        # 'stop' : la recherche est synchrone, rien à faire.


if __name__ == "__main__":
    main()
