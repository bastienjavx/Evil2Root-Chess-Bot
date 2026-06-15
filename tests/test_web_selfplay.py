from __future__ import annotations

import chess
import pytest


class DummyManager:
    device = "cpu"

    def analyze(self, board, model_name=None, nodes=None, movetime=None, top_k=6):
        mv = next(iter(board.legal_moves), None)
        top = []
        if mv:
            top.append({
                "uci": mv.uci(),
                "san": board.san(mv),
                "visits": nodes or 1,
                "prior": 1.0,
                "q": 0.0,
                "cp": 0,
            })
        return {
            "model": model_name or "latest.pt",
            "step": 1,
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "root_visits": nodes or 1,
            "elapsed": 0.01,
            "nps": 1000,
            "value": 0.0,
            "cp": 0,
            "win_prob": 0.5,
            "bestmove": mv.uci() if mv else None,
            "bestmove_san": board.san(mv) if mv else None,
            "top_moves": top,
            "is_game_over": board.is_game_over(claim_draw=True),
            "result": board.result(claim_draw=True)
            if board.is_game_over(claim_draw=True) else None,
        }


class FakeWebSocket:
    def __init__(self, cfg):
        self.cfg = cfg
        self.accepted = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def receive_json(self):
        return self.cfg

    async def send_json(self, msg):
        self.sent.append(msg)


@pytest.mark.anyio
async def test_selfplay_websocket_streams_moves_and_gameover(monkeypatch):
    import sanchess.web.server as server

    async def inline_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(server, "manager", DummyManager())
    monkeypatch.setattr(server, "_run", inline_run)

    ws = FakeWebSocket({
        "model_white": "white.pt",
        "model_black": "black.pt",
        "nodes": 7,
        "max_plies": 2,
    })
    await server.ws_selfplay(ws)

    assert ws.accepted is True
    start, first, second, done = ws.sent

    assert start["type"] == "start"
    assert start["white"] == "white.pt"
    assert start["black"] == "black.pt"
    assert start["max_plies"] == 2

    assert first["type"] == "move"
    assert first["ply"] == 1
    assert first["visits"] == 7
    assert first["elapsed"] == 0.01
    assert first["model"] == "white.pt"

    assert second["type"] == "move"
    assert second["ply"] == 2
    assert second["model"] == "black.pt"

    assert done == {
        "type": "gameover",
        "fen": done["fen"],
        "result": "*",
        "plies": 2,
        "reason": "max_plies",
    }
