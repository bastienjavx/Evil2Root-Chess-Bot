"""Serveur FastAPI San-o1 : API REST + WebSocket + frontend statique.

Routes principales :
  GET  /                     -> interface web (SPA)
  GET  /api/models           -> checkpoints disponibles (+ métadonnées)
  GET  /api/config           -> config.yaml courante
  GET  /api/system           -> GPU / services / buffer / process / entraînement
  GET  /api/training         -> séries de courbes (policy/value/loss)
  POST /api/analyze          -> analyse MCTS d'une position (FEN)
  POST /api/move             -> le modèle joue un coup sur une position (live)
  WS   /ws/play              -> partie en direct, un coup par message
  WS   /ws/selfplay          -> le modèle joue contre lui-même, streamé coup par coup

Exposition publique :  cloudflared tunnel --url http://localhost:8000
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import chess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..utils import load_config
from . import stats
from .engine import EngineManager

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="San-o1 Web", version="1.0")

_cfg = load_config(os.environ.get("SANCHESS_CONFIG"))
manager = EngineManager(_cfg)


# --- Modèles de requête -------------------------------------------------------
class AnalyzeReq(BaseModel):
    fen: str | None = None
    model: str | None = None
    nodes: int | None = None
    movetime: float | None = None
    top_k: int = 6


class MoveReq(BaseModel):
    fen: str | None = None
    moves: list[str] | None = None  # coups UCI depuis la position de départ
    model: str | None = None
    nodes: int | None = None
    movetime: float | None = None


def _board_from(fen: str | None, moves: list[str] | None) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    for mv in (moves or []):
        board.push_uci(mv)
    return board


def _run(fn, *a, **k):
    """Exécute un appel bloquant (MCTS/Torch) dans le threadpool."""
    return asyncio.get_event_loop().run_in_executor(None, lambda: fn(*a, **k))


# --- API REST -----------------------------------------------------------------
@app.get("/api/models")
def api_models():
    return {"device": manager.device, "models": manager.list_models()}


@app.get("/api/config")
def api_config():
    return _cfg


@app.get("/api/system")
def api_system():
    return stats.system_status()


@app.get("/api/training")
def api_training(points: int = 400):
    return stats.training_series(max_points=points)


@app.get("/api/legal")
def api_legal(fen: str | None = None, moves: str | None = None):
    """Coups légaux d'une position (logique d'échecs côté serveur -> frontend
    sans dépendance). `moves` = coups UCI séparés par des virgules."""
    try:
        mlist = moves.split(",") if moves else []
        board = _board_from(fen, [m for m in mlist if m])
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    legal = []
    for mv in board.legal_moves:
        legal.append({"uci": mv.uci(), "san": board.san(mv),
                      "from": chess.square_name(mv.from_square),
                      "to": chess.square_name(mv.to_square)})
    over = board.is_game_over(claim_draw=True)
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "in_check": board.is_check(),
        "legal": legal,
        "is_game_over": over,
        "result": board.result(claim_draw=True) if over else None,
        "last_move": board.peek().uci() if board.move_stack else None,
    }


@app.post("/api/analyze")
async def api_analyze(req: AnalyzeReq):
    try:
        board = chess.Board(req.fen) if req.fen else chess.Board()
    except ValueError:
        return JSONResponse({"error": "FEN invalide"}, status_code=400)
    res = await _run(manager.analyze, board, req.model, req.nodes,
                     req.movetime, req.top_k)
    return res


@app.post("/api/move")
async def api_move(req: MoveReq):
    try:
        board = _board_from(req.fen, req.moves)
    except ValueError as exc:
        return JSONResponse({"error": f"position invalide : {exc}"}, status_code=400)
    if board.is_game_over(claim_draw=True):
        return {"bestmove": None, "fen": board.fen(),
                "is_game_over": True, "result": board.result(claim_draw=True)}
    res = await _run(manager.analyze, board, req.model, req.nodes, req.movetime)
    if res["bestmove"]:
        board.push_uci(res["bestmove"])
        res["fen_after"] = board.fen()
        res["is_game_over"] = board.is_game_over(claim_draw=True)
        res["result"] = board.result(claim_draw=True) if res["is_game_over"] else None
    return res


# --- WebSocket : partie humain vs modèle -------------------------------------
@app.websocket("/ws/play")
async def ws_play(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            fen = msg.get("fen")
            moves = msg.get("moves")
            try:
                board = _board_from(fen, moves)
            except ValueError as exc:
                await ws.send_json({"type": "error", "error": str(exc)})
                continue
            if board.is_game_over(claim_draw=True):
                await ws.send_json({"type": "gameover",
                                    "result": board.result(claim_draw=True),
                                    "fen": board.fen()})
                continue
            res = await _run(manager.analyze, board, msg.get("model"),
                             msg.get("nodes"), msg.get("movetime"))
            if res["bestmove"]:
                board.push_uci(res["bestmove"])
                res["fen_after"] = board.fen()
                res["is_game_over"] = board.is_game_over(claim_draw=True)
                res["result"] = board.result(claim_draw=True) if res["is_game_over"] else None
            res["type"] = "move"
            await ws.send_json(res)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # ne jamais laisser tomber la connexion sans message
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass


# --- WebSocket : le modèle joue contre lui-même, streamé ---------------------
@app.websocket("/ws/selfplay")
async def ws_selfplay(ws: WebSocket):
    await ws.accept()
    stop = False
    try:
        cfg = await ws.receive_json()
        model = cfg.get("model")
        model_white = cfg.get("model_white", model)
        model_black = cfg.get("model_black", model)
        nodes = cfg.get("nodes", 200)
        movetime = cfg.get("movetime")
        max_plies = int(cfg.get("max_plies", 300))
        start_fen = cfg.get("fen")
        board = chess.Board(start_fen) if start_fen else chess.Board()

        await ws.send_json({"type": "start", "fen": board.fen(),
                            "white": (model_white or "latest"),
                            "black": (model_black or "latest")})

        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < max_plies and not stop:
            who = model_white if board.turn == chess.WHITE else model_black
            res = await _run(manager.analyze, board, who, nodes, movetime, 4)
            mv = res["bestmove"]
            if not mv:
                break
            san = res["bestmove_san"]
            board.push_uci(mv)
            ply += 1
            await ws.send_json({
                "type": "move", "ply": ply, "uci": mv, "san": san,
                "fen": board.fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "cp": res["cp"], "value": res["value"], "win_prob": res["win_prob"],
                "visits": res["root_visits"], "nps": res["nps"],
                "top_moves": res["top_moves"],
            })
            # laisse le temps au client de respirer / d'annuler
            await asyncio.sleep(0.01)

        await ws.send_json({"type": "gameover", "fen": board.fen(),
                            "result": board.result(claim_draw=True)
                            if board.is_game_over(claim_draw=True) else "*",
                            "plies": ply})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass


# --- Frontend statique --------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
