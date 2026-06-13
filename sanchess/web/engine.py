"""Gestionnaire de moteurs pour l'interface web.

- Découvre les checkpoints du dossier `checkpoints/` et expose leurs métadonnées.
- Charge paresseusement un réseau par checkpoint (cache), avec hot-reload de
  `latest.pt` quand le fichier change (l'entraînement continu se reflète en jeu).
- `analyze()` : lance le MCTS sur une position et renvoie le meilleur coup, la
  valeur, et les coups candidats (visites / prior / Q) -> « pensée » du modèle.
- Thread-safe : chaque modèle a son verrou (le MCTS et l'inférence Torch ne sont
  pas réentrants) ; le cache lui-même est protégé par un verrou global.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import chess
import torch

from ..export import make_inference_model
from ..model import build_model, build_model_from_checkpoint
from ..search.mcts import MCTS, Evaluator, best_move
from ..utils import (load_checkpoint, load_config, load_model_state,
                     resolve_device)

# Mapping valeur [-1,1] -> centipions (style Leela), borné pour l'affichage.
def value_to_cp(v: float) -> int:
    v = max(-0.999, min(0.999, float(v)))
    return int(round(90.0 * math.tan(1.5637541897 * v)))


def value_to_winprob(v: float) -> float:
    return max(0.0, min(1.0, (float(v) + 1.0) / 2.0))


class _ModelEntry:
    __slots__ = ("mcts", "mtime", "step", "model_cfg", "last_root", "last_board")

    def __init__(self, mcts, mtime, step, model_cfg):
        self.mcts = mcts
        self.mtime = mtime
        self.step = step
        self.model_cfg = model_cfg
        # Réutilisation d'arbre entre coups consécutifs d'une même partie (onglets
        # Jouer / Regarder) : la dernière recherche et sa position. `advance_root`
        # valide la position exacte -> une analyse d'une position sans lien repart
        # simplement d'un arbre frais (aucune réutilisation erronée possible).
        self.last_root = None
        self.last_board = None


class EngineManager:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.device = resolve_device(self.cfg.get("train", {}).get("device", "auto"))
        self.ckpt_dir = Path(self.cfg["paths"]["checkpoint_dir"])
        self.latest_path = Path(self.cfg["paths"]["latest"])
        self.default_nodes = self.cfg.get("mcts", {}).get("default_nodes", 800)
        self.tree_reuse = bool(self.cfg.get("mcts", {}).get("tree_reuse", True))
        self._cache: dict[str, _ModelEntry] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._cache_lock = threading.Lock()
        self._step_cache: dict[tuple[str, float], dict] = {}

    # --- Découverte des checkpoints ------------------------------------------
    def _read_meta(self, path: Path) -> dict:
        """Lit (step, model_cfg) sans matérialiser les poids (mmap = lazy)."""
        st = path.stat()
        key = (str(path), st.st_mtime)
        if key not in self._step_cache:
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                self._step_cache[key] = {
                    "step": int(ck.get("step", 0)),
                    "model_cfg": ck.get("model_cfg", {}),
                }
            except Exception as exc:  # checkpoint corrompu / en cours d'écriture
                self._step_cache[key] = {"step": None, "model_cfg": {}, "error": str(exc)}
        return self._step_cache[key]

    def list_models(self) -> list[dict]:
        out = []
        if not self.ckpt_dir.exists():
            return out
        for p in sorted(self.ckpt_dir.glob("*.pt")):
            st = p.stat()
            meta = self._read_meta(p)
            mcfg = meta.get("model_cfg") or {}
            out.append({
                "name": p.name,
                "size": st.st_size,
                "size_mb": round(st.st_size / 1e6, 1),
                "mtime": st.st_mtime,
                "step": meta.get("step"),
                "is_latest": p.resolve() == self.latest_path.resolve(),
                "channels": mcfg.get("channels"),
                "blocks": mcfg.get("blocks"),
                "policy_head": mcfg.get("policy_head"),
                "activation": mcfg.get("activation"),
                "value_head": mcfg.get("value_head"),
                "error": meta.get("error"),
            })
        # le plus récent d'abord
        out.sort(key=lambda m: m["mtime"], reverse=True)
        return out

    # --- Chargement / hot-reload ---------------------------------------------
    def _resolve_path(self, model_name: str | None) -> Path:
        if not model_name or model_name in ("latest", "latest.pt"):
            return self.latest_path
        p = self.ckpt_dir / model_name
        if not p.exists():
            raise FileNotFoundError(f"checkpoint introuvable : {model_name}")
        return p

    def _build(self, path: Path) -> _ModelEntry:
        if path.exists():
            ck = load_checkpoint(path, map_location=self.device)
            model = build_model_from_checkpoint(ck, self.cfg)
            load_model_state(model, ck["model_state"])
            step = int(ck.get("step", 0))
            model_cfg = ck.get("model_cfg", {})
            mtime = path.stat().st_mtime
        else:  # aucun checkpoint : réseau aléatoire (jeu faible mais fonctionnel)
            model = build_model(self.cfg)
            step, model_cfg, mtime = 0, self.cfg.get("model", {}), 0.0
        model = make_inference_model(self.cfg, model, self.device)
        mcts = MCTS(Evaluator(model, self.device), self.cfg)
        return _ModelEntry(mcts, mtime, step, model_cfg)

    def get_engine(self, model_name: str | None) -> tuple[MCTS, threading.Lock, _ModelEntry]:
        path = self._resolve_path(model_name)
        key = str(path.resolve())
        with self._cache_lock:
            lock = self._locks.setdefault(key, threading.Lock())
            entry = self._cache.get(key)
            # (re)construit si absent ou si latest.pt a changé sur disque
            needs = entry is None
            if entry is not None and path.exists():
                if path.stat().st_mtime != entry.mtime:
                    needs = True
            if needs:
                entry = self._build(path)
                self._cache[key] = entry
        return entry.mcts, lock, entry

    # --- Analyse d'une position ----------------------------------------------
    def analyze(self, board: chess.Board, model_name: str | None = None,
                nodes: int | None = None, movetime: float | None = None,
                top_k: int = 6) -> dict:
        mcts, lock, entry = self.get_engine(model_name)
        num_nodes = nodes if nodes else (10_000_000 if movetime else self.default_nodes)
        t0 = time.time()
        with lock:
            reuse = None
            if self.tree_reuse and entry.last_root is not None:
                reuse = mcts.advance_root(entry.last_root, entry.last_board, board)
            root = mcts.run(board, num_nodes, max_seconds=movetime, root=reuse)
            if self.tree_reuse:
                entry.last_root = root
                entry.last_board = board.copy(stack=False)
        elapsed = time.time() - t0

        bm = best_move(root)
        # Coups candidats triés par nombre de visites.
        ranked = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
        top = []
        for mv, ch in ranked[:top_k]:
            # Q vu du joueur au trait à la racine = -Q(fils).
            q_root = -ch.q() if ch.N > 0 else None
            top.append({
                "uci": mv.uci(),
                "san": board.san(mv),
                "visits": ch.N,
                "prior": round(float(ch.prior), 4),
                "q": round(q_root, 4) if q_root is not None else None,
                "cp": value_to_cp(q_root) if q_root is not None else None,
            })

        root_value = root.q()  # côté joueur au trait
        result = {
            "model": self._resolve_path(model_name).name,
            "step": entry.step,
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "root_visits": root.N,
            "nodes": num_nodes if not movetime else None,
            "elapsed": round(elapsed, 3),
            "nps": int(root.N / elapsed) if elapsed > 0 else 0,
            "value": round(float(root_value), 4),
            "cp": value_to_cp(root_value),
            "win_prob": round(value_to_winprob(root_value), 4),
            "bestmove": bm.uci() if bm else None,
            "bestmove_san": board.san(bm) if bm else None,
            "top_moves": top,
            "is_game_over": board.is_game_over(claim_draw=True),
            "result": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None,
        }
        return result
