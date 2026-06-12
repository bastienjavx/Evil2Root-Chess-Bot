"""Livre des ouvertures Polyglot (.bin) — jouer la théorie sans le réseau.

Améliore le jeu en ouverture SANS réentraîner : pour les `max_ply` premiers
demi-coups on consulte un livre Polyglot standard, et on ne lance le MCTS que
lorsqu'on en sort. Compatible avec n'importe quel `.bin` Polyglot (gm2600,
Perfect20xx, Cerebellum, Titans…) ou un livre construit depuis nos propres PGN
via `python -m sanchess.data.build_book`.

Sélection d'un coup parmi les entrées du livre :
  - "weighted" : tirage pondéré par les poids du livre (varié, plus humain) ;
  - "best"     : toujours le coup le plus lourd (déterministe, en général plus fort).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import chess
import chess.polyglot


class OpeningBook:
    """Un ou plusieurs livres Polyglot consultés dans l'ordre (premier qui répond)."""

    def __init__(self, paths, max_ply: int = 16, min_weight: int = 1,
                 selection: str = "weighted", temperature: float = 1.0):
        self.max_ply = int(max_ply)
        self.min_weight = int(min_weight)
        self.selection = selection if selection in ("weighted", "best") else "weighted"
        self.temperature = max(1e-3, float(temperature))
        self.readers: list[chess.polyglot.MemoryMappedReader] = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                sys.stderr.write(f"[book] introuvable, ignoré : {path}\n")
                continue
            try:
                self.readers.append(chess.polyglot.open_reader(str(path)))
                sys.stderr.write(f"[book] chargé : {path}\n")
            except Exception as e:  # noqa: BLE001 — un livre illisible ne doit pas crasher
                sys.stderr.write(f"[book] échec d'ouverture {path} ({e})\n")

    @classmethod
    def from_config(cls, cfg: dict):
        """Construit le livre depuis la section `book:` de la config, ou None.

        Renvoie None si le livre est désactivé, sans chemin valide, ou si aucun
        fichier n'a pu être ouvert -> l'appelant retombe simplement sur le MCTS.
        """
        b = (cfg or {}).get("book") or {}
        if not b.get("enabled", False):
            return None
        paths = list(b.get("paths") or [])
        if b.get("path"):
            paths.append(b["path"])
        if not paths:
            return None
        book = cls(
            paths,
            max_ply=b.get("max_ply", 16),
            min_weight=b.get("min_weight", 1),
            selection=b.get("selection", "weighted"),
            temperature=b.get("temperature", 1.0),
        )
        return book if book.readers else None

    def lookup(self, board: chess.Board) -> "chess.Move | None":
        """Retourne un coup légal tiré du livre, ou None (hors livre / désactivé)."""
        if not self.readers or board.ply() >= self.max_ply:
            return None
        legal = set(board.legal_moves)
        for reader in self.readers:
            try:
                entries = [e for e in reader.find_all(board)
                           if e.weight >= self.min_weight and e.move in legal]
            except Exception:  # noqa: BLE001 — lecture robuste, on passe au livre suivant
                continue
            if not entries:
                continue
            if self.selection == "best":
                return max(entries, key=lambda e: e.weight).move
            # Tirage pondéré ; température >1 aplatit (plus varié), <1 durcit.
            weights = [max(e.weight, 1) ** (1.0 / self.temperature) for e in entries]
            return random.choices(entries, weights=weights, k=1)[0].move
        return None

    def close(self) -> None:
        for r in self.readers:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
