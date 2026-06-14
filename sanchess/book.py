"""Livre des ouvertures Polyglot (.bin) — théorie d'ouverture, seule ou couplée
au réseau.

Améliore le jeu en ouverture SANS réentraîner. Deux modes (cf. `book.mode`) :
  - "play"  : pour les `max_ply` premiers demi-coups on JOUE directement un coup
              du livre (instantané, pas de MCTS) ; dès qu'on en sort, le MCTS
              reprend la main. C'est l'ancien comportement.
  - "blend" : le livre ET le réseau décident ENSEMBLE. On lance toujours le MCTS
              (donc le réseau évalue la position), mais les coups du livre voient
              leur prior renforcé à la racine (mélange `mix` livre / réseau).
              La recherche reste libre d'écarter une suite de théorie que le
              réseau juge mauvaise -> on garde le savoir des ouvertures sans
              jouer en aveugle. C'est le mode recommandé.

Compatible avec n'importe quel `.bin` Polyglot (gm2600, Perfect20xx, Cerebellum,
Titans…) ou un livre construit depuis nos propres PGN via
`python -m sanchess.data.build_book`.

Sélection d'un coup en mode "play" :
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
                 selection: str = "weighted", temperature: float = 1.0,
                 mode: str = "blend", mix: float = 0.5):
        self.max_ply = int(max_ply)
        self.min_weight = int(min_weight)
        self.selection = selection if selection in ("weighted", "best") else "weighted"
        self.temperature = max(1e-3, float(temperature))
        # "blend" : le livre biaise les priors MCTS (livre + réseau ensemble) ;
        # "play"  : on joue directement un coup du livre (instantané, sans MCTS).
        self.mode = mode if mode in ("blend", "play") else "blend"
        # Poids du livre dans le mélange des priors à la racine (mode "blend") :
        # 0 = réseau seul, 1 = livre seul. Borné à [0,1].
        self.mix = max(0.0, min(1.0, float(mix)))
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
            mode=b.get("mode", "blend"),
            mix=b.get("mix", 0.5),
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

    def move_weights(self, board: chess.Board) -> dict:
        """Distribution {coup: probabilité} des coups de théorie pour `board`.

        Sert au mode "blend" : ces probabilités sont mélangées aux priors du
        réseau à la racine du MCTS pour que la recherche privilégie la théorie
        sans jouer en aveugle. Normalisée à somme 1 sur les coups du livre.
        Renvoie {} hors livre (profondeur dépassée, aucune entrée, désactivé) ->
        l'appelant lance alors un MCTS normal (réseau seul).

        Le premier livre (dans l'ordre `paths`) qui répond fait foi, comme
        `lookup`. La température aplatit (>1) ou durcit (<1) la distribution.
        """
        if not self.readers or board.ply() >= self.max_ply:
            return {}
        legal = set(board.legal_moves)
        for reader in self.readers:
            try:
                entries = [e for e in reader.find_all(board)
                           if e.weight >= self.min_weight and e.move in legal]
            except Exception:  # noqa: BLE001 — lecture robuste, on passe au suivant
                continue
            if not entries:
                continue
            weights: dict = {}
            for e in entries:
                w = max(e.weight, 1) ** (1.0 / self.temperature)
                # Un même coup peut apparaître en plusieurs entrées : on cumule.
                weights[e.move] = weights.get(e.move, 0.0) + w
            total = sum(weights.values())
            if total <= 0:
                continue
            return {m: w / total for m, w in weights.items()}
        return {}

    def close(self) -> None:
        for r in self.readers:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
