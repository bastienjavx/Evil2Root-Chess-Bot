"""Format de samples partagé : lignes texte gzip.

  fen <TAB> move_uci <TAB> value [ <TAB> policy ]

- value ∈ {-1, 0, 1} = résultat de la partie DU POINT DE VUE DU JOUEUR AU TRAIT
  dans cette position (cohérent avec la tête valeur du réseau).
- policy (4ᵉ colonne, OPTIONNELLE) = distribution de visites MCTS pour le
  self-play AlphaZero, encodée `uci:poids uci:poids …` (poids = entier de visites).
  Absente pour les samples humains (stream/pgn) : un seul coup connu -> la cible
  politique retombe sur un one-hot du `move_uci`.

On stocke la FEN plutôt que les plans encodés : disque léger, encodage à la volée.
`iter_samples` ignore la 4ᵉ colonne (rétro-compatible) ; `iter_samples_pi` la rend.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator


def _format_policy(pi) -> str:
    """{uci: poids} (ou liste de paires) -> 'uci:poids uci:poids …'."""
    items = pi.items() if isinstance(pi, dict) else pi
    return " ".join(f"{uci}:{int(w)}" for uci, w in items)


def _parse_policy(field: str) -> dict[str, float] | None:
    field = field.strip()
    if not field:
        return None
    out: dict[str, float] = {}
    for tok in field.split():
        uci, _, w = tok.partition(":")
        if uci:
            out[uci] = float(w) if w else 1.0
    return out or None


def write_samples(path: str | Path, rows: Iterable[tuple], mode: str = "wt") -> int:
    """Écrit/ajoute des samples. Chaque row est (fen, move, value) OU
    (fen, move, value, policy) avec policy = {uci: poids} (ou None). Retourne le
    nombre de lignes écrites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, mode, encoding="utf-8") as f:
        for row in rows:
            fen, move_uci, value = row[0], row[1], row[2]
            pi = row[3] if len(row) > 3 else None
            if pi:
                f.write(f"{fen}\t{move_uci}\t{value}\t{_format_policy(pi)}\n")
            else:
                f.write(f"{fen}\t{move_uci}\t{value}\n")
            n += 1
    return n


def iter_samples(path: str | Path) -> Iterator[tuple[str, str, int]]:
    """Rétro-compatible : (fen, move_uci, value). Ignore une éventuelle 4ᵉ colonne."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            yield parts[0], parts[1], int(parts[2])


def iter_samples_pi(path: str | Path):
    """(fen, move_uci, value, pi) où pi = {uci: poids} ou None (samples humains)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            pi = _parse_policy(parts[3]) if len(parts) > 3 else None
            yield parts[0], parts[1], int(parts[2]), pi
