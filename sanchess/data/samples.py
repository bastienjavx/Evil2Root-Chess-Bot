"""Format de samples partagé : lignes texte gzip `fen<TAB>move_uci<TAB>value`.

value ∈ {-1, 0, 1} = résultat de la partie DU POINT DE VUE DU JOUEUR AU TRAIT
dans cette position (cohérent avec la tête valeur du réseau).
On stocke la FEN plutôt que les plans encodés : disque léger, encodage à la volée.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator


def write_samples(path: str | Path, rows: Iterable[tuple[str, str, int]],
                  mode: str = "wt") -> int:
    """Écrit/ajoute des samples. Retourne le nombre de lignes écrites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, mode, encoding="utf-8") as f:
        for fen, move_uci, value in rows:
            f.write(f"{fen}\t{move_uci}\t{value}\n")
            n += 1
    return n


def iter_samples(path: str | Path) -> Iterator[tuple[str, str, int]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            fen, move_uci, value = line.rstrip("\n").split("\t")
            yield fen, move_uci, int(value)
