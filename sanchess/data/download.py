"""Téléchargement des dumps mensuels Lichess (.pgn.zst).

Source : https://database.lichess.org/  (parties standard rated).
Usage :  python -m sanchess.data.download 2024-01
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

from ..utils import load_config

BASE = "https://database.lichess.org/standard"


def download_month(year_month: str, out_dir: Path) -> Path:
    """Télécharge lichess_db_standard_rated_<year_month>.pgn.zst."""
    fname = f"lichess_db_standard_rated_{year_month}.pgn.zst"
    url = f"{BASE}/{fname}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / fname
    if dest.exists():
        print(f"Déjà présent : {dest}")
        return dest

    print(f"Téléchargement {url}\n(les dumps font plusieurs Go — soyez patient)")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / 1e9:.2f} / {total / 1e9:.2f} Go", end="")
        tmp.rename(dest)
    print(f"\nÉcrit : {dest}")
    return dest


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m sanchess.data.download YYYY-MM")
        sys.exit(1)
    cfg = load_config()
    out_dir = Path(cfg["data"]["raw_dir"])
    download_month(sys.argv[1], out_dir)


if __name__ == "__main__":
    main()
