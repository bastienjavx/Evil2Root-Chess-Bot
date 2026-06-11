"""Point d'entrée : python -m sanchess.web

Variables d'environnement :
  SANO1_WEB_HOST (def. 0.0.0.0)  SANO1_WEB_PORT (def. 8000)
  SANCHESS_CONFIG (chemin config.yaml alternatif)
"""

from __future__ import annotations

import os

import uvicorn


def main():
    host = os.environ.get("SANO1_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("SANO1_WEB_PORT", "8000"))
    uvicorn.run("sanchess.web.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
