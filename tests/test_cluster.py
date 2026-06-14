"""Tests du coordinateur de cluster (sanchess.cluster.server).

Couvre register/work, upload (valide + invalide), publish (auth), stats et la rotation
du buffer. N'a PAS besoin de GPU ni de torch : le serveur est volontairement torch-free
et on fabrique les shards avec data.samples.write_samples (le format réel des workers).
"""

from __future__ import annotations

import gzip
import importlib
import io

import chess
import pytest

# Le serveur lit POOL_DIR / TRAINER_TOKEN À L'IMPORT (constantes de module + _init_storage).
# On configure donc l'environnement AVANT d'importer, dans une fixture qui (re)charge le
# module sur un répertoire temporaire neuf par test -> isolation totale.
TRAINER_TOKEN = "test-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("POOL_DIR", str(tmp_path))
    monkeypatch.setenv("TRAINER_TOKEN", TRAINER_TOKEN)
    monkeypatch.setenv("POOL_MAX_SHARDS", "5")
    monkeypatch.setenv("POOL_UPLOADS_PER_MIN", "1000")

    import sanchess.cluster.server as server
    server = importlib.reload(server)        # relit l'env, réinitialise le stockage
    with TestClient(server.app) as c:
        yield c, server


def _make_shard(n_games: int = 1, plies: int = 8) -> bytes:
    """Génère un shard gzip valide (mini self-play réel) au format samples.py."""
    from sanchess.data.samples import write_samples
    rows = []
    for _ in range(n_games):
        board = chess.Board()
        history = []
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            mv = moves[0]
            history.append((board.fen(), mv.uci(), board.turn))
            board.push(mv)
        n = len(history)
        for i, (fen, uci, turn) in enumerate(history):
            rows.append((fen, uci, 0, {uci: 10}, n - i))
    buf = io.BytesIO()
    # write_samples écrit sur disque ; on passe par un fichier mémoire via gzip direct.
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".txt.gz")
    os.close(fd)
    write_samples(path, rows)
    data = open(path, "rb").read()
    os.unlink(path)
    return data


def _upload(client, raw: bytes, name: str = "tester", worker_id: str = "w1"):
    return client.post("/cluster/upload",
                       files={"shard": ("s.txt.gz", raw, "application/gzip")},
                       data={"worker_id": worker_id, "name": name})


def test_stats_empty(client):
    c, _ = client
    r = c.get("/cluster/stats")
    assert r.status_code == 200
    s = r.json()
    assert s["model_version"] == 0
    assert s["has_model"] is False
    assert s["total_samples"] == 0


def test_model_current_and_download_absent(client):
    c, _ = client
    assert c.get("/cluster/model/current").json()["has_model"] is False
    assert c.get("/cluster/model/download").status_code == 404


def test_work_returns_job_and_worker_id(client):
    c, _ = client
    r = c.post("/cluster/work", json={"name": "alice", "device": "cpu"})
    assert r.status_code == 200
    body = r.json()
    assert body["worker_id"]
    assert "job" in body and body["job"]["nodes"] > 0
    assert body["model"]["version"] == 0


def test_upload_valid_credits_leaderboard(client):
    c, _ = client
    raw = _make_shard(n_games=2, plies=10)
    r = _upload(c, raw)
    assert r.status_code == 200, r.text
    ack = r.json()
    assert ack["ok"] is True
    assert ack["accepted_samples"] == 20      # 2 parties x 10 demi-coups
    assert ack["rank"] == 1
    stats = c.get("/cluster/stats").json()
    assert stats["total_samples"] == 20
    assert stats["leaderboard"][0]["name"] == "tester"


def test_upload_invalid_gzip_rejected(client):
    c, _ = client
    r = _upload(c, b"ceci n'est pas du gzip")
    assert r.status_code == 422
    assert "refusé" in r.json()["error"]


def test_upload_illegal_move_rejected(client):
    c, _ = client
    # FEN de départ mais coup illégal (e2e5).
    buf = io.BytesIO()
    with gzip.open(buf, "wt", encoding="utf-8") as f:
        f.write(f"{chess.STARTING_FEN}\te2e5\t0\n")
    r = _upload(c, buf.getvalue())
    assert r.status_code == 422


def test_upload_bad_value_rejected(client):
    c, _ = client
    buf = io.BytesIO()
    with gzip.open(buf, "wt", encoding="utf-8") as f:
        f.write(f"{chess.STARTING_FEN}\te2e4\t5\n")     # valeur hors {-1,0,1}
    r = _upload(c, buf.getvalue())
    assert r.status_code == 422


def test_publish_requires_token(client):
    c, _ = client
    blob = b"\x80\x02fake-checkpoint-bytes"
    r = c.post("/cluster/model/publish",
               files={"model": ("weights.pt", blob, "application/octet-stream")},
               data={"step": 100})
    assert r.status_code == 401


def test_publish_then_download_roundtrip(client):
    c, _ = client
    blob = b"\x80\x02fake-checkpoint-bytes-123456"
    r = c.post("/cluster/model/publish",
               headers={"Authorization": f"Bearer {TRAINER_TOKEN}"},
               files={"model": ("weights.pt", blob, "application/octet-stream")},
               data={"step": 100})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["version"] == 1
    assert info["step"] == 100
    assert info["has_model"] is True
    # /current reflète la nouvelle version, /download rend les octets exacts.
    assert c.get("/cluster/model/current").json()["version"] == 1
    dl = c.get("/cluster/model/download")
    assert dl.status_code == 200
    assert dl.content == blob
    # Republier réincrémente la version.
    r2 = c.post("/cluster/model/publish",
                headers={"Authorization": f"Bearer {TRAINER_TOKEN}"},
                files={"model": ("weights.pt", blob + b"x", "application/octet-stream")},
                data={"step": 200})
    assert r2.json()["version"] == 2


def test_shards_listing_and_cursor(client):
    c, _ = client
    _upload(c, _make_shard(), name="a", worker_id="wa")
    _upload(c, _make_shard(), name="b", worker_id="wb")
    listing = c.get("/cluster/shards").json()
    assert len(listing["shards"]) == 2
    cursor = listing["shards"][0]
    after = c.get("/cluster/shards", params={"since": cursor}).json()
    assert len(after["shards"]) == 1
    # Téléchargement d'un shard par le trainer.
    name = listing["shards"][0]
    dl = c.get("/cluster/shards", params={"download": name})
    assert dl.status_code == 200 and len(dl.content) > 0


def test_buffer_rotation_caps_shards(client):
    c, server = client
    for i in range(8):                        # MAX_SHARDS=5 (fixture)
        _upload(c, _make_shard(), name="r", worker_id=f"w{i}")
    remaining = list(server.INCOMING_DIR.glob("*.txt.gz"))
    assert len(remaining) <= 5
