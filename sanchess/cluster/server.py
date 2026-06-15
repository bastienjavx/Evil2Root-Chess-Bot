"""Coordinateur San-o1 (déployé sur Railway, SANS torch).

Distribue les jobs de self-play, collecte/valide les parties des bénévoles, sert le
modèle courant (octets opaques) et un dashboard public + leaderboard. Tout l'état
persiste sous `POOL_DIR` (un Volume Railway monté sur /data) :

    POOL_DIR/
      weights.pt        # checkpoint weights-only servi aux workers (publié par le trainer)
      weights.json      # ModelInfo : version, step, sha256, size
      incoming/*.txt.gz # shards reçus (buffer glissant, rotation à POOL_MAX_SHARDS)
      pool.sqlite       # registre workers + leaderboard + compteurs (WAL)

Le coordinateur N'IMPORTE PAS torch : il route des octets et ne valide les shards
qu'avec python-chess. Image Railway minimale.

Lancement (local) :
    POOL_DIR=/tmp/pool TRAINER_TOKEN=dev \
        uvicorn sanchess.cluster.server:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sqlite3
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import chess
from fastapi import FastAPI, Header, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import protocol as P

# --- Configuration via environnement -----------------------------------------
POOL_DIR = Path(os.environ.get("POOL_DIR", "/data"))
TRAINER_TOKEN = os.environ.get("TRAINER_TOKEN", "")
MAX_SHARDS = int(os.environ.get("POOL_MAX_SHARDS", P.DEFAULT_MAX_SHARDS))
MAX_UPLOAD_BYTES = int(os.environ.get("POOL_MAX_UPLOAD_BYTES", P.DEFAULT_MAX_UPLOAD_BYTES))
MAX_SHARD_LINES = int(os.environ.get("POOL_MAX_SHARD_LINES", P.DEFAULT_MAX_SHARD_LINES))
VALIDATE_SAMPLE = int(os.environ.get("POOL_VALIDATE_SAMPLE", P.DEFAULT_VALIDATE_SAMPLE))
UPLOADS_PER_MIN = int(os.environ.get("POOL_UPLOADS_PER_MIN", P.DEFAULT_UPLOADS_PER_MIN))
ACTIVE_WINDOW_SEC = int(os.environ.get("POOL_ACTIVE_WINDOW_SEC", 300))

# Federated averaging (plusieurs trainers ; cf. CLUSTER.md). Tailles de blob de poids
# bien plus grandes que des shards -> garde-fou dédié.
FED_ROUND_TIMEOUT = float(os.environ.get("FED_ROUND_TIMEOUT", P.DEFAULT_FED_ROUND_TIMEOUT))
FED_QUORUM = int(os.environ.get("FED_QUORUM", P.DEFAULT_FED_QUORUM))
FED_FINALIZE_GRACE = float(os.environ.get("FED_FINALIZE_GRACE", P.DEFAULT_FED_FINALIZE_GRACE))
MAX_WEIGHTS_BYTES = int(os.environ.get("FED_MAX_WEIGHTS_BYTES", 512 * 1024 * 1024))

# Paramètres du job distribué (réglables sans toucher au code, via env Railway).
JOB_DEFAULTS = P.JobSpec(
    nodes=int(os.environ.get("JOB_NODES", 160)),
    max_plies=int(os.environ.get("JOB_MAX_PLIES", 250)),
    temp_moves=int(os.environ.get("JOB_TEMP_MOVES", 20)),
    temperature=float(os.environ.get("JOB_TEMPERATURE", 1.0)),
    dirichlet_eps=float(os.environ.get("JOB_DIRICHLET_EPS", 0.25)),
    gpu_games=int(os.environ.get("JOB_GPU_GAMES", 64)),
    gpu_leaves_per_game=int(os.environ.get("JOB_GPU_LEAVES", 8)),
    target_samples=int(os.environ.get("JOB_TARGET_SAMPLES", 2000)),
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_PATH = POOL_DIR / P.MODEL_FILENAME
MODEL_META_PATH = POOL_DIR / P.MODEL_META_FILENAME
INCOMING_DIR = POOL_DIR / P.INCOMING_DIRNAME
ROUNDS_DIR = POOL_DIR / P.ROUNDS_DIRNAME
ROUND_META_PATH = POOL_DIR / P.ROUND_META_FILENAME
DB_PATH = POOL_DIR / P.DB_FILENAME


# --- Stockage / base SQLite ---------------------------------------------------

_db_lock = Lock()
_rate: dict[str, deque] = {}          # IP -> timestamps récents (rate-limit upload)
_games_log: deque = deque(maxlen=4000)  # timestamps des parties reçues (débit/min)
# Séries temporelles en mémoire (réinitialisées au redémarrage : OK pour des courbes
# « live ». Les totaux cumulés, eux, sont persistés en sqlite). Un seau par minute.
HISTORY_MINUTES = int(os.environ.get("POOL_HISTORY_MINUTES", 360))
_minute_buckets: deque = deque()      # [minute_epoch, games, samples]
_START_TIME = time.time()


def _record_activity(now: float, games: int, samples: int) -> None:
    """Agrège l'activité dans le seau de la minute courante (pour les graphiques)."""
    minute = int(now // 60) * 60
    if _minute_buckets and _minute_buckets[-1][0] == minute:
        _minute_buckets[-1][1] += games
        _minute_buckets[-1][2] += samples
    else:
        _minute_buckets.append([minute, games, samples])
    cutoff = minute - HISTORY_MINUTES * 60
    while _minute_buckets and _minute_buckets[0][0] < cutoff:
        _minute_buckets.popleft()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_storage() -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    if not ROUND_META_PATH.exists():
        _write_round({"round": 1, "base_version": _read_model_info().version,
                      "started_at": time.time(), "finalizer_id": "",
                      "prev_contributors": []})
    with _db_lock, _connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS contributors ("
            " id TEXT PRIMARY KEY, name TEXT, device TEXT,"
            " games INTEGER DEFAULT 0, samples INTEGER DEFAULT 0,"
            " first_seen REAL, last_seen REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.commit()


def _meta_get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def _read_model_info() -> P.ModelInfo:
    if MODEL_META_PATH.exists():
        try:
            return P.ModelInfo.from_dict(json.loads(MODEL_META_PATH.read_text()))
        except (OSError, ValueError):
            pass
    return P.ModelInfo()


def _write_model_info(info: P.ModelInfo) -> None:
    tmp = MODEL_META_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info.to_dict()))
    os.replace(tmp, MODEL_META_PATH)


# --- Federated averaging : cycle des rounds ----------------------------------
# round.json persiste l'état DURABLE du round (numéro, base_version, finalizer désigné,
# contributeurs du round précédent -> sert à dimensionner le quorum « expected »). Les
# contributions, elles, vivent dans rounds/<R>/ et sont comptées à la volée.

def _read_round() -> dict:
    if ROUND_META_PATH.exists():
        try:
            return json.loads(ROUND_META_PATH.read_text())
        except (OSError, ValueError):
            pass
    return {"round": 1, "base_version": _read_model_info().version,
            "started_at": time.time(), "finalizer_id": "", "prev_contributors": []}


def _write_round(data: dict) -> None:
    tmp = ROUND_META_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, ROUND_META_PATH)


def _round_contributors(round_no: int) -> list[str]:
    """ids des trainers ayant contribué au round (un .pt = une contribution)."""
    rdir = ROUNDS_DIR / str(round_no)
    if not rdir.exists():
        return []
    return sorted(p.stem for p in rdir.glob("*.pt"))


def _gc_round_dir(round_no: int) -> None:
    """Supprime le dossier d'un round finalisé (les blobs de poids sont volumineux)."""
    import shutil
    try:
        shutil.rmtree(ROUNDS_DIR / str(round_no))
    except OSError:
        pass


def _round_state(now: float) -> P.RoundInfo:
    """Instantané live du round courant : combine round.json (durable) + comptage des
    contributions sur disque. Ferme par quorum OU deadline ; `can_finalize` ouvre la
    finalisation aux autres trainers après le délai de grâce (reprise si finalizer mort)."""
    rec = _read_round()
    contributors = _round_contributors(rec["round"])
    n = len(contributors)
    expected = max(FED_QUORUM, len(rec.get("prev_contributors") or []))
    deadline = rec["started_at"] + FED_ROUND_TIMEOUT
    closed = n >= 1 and (n >= expected or now >= deadline)
    can_finalize = closed and (n >= expected or now >= deadline + FED_FINALIZE_GRACE)
    return P.RoundInfo(
        round=rec["round"], base_version=rec["base_version"],
        started_at=rec["started_at"], deadline=deadline, expected=expected,
        finalizer_id=rec.get("finalizer_id", ""), num_contributions=n,
        contributors=contributors, closed=closed, can_finalize=can_finalize)


# --- Validation des shards uploadés ------------------------------------------

class _Reject(Exception):
    """Shard refusé : message lisible renvoyé au worker (422)."""


def _validate_shard(raw: bytes) -> int:
    """Décompresse, vérifie le format, échantillonne quelques lignes (FEN parsable +
    coup légal). Retourne le nombre de lignes (samples). Lève _Reject si invalide.

    On ne valide qu'un échantillon : valider chaque coup d'un gros shard serait
    coûteux, mais quelques tirages suffisent à écarter le bruit/le spam évident.
    """
    if len(raw) > MAX_UPLOAD_BYTES:
        raise _Reject(f"shard trop gros ({len(raw)} > {MAX_UPLOAD_BYTES} o)")
    try:
        with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
            text = f.read(P.DEFAULT_MAX_DECOMPRESSED_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise _Reject(f"gzip illisible : {exc}")
    if len(text) > P.DEFAULT_MAX_DECOMPRESSED_BYTES:
        raise _Reject("shard décompressé trop volumineux")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise _Reject("shard vide")
    if len(lines) > MAX_SHARD_LINES:
        raise _Reject(f"trop de lignes ({len(lines)} > {MAX_SHARD_LINES})")

    # Échantillon réparti sur tout le shard.
    n = len(lines)
    step = max(1, n // max(1, VALIDATE_SAMPLE))
    checked = 0
    for i in range(0, n, step):
        parts = lines[i].rstrip("\n").split("\t")
        if len(parts) < 3:
            raise _Reject(f"ligne {i} mal formée (colonnes < 3)")
        fen, move_uci, value = parts[0], parts[1], parts[2]
        if value not in ("-1", "0", "1"):
            raise _Reject(f"ligne {i} : valeur invalide {value!r}")
        try:
            board = chess.Board(fen)
        except ValueError:
            raise _Reject(f"ligne {i} : FEN invalide")
        try:
            mv = chess.Move.from_uci(move_uci)
        except ValueError:
            raise _Reject(f"ligne {i} : coup UCI invalide {move_uci!r}")
        if mv not in board.legal_moves:
            raise _Reject(f"ligne {i} : coup illégal {move_uci} pour la position")
        checked += 1
        if checked >= VALIDATE_SAMPLE:
            break
    return n


def _count_games(raw: bytes) -> int:
    """Estime le nb de parties d'un shard : une partie se termine par une position
    dont plies_to_end (5ᵉ colonne) vaut 1 ou 0. Repli : ~1 partie / 80 samples."""
    try:
        with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
            games = 0
            total = 0
            for line in f:
                total += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 5 and parts[4].strip() in ("0", "1"):
                    games += 1
        return games if games else max(1, total // 80)
    except (OSError, EOFError):
        return 1


def _rotate_incoming() -> int:
    """Supprime les plus vieux shards au-delà de MAX_SHARDS. Retourne le nb restant."""
    shards = sorted(INCOMING_DIR.glob("*.txt.gz"))
    excess = len(shards) - MAX_SHARDS
    for shard in shards[:max(0, excess)]:
        try:
            shard.unlink()
        except OSError:
            pass
    return min(len(shards), MAX_SHARDS)


def _rate_limited(ip: str) -> bool:
    now = time.time()
    dq = _rate.setdefault(ip, deque())
    while dq and now - dq[0] > 60.0:
        dq.popleft()
    if len(dq) >= UPLOADS_PER_MIN:
        return True
    dq.append(now)
    return False


# --- Application FastAPI ------------------------------------------------------

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Au démarrage (uvicorn / TestClient), pas à l'import : évite de planter si le
    # Volume Railway n'est pas encore monté ou si /data n'est pas accessible à l'import.
    _init_storage()
    yield


app = FastAPI(title="San-o1 Cluster Coordinator", version="1.0", lifespan=_lifespan)


@app.get(P.EP_MODEL_CURRENT)
def model_current():
    return _read_model_info().to_dict()


@app.get(P.EP_MODEL_DOWNLOAD)
def model_download():
    if not MODEL_PATH.exists():
        return JSONResponse({"error": "aucun modèle publié"}, status_code=404)
    return FileResponse(MODEL_PATH, media_type="application/octet-stream",
                        filename=P.MODEL_FILENAME)


@app.post(P.EP_WORK)
async def work(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    name = (body.get("name") or "anon").strip()[:48]
    device = (body.get("device") or "?")[:16]
    wid = (body.get("worker_id") or "").strip() or uuid.uuid4().hex
    now = time.time()
    with _db_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO contributors(id, name, device, first_seen, last_seen) "
            "VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "name=excluded.name, device=excluded.device, last_seen=excluded.last_seen",
            (wid, name, device, now, now))
        conn.commit()
    info = _read_model_info()
    job = P.JobSpec(**{**JOB_DEFAULTS.to_dict(), "model_version": info.version})
    return {"worker_id": wid, "job": job.to_dict(), "model": info.to_dict()}


@app.post(P.EP_UPLOAD)
async def upload(request: Request,
                 shard: UploadFile = File(...),
                 worker_id: str = Form(""),
                 name: str = Form("anon")):
    ip = request.client.host if request.client else "?"
    if _rate_limited(ip):
        return JSONResponse({"error": "rate limit : trop d'uploads"}, status_code=429)
    raw = await shard.read()
    try:
        n_samples = _validate_shard(raw)
    except _Reject as exc:
        return JSONResponse({"error": f"shard refusé : {exc}"}, status_code=422)

    n_games = _count_games(raw)
    fname = f"w_{(worker_id or 'anon')[:16]}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.txt.gz"
    tmp = INCOMING_DIR / f".{fname}.partial"
    tmp.write_bytes(raw)
    os.replace(tmp, INCOMING_DIR / fname)
    pending = _rotate_incoming()

    now = time.time()
    wid = (worker_id or "").strip() or uuid.uuid4().hex
    with _db_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO contributors(id, name, device, games, samples, first_seen, last_seen) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "name=excluded.name, games=games+excluded.games, "
            "samples=samples+excluded.samples, last_seen=excluded.last_seen",
            (wid, name.strip()[:48], "?", n_games, n_samples, now, now))
        total = conn.execute("SELECT games, samples FROM contributors WHERE id=?",
                             (wid,)).fetchone()
        better = conn.execute(
            "SELECT COUNT(*) FROM contributors WHERE samples > ?", (total[1],)).fetchone()[0]
        conn.commit()
    for _ in range(n_games):
        _games_log.append(now)
    _record_activity(now, n_games, n_samples)

    ack = P.UploadAck(ok=True, accepted_samples=n_samples, shard=fname,
                      total_samples=total[1], rank=better + 1)
    return ack.to_dict()


@app.get(P.EP_SHARDS)
def shards(since: str = "", download: str = ""):
    """Trainer : liste les nouveaux shards (since = dernier nom traité), ou en
    télécharge un (download = nom de fichier)."""
    if download:
        target = INCOMING_DIR / Path(download).name
        if not target.exists() or target.suffix != ".gz":
            return JSONResponse({"error": "shard introuvable"}, status_code=404)
        return FileResponse(target, media_type="application/octet-stream",
                            filename=target.name)
    names = sorted(p.name for p in INCOMING_DIR.glob("*.txt.gz"))
    if since:
        names = [n for n in names if n > since]
    return {"shards": names[:500], "cursor": names[-1] if names else since}


@app.post(P.EP_MODEL_PUBLISH)
async def model_publish(request: Request,
                        model: UploadFile = File(...),
                        step: int = Form(0),
                        round: int = Form(-1),
                        authorization: str = Header("")):
    token = authorization.replace("Bearer", "").strip()
    if not TRAINER_TOKEN or token != TRAINER_TOKEN:
        return JSONResponse({"error": "trainer token invalide"}, status_code=401)
    raw = await model.read()
    if not raw:
        return JSONResponse({"error": "fichier vide"}, status_code=400)
    if len(raw) > MAX_WEIGHTS_BYTES:
        return JSONResponse({"error": "modèle trop volumineux"}, status_code=413)
    sha = hashlib.sha256(raw).hexdigest()
    # Section critique sous _db_lock : la garde idempotente par round ET l'incrément de
    # version doivent être atomiques, sinon deux finalizers FedAvg incrémenteraient la
    # version deux fois pour le même round. `round=-1` = publish mono-trainer historique
    # (aucune garde, comportement strictement inchangé).
    with _db_lock:
        if round >= 0:
            cur = _read_round()
            if round != cur["round"]:
                return JSONResponse(
                    {"error": "round déjà finalisé", "round": cur["round"],
                     "version": _read_model_info().version}, status_code=409)
        tmp = MODEL_PATH.with_suffix(".pt.tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, MODEL_PATH)
        prev = _read_model_info()
        info = P.ModelInfo(version=prev.version + 1, step=int(step), sha256=sha,
                           size=len(raw), has_model=True, published_at=time.time())
        _write_model_info(info)
        if round >= 0:
            # Avance au round suivant : base = version publiée ; on garde les
            # contributeurs de ce round pour dimensionner le quorum du suivant ; purge.
            done = _round_contributors(round)
            _write_round({"round": round + 1, "base_version": info.version,
                          "started_at": time.time(), "finalizer_id": "",
                          "prev_contributors": done})
            _gc_round_dir(round)
        else:
            # Publish mono-trainer / amorçage (round=-1) : on n'avance PAS le round,
            # mais on aligne sa base_version sur le modèle servi pour que les
            # contributions FedAvg ultérieures (base_version == version courante) passent.
            cur = _read_round()
            if cur.get("base_version") != info.version:
                cur["base_version"] = info.version
                _write_round(cur)
    return info.to_dict()


@app.get(P.EP_ROUND)
def round_state():
    return _round_state(time.time()).to_dict()


@app.post(P.EP_CONTRIBUTE)
async def contribute(weights: UploadFile = File(...),
                     trainer_id: str = Form(...),
                     base_version: int = Form(0),
                     num_samples: int = Form(0),
                     step: int = Form(0),
                     authorization: str = Header("")):
    token = authorization.replace("Bearer", "").strip()
    if not TRAINER_TOKEN or token != TRAINER_TOKEN:
        return JSONResponse({"error": "trainer token invalide"}, status_code=401)
    raw = await weights.read()
    if not raw:
        return JSONResponse({"error": "fichier vide"}, status_code=400)
    if len(raw) > MAX_WEIGHTS_BYTES:
        return JSONResponse({"error": "modèle trop volumineux"}, status_code=413)
    tid = "".join(c for c in trainer_id if c.isalnum() or c in "-_")[:48] or "anon"
    sha = hashlib.sha256(raw).hexdigest()
    with _db_lock:
        cur = _read_round()
        if base_version != cur["base_version"]:
            # Le trainer a entraîné sur une version périmée -> sa contribution
            # mélangerait deux lignées. On le renvoie resynchroniser le modèle global.
            return JSONResponse(
                {"error": "base_version périmée : resynchronise le modèle global",
                 "round": cur["round"], "base_version": cur["base_version"]},
                status_code=409)
        rdir = ROUNDS_DIR / str(cur["round"])
        rdir.mkdir(parents=True, exist_ok=True)
        tmp = rdir / f".{tid}.pt.partial"
        tmp.write_bytes(raw)
        os.replace(tmp, rdir / f"{tid}.pt")
        (rdir / f"{tid}.json").write_text(json.dumps(
            {"trainer_id": tid, "num_samples": int(num_samples), "step": int(step),
             "sha256": sha, "at": time.time()}))
        if not cur.get("finalizer_id"):     # 1er contributeur = finalizer désigné
            cur["finalizer_id"] = tid
            _write_round(cur)
        state = _round_state(time.time())
    return {"round": state.round, "is_finalizer": state.finalizer_id == tid,
            "finalizer_id": state.finalizer_id,
            "num_contributions": state.num_contributions,
            "expected": state.expected, "closed": state.closed}


@app.get(P.EP_CONTRIBUTIONS)
def contributions(round: int = -1, download: str = ""):
    """Finalizer : liste les contributions du round (avec num_samples pour la moyenne
    pondérée), ou en télécharge une (`download=<trainer_id>`)."""
    r = _read_round()["round"] if round < 0 else round
    rdir = ROUNDS_DIR / str(r)
    if download:
        target = rdir / f"{Path(download).name}.pt"
        if not target.exists():
            return JSONResponse({"error": "contribution introuvable"}, status_code=404)
        return FileResponse(target, media_type="application/octet-stream",
                            filename=target.name)
    items = []
    for meta in sorted(rdir.glob("*.json")):
        try:
            items.append(json.loads(meta.read_text()))
        except (OSError, ValueError):
            continue
    return {"round": r, "contributions": items}


def _snapshot_stats() -> P.Stats:
    now = time.time()
    while _games_log and now - _games_log[0] > 60.0:
        _games_log.popleft()
    info = _read_model_info()
    with _db_lock, _connect() as conn:
        agg = conn.execute(
            "SELECT COALESCE(SUM(games),0), COALESCE(SUM(samples),0), COUNT(*) "
            "FROM contributors").fetchone()
        active = conn.execute(
            "SELECT COUNT(*) FROM contributors WHERE last_seen > ?",
            (now - ACTIVE_WINDOW_SEC,)).fetchone()[0]
        board = conn.execute(
            "SELECT name, games, samples, device FROM contributors "
            "ORDER BY samples DESC LIMIT 20").fetchall()
        devices = conn.execute(
            "SELECT COALESCE(device,'?'), COUNT(*), COALESCE(SUM(samples),0) "
            "FROM contributors GROUP BY device ORDER BY 3 DESC").fetchall()
    pending = len(list(INCOMING_DIR.glob("*.txt.gz")))
    hour_ago = (int(now // 60) * 60) - 3600
    games_hr = sum(b[1] for b in _minute_buckets if b[0] >= hour_ago)
    samples_hr = sum(b[2] for b in _minute_buckets if b[0] >= hour_ago)
    rstate = _round_state(now)
    return P.Stats(
        model_version=info.version, model_step=info.step, has_model=info.has_model,
        model_round=rstate.round, active_trainers=rstate.num_contributions,
        model_published_at=info.published_at,
        total_games=agg[0], total_samples=agg[1], total_contributors=agg[2],
        active_workers=active, games_per_min=float(len(_games_log)),
        games_last_hour=games_hr, samples_last_hour=samples_hr,
        pending_shards=pending, uptime_sec=now - _START_TIME, server_time=now,
        device_breakdown=[{"device": d[0], "workers": d[1], "samples": d[2]}
                          for d in devices],
        leaderboard=[{"name": r[0] or "anon", "games": r[1], "samples": r[2],
                      "device": r[3] or "?"} for r in board])


@app.get(P.EP_STATS)
def stats():
    return _snapshot_stats().to_dict()


@app.get(P.EP_HISTORY)
def history(minutes: int = 180):
    """Série temporelle par minute pour les graphiques (débit + croissance cumulée).
    Réinitialisée au redémarrage du serveur (courbes 'live' ; les totaux persistent)."""
    minutes = max(1, min(int(minutes), HISTORY_MINUTES))
    now = time.time()
    cutoff = (int(now // 60) * 60) - minutes * 60
    cum = 0
    series = []
    # Cumul reconstruit sur la fenêtre conservée (point de départ relatif).
    for ts, g, s in _minute_buckets:
        cum += s
        if ts >= cutoff:
            series.append({"t": ts, "games": g, "samples": s, "cum_samples": cum})
    return {"series": series, "server_time": now, "minutes": minutes}


# --- Dashboard ----------------------------------------------------------------

@app.get("/cluster/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def dashboard():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>San-o1 Cluster</h1><p>dashboard manquant</p>")


if STATIC_DIR.exists():
    app.mount("/cluster/static", StaticFiles(directory=STATIC_DIR), name="cluster-static")
