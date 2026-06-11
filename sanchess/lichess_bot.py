"""Bot Lichess natif pour San-o1.

Pilote directement le moteur MCTS (pas de pont UCI) via l'API Bot de Lichess :
  - écoute les événements (/api/stream/event)
  - accepte les défis (échecs standard)
  - joue chaque partie en streamant son état et en postant ses coups

Pré-requis :
  - un token Lichess avec le scope `bot:play` (dans .env -> LICHESS_TOKEN)
  - le compte doit être un COMPTE BOT (voir `--upgrade`, irréversible)

Usage :
  python -m sanchess.lichess_bot --check      # affiche l'état du compte (lecture seule)
  python -m sanchess.lichess_bot --upgrade    # convertit le compte en BOT (IRRÉVERSIBLE)
  python -m sanchess.lichess_bot              # lance le bot (boucle de jeu)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

import chess
import requests
import torch

from .model import build_model
from .search.mcts import MCTS, Evaluator, best_move
from .utils import load_checkpoint, load_config, load_dotenv

API = "https://lichess.org"


def _parse_line(line):
    """Parse une ligne ndjson Lichess. Retourne None pour les keep-alive (lignes
    vides envoyées régulièrement par les streams) et toute ligne non-JSON."""
    if not line:
        return None
    if isinstance(line, bytes):
        line = line.decode("utf-8", "ignore")
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


class SearchEngine:
    """Charge le réseau + MCTS, avec rechargement à chaud des poids."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        dev = cfg.get("train", {}).get("device", "cuda")
        self.device = dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu"
        self.ckpt_path = Path(cfg["paths"]["latest"])
        self._mtime = None
        self.default_nodes = cfg.get("mcts", {}).get("default_nodes", 800)
        # Sérialise rechargement de poids et recherches : un seul GPU, et évite
        # de remplacer self.mcts pendant qu'un thread de partie l'utilise.
        self._lock = threading.Lock()
        self._build()

    def _build(self):
        model = build_model(self.cfg)
        if self.ckpt_path.exists():
            ck = load_checkpoint(self.ckpt_path, self.device)
            model.load_state_dict(ck["model_state"])
            self._mtime = self.ckpt_path.stat().st_mtime
            sys.stderr.write(f"[engine] poids chargés (step {ck.get('step','?')})\n")
        else:
            sys.stderr.write("[engine] AUCUN checkpoint -> réseau aléatoire (jeu faible)\n")
        self.mcts = MCTS(Evaluator(model, self.device), self.cfg)

    def maybe_reload(self):
        # Ne JAMAIS laisser une erreur de rechargement remonter : sinon le thread
        # de la partie meurt et le bot « ne répond plus ». On garde les poids
        # actuels en cas d'échec (checkpoint en cours d'écriture, OOM, etc.).
        if not self.ckpt_path.exists():
            return
        try:
            m = self.ckpt_path.stat().st_mtime
            if m != self._mtime:
                with self._lock:
                    self._build()
        except Exception as e:  # noqa: BLE001 — robustesse volontaire
            sys.stderr.write(f"[engine] rechargement ignoré ({e})\n")

    def choose_move(self, board: chess.Board, think_seconds: float | None):
        nodes = 10_000_000 if think_seconds else self.default_nodes
        with self._lock:
            root = self.mcts.run(board, nodes, max_seconds=think_seconds)
        return best_move(root)


class LichessBot:
    def __init__(self, cfg: dict, token: str):
        self.cfg = cfg
        self.engine = SearchEngine(cfg)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        # Lichess peut être injoignable au démarrage (panne, coupure réseau).
        # On réessaie indéfiniment au lieu de planter : le bot se connectera tout
        # seul dès que Lichess revient, sans intervention.
        self.account = self._get_with_retry("/api/account")
        self.bot_id = self.account["id"]
        bcfg = cfg.get("bot", {})
        self.accept_variants = set(bcfg.get("accept_variants", ["standard"]))
        self.think_divisor = bcfg.get("think_divisor", 40)
        self.max_think = bcfg.get("max_think_seconds", 10.0)
        self.active_games: set[str] = set()
        # Défi automatique d'autres bots (cooldown par adversaire pour ne pas spammer).
        self.challenge_bots = bool(bcfg.get("challenge_bots", False))
        self._recent_challenges: dict[str, float] = {}

    # --- HTTP ---------------------------------------------------------------
    def _get(self, path, **kw):
        r = self.session.get(API + path, timeout=30, **kw)
        r.raise_for_status()
        return r.json()

    def _get_with_retry(self, path, delay: float = 5.0):
        while True:
            try:
                return self._get(path)
            except Exception as e:  # noqa: BLE001 — on attend que Lichess revienne
                print(f"[connexion] Lichess injoignable ({e}); nouvel essai dans {delay:.0f}s")
                time.sleep(delay)

    def _post(self, path, data=None):
        r = self.session.post(API + path, data=data, timeout=30)
        return r

    def _stream(self, path):
        return self.session.get(API + path, stream=True, timeout=None)

    # --- Boucle d'événements ------------------------------------------------
    def run(self):
        title = self.account.get("title")
        print(f"Connecté : {self.account['username']} (title={title})")
        if title != "BOT":
            print("ATTENTION : ce compte n'est pas un BOT. Lance --upgrade d'abord.")
            return
        print("En écoute des défis… (envoie un défi à ce compte pour jouer)")
        if self.challenge_bots:
            threading.Thread(target=self._challenge_loop, daemon=True).start()
        # Reprend les parties déjà en cours (essentiel en correspondance : le bot
        # peut avoir été relancé alors qu'une partie attend son coup).
        try:
            for g in self._get("/api/account/playing").get("nowPlaying", []):
                print(f"[reprise] partie en cours {g['gameId']}")
                self._start_game(g["gameId"])
        except Exception as e:
            print(f"[reprise] échec récupération des parties en cours: {e}")
        while True:
            try:
                resp = self._stream("/api/stream/event")
                for line in resp.iter_lines():
                    event = _parse_line(line)
                    if event is None:
                        continue
                    self._handle_event(event)
            except Exception as e:
                print(f"[event] reconnexion ({e})")
                time.sleep(3)

    def _handle_event(self, event):
        etype = event.get("type")
        if etype == "challenge":
            self._on_challenge(event["challenge"])
        elif etype == "gameStart":
            self._start_game(event["game"]["id"])

    def _start_game(self, gid: str):
        if gid not in self.active_games:
            self.active_games.add(gid)
            threading.Thread(target=self._play_game, args=(gid,), daemon=True).start()

    def _on_challenge(self, ch):
        cid = ch["id"]
        # Défi sortant créé par nous-mêmes (relayé sur le flux d'events) : rien à faire.
        if ch.get("challenger", {}).get("id") == self.bot_id:
            return
        variant = ch.get("variant", {}).get("key", "standard")
        if variant not in self.accept_variants:
            self._post(f"/api/challenge/{cid}/decline")
            print(f"[défi] {cid} décliné (variante {variant})")
            return
        r = self._post(f"/api/challenge/{cid}/accept")
        if r.status_code == 200:
            print(f"[défi] {cid} accepté ({ch.get('challenger',{}).get('name','?')})")
        else:
            print(f"[défi] échec acceptation {cid}: {r.status_code}")

    # --- Défi automatique d'autres bots -------------------------------------
    def _online_bots(self, nb: int = 50) -> list[str]:
        """Identifiants des bots actuellement en ligne (/api/bot/online, ndjson)."""
        ids: list[str] = []
        resp = self._stream(f"/api/bot/online?nb={nb}")
        try:
            for line in resp.iter_lines():
                u = _parse_line(line)
                if u and u.get("id"):
                    ids.append(u["id"])
        finally:
            resp.close()
        return ids

    def _send_challenge(self, username: str, bcfg: dict):
        data = {
            "rated": "true" if bcfg.get("challenge_rated", False) else "false",
            "color": "random",
            "variant": "standard",
        }
        if bcfg.get("challenge_correspondence", False):
            # Correspondance : délai en jours par coup, pas de pendule temps-réel.
            data["days"] = bcfg.get("challenge_days", 3)
            kind = f"correspondance {data['days']}j/coup"
        else:
            data["clock.limit"] = bcfg.get("challenge_clock_limit", 300)
            data["clock.increment"] = bcfg.get("challenge_clock_increment", 3)
            kind = f"{data['clock.limit']}+{data['clock.increment']}"
        rated = "rated" if data["rated"] == "true" else "casual"
        r = self._post(f"/api/challenge/{username}", data=data)
        if r.status_code == 200:
            print(f"[défi-bot] défi envoyé à {username} ({kind}, {rated})")
        else:
            print(f"[défi-bot] échec défi {username}: {r.status_code} {r.text[:120]}")

    def _challenge_loop(self):
        """Défie périodiquement un bot en ligne tant qu'on est sous le quota de
        parties simultanées. Cooldown par adversaire pour ne pas spammer."""
        bcfg = self.cfg.get("bot", {})
        interval = bcfg.get("challenge_interval_sec", 60)
        cooldown = bcfg.get("challenge_cooldown_sec", 600)
        max_games = bcfg.get("max_concurrent_games", 1)
        print(f"[défi-bot] activé : défie un bot en ligne toutes les {interval}s "
              f"si < {max_games} partie(s) en cours.")
        while True:
            time.sleep(interval)
            try:
                if len(self.active_games) >= max_games:
                    continue
                now = time.time()
                candidates = [b for b in self._online_bots()
                              if b != self.bot_id
                              and now - self._recent_challenges.get(b, 0) >= cooldown]
                if not candidates:
                    continue
                opp = random.choice(candidates)
                self._recent_challenges[opp] = now
                self._send_challenge(opp, bcfg)
            except Exception as e:  # noqa: BLE001 — le thread ne doit jamais mourir
                print(f"[défi-bot] erreur: {e}")

    # --- Partie -------------------------------------------------------------
    def _play_game(self, game_id: str):
        self.engine.maybe_reload()        # prend les derniers poids appris
        print(f"[partie {game_id}] démarrée")
        my_color = None
        initial_fen = "startpos"
        game_over = False
        # En correspondance, Lichess ferme les flux inactifs après une longue
        # attente : on se reconnecte tant que la partie n'est pas terminée, sinon
        # le bot « oublie » la partie et ne rejoue jamais.
        while not game_over:
            try:
                resp = self._stream(f"/api/bot/game/stream/{game_id}")
                for line in resp.iter_lines():
                    msg = _parse_line(line)
                    if msg is None:
                        continue
                    t = msg.get("type")
                    if t == "gameFull":
                        initial_fen = msg.get("initialFen", "startpos")
                        my_color = (chess.WHITE if msg["white"].get("id") == self.bot_id
                                    else chess.BLACK)
                        state = msg["state"]
                    elif t == "gameState":
                        state = msg
                    else:
                        continue
                    if state.get("status", "started") != "started":
                        print(f"[partie {game_id}] terminée ({state.get('status')})")
                        game_over = True
                        break
                    self._maybe_move(game_id, initial_fen, state, my_color)
            except Exception as e:
                print(f"[partie {game_id}] erreur stream: {e}")
            if not game_over:
                print(f"[partie {game_id}] flux interrompu, reconnexion…")
                time.sleep(2)
        self.active_games.discard(game_id)

    def _build_board(self, initial_fen: str, moves: str) -> chess.Board:
        board = chess.Board() if initial_fen == "startpos" else chess.Board(initial_fen)
        for mv in moves.split():
            board.push_uci(mv)
        return board

    def _think_seconds(self, state, my_color) -> float | None:
        key = "wtime" if my_color == chess.WHITE else "btime"
        inc_key = "winc" if my_color == chess.WHITE else "binc"
        remaining = state.get(key)
        if remaining is None:                      # correspondance / sans pendule
            return None
        inc = state.get(inc_key, 0) / 1000.0
        t = (remaining / 1000.0) / self.think_divisor + 0.8 * inc
        return max(0.1, min(t, self.max_think))

    def _maybe_move(self, game_id, initial_fen, state, my_color):
        board = self._build_board(initial_fen, state.get("moves", ""))
        if board.turn != my_color or board.is_game_over():
            return
        move = self.engine.choose_move(board, self._think_seconds(state, my_color))
        if move is None:
            return
        r = self._post(f"/api/bot/game/{game_id}/move/{move.uci()}")
        if r.status_code != 200:
            print(f"[partie {game_id}] coup refusé {move.uci()}: {r.status_code} {r.text}")


# --- Commandes ---------------------------------------------------------------
def acquire_single_instance_lock():
    """Empêche deux bots de tourner avec le même token (ils se voleraient les
    events Lichess et leurs coups seraient refusés -> « le bot ne répond plus »).
    Retourne le descripteur du verrou (à garder ouvert toute la vie du process)."""
    lock_path = Path(tempfile.gettempdir()) / "sanchess_lichess_bot.lock"
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"Un bot tourne déjà (verrou {lock_path}). "
                 f"Arrête-le d'abord : pkill -f sanchess.lichess_bot")
    fd.write(str(os.getpid()))
    fd.flush()
    return fd


def get_token() -> str:
    load_dotenv()
    token = os.environ.get("LICHESS_TOKEN")
    if not token:
        sys.exit("LICHESS_TOKEN absent (.env ou variable d'env). "
                 "Crée un token avec le scope bot:play sur lichess.org/account/oauth/token")
    return token


def cmd_check(token):
    r = requests.get(API + "/api/account",
                     headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    acc = r.json()
    title = acc.get("title")
    nb_games = acc.get("count", {}).get("all", 0)
    print(f"Compte    : {acc['username']}")
    print(f"Titre     : {title or '(aucun)'}")
    print(f"Parties   : {nb_games}")
    if title == "BOT":
        print("=> Déjà un compte BOT : prêt à lancer `python -m sanchess.lichess_bot`.")
    elif nb_games == 0:
        print("=> Compte vierge : éligible à --upgrade (conversion en BOT, IRRÉVERSIBLE).")
    else:
        print("=> NON éligible : ce compte a déjà joué. Crée un NOUVEAU compte pour le bot.")


def cmd_upgrade(token):
    r = requests.post(API + "/api/bot/account/upgrade",
                      headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code == 200:
        print("OK : compte converti en BOT.")
    else:
        print(f"Échec upgrade : {r.status_code} {r.text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="afficher l'état du compte (lecture seule)")
    ap.add_argument("--upgrade", action="store_true", help="convertir en compte BOT (IRRÉVERSIBLE)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    token = get_token()
    if args.check:
        cmd_check(token); return
    if args.upgrade:
        cmd_upgrade(token); return

    lock = acquire_single_instance_lock()  # garde le fd ouvert tant que le bot vit
    cfg = load_config(args.config)
    LichessBot(cfg, token).run()
    del lock


if __name__ == "__main__":
    main()
