"""Bot Lichess natif pour San-o1.

Pilote directement le moteur MCTS (pas de pont UCI) via l'API Bot de Lichess :
  - écoute les événements (/api/stream/event)
  - accepte les défis (échecs standard)
  - joue chaque partie en streamant son état et en postant ses coups
  - salue l'adversaire et répond aux commandes de chat :
      !help  : liste des commandes
      !eval  : évaluation et coup choisi (gain estimé)
      !pv    : variante principale prévue
      !top   : meilleurs coups avec leur nombre de visites
      !info  : moteur, pas d'entraînement, nœuds/coup, appareil
      !rating: classements Elo du bot
      !fen   : FEN de la position courante
      !name / !source / !ping / !gg / hi
  - accepte une nulle proposée si la position n'est pas en sa faveur, et peut
    abandonner les positions désespérées (cf. bloc `bot` du config.yaml)

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
from dataclasses import dataclass, field
from pathlib import Path

import chess
import requests
import torch

from .model import build_model, build_model_from_checkpoint
from .search.mcts import MCTS, Evaluator, best_move
from .utils import load_checkpoint, load_config, load_dotenv, load_model_state, resolve_device

API = "https://lichess.org"


@dataclass
class MoveAnalysis:
    """Résumé lisible d'une recherche MCTS, pour jouer ET pour le chat."""
    move: "chess.Move | None"
    move_san: str
    score: float          # valeur [-1,1] du point de vue du bot (>0 = bot mieux)
    win: float            # probabilité de gain estimée [0,1]
    visits: int           # visites à la racine (taille de l'arbre)
    pv_san: str           # variante principale en notation SAN
    top: list             # [(san, visites, valeur_bot), ...] triés par visites
    step: "int | None"    # pas d'entraînement du checkpoint courant

    def eval_str(self) -> str:
        s = f"+{self.score:.2f}" if self.score >= 0 else f"{self.score:.2f}"
        return f"{s} (gain ~{self.win * 100:.0f}%)"


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
        # Résoudre "auto"/"cuda"/"mps" en un id Torch concret : passer "auto"
        # tel quel à torch.load (map_location) plante car PyTorch ne sait pas
        # restaurer vers un appareil nommé "auto".
        self.device = resolve_device(cfg.get("train", {}).get("device", "auto"))
        self.ckpt_path = Path(cfg["paths"]["latest"])
        self._mtime = None
        self.default_nodes = cfg.get("mcts", {}).get("default_nodes", 800)
        # Sérialise rechargement de poids et recherches : un seul GPU, et évite
        # de remplacer self.mcts pendant qu'un thread de partie l'utilise.
        self._lock = threading.Lock()
        self._build()

    def _build(self):
        if self.ckpt_path.exists():
            ck = load_checkpoint(self.ckpt_path, self.device)
            # Construire le réseau d'après l'ARCHI enregistrée dans le checkpoint
            # (et non la config courante) : un latest.pt wdl se charge même si
            # config.yaml dit `scalar`, et inversement -> plus de crash de forme
            # quand config et checkpoint divergent. `load_model_state` reste
            # tolérant (poids partiels) par-dessus.
            model = build_model_from_checkpoint(ck, fallback_cfg=self.cfg)
            load_model_state(model, ck["model_state"])
            self._mtime = self.ckpt_path.stat().st_mtime
            self.step = ck.get("step")
            sys.stderr.write(f"[engine] poids chargés (step {ck.get('step','?')})\n")
        else:
            model = build_model(self.cfg)
            self.step = None
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

    def analyze(self, board: chess.Board, think_seconds: float | None) -> MoveAnalysis:
        """Lance la recherche et renvoie un résumé complet (coup + éval + PV)."""
        nodes = 10_000_000 if think_seconds else self.default_nodes
        with self._lock:
            root = self.mcts.run(board, nodes, max_seconds=think_seconds)
        return self._summarize(root, board)

    def choose_move(self, board: chess.Board, think_seconds: float | None):
        return self.analyze(board, think_seconds).move

    # --- Mise en forme de l'arbre de recherche -------------------------------
    @staticmethod
    def _principal_variation(root, board: chess.Board, max_len: int = 10) -> str:
        node, b, sans = root, board.copy(), []
        while node.children and len(sans) < max_len:
            mv, child = max(node.children.items(), key=lambda kv: kv[1].N)
            if child.N == 0:
                break
            sans.append(b.san(mv))
            b.push(mv)
            node = child
        return " ".join(sans)

    def _summarize(self, root, board: chess.Board) -> MoveAnalysis:
        mv = best_move(root)
        if mv is None:                              # position terminale
            return MoveAnalysis(None, "—", 0.0, 0.5, root.N, "", [], self.step)
        score = root.q() if root.N else 0.0         # point de vue du trait = le bot
        score = max(-1.0, min(1.0, score))
        children = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
        top = [(board.san(m), c.N, (-c.q() if c.N else 0.0)) for m, c in children[:5]]
        return MoveAnalysis(
            move=mv,
            move_san=board.san(mv),
            score=score,
            win=(score + 1) / 2,
            visits=root.N,
            pv_san=self._principal_variation(root, board),
            top=top,
            step=self.step,
        )


@dataclass
class _Game:
    """État vivant d'une partie en cours (board, couleur, dernière analyse…)."""
    id: str
    my_color: "bool | None" = None
    initial_fen: str = "startpos"
    board: chess.Board = field(default_factory=chess.Board)
    last: "MoveAnalysis | None" = None     # dernière analyse jouée par le bot
    opponent: str = "?"
    greeted: bool = False
    bad_streak: int = 0                    # coups d'affilée jugés perdus (abandon)
    moves_played: int = 0                  # coups joués par le bot dans la partie


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
        self.username = self.account.get("username", self.bot_id)
        bcfg = cfg.get("bot", {})
        self.accept_variants = set(bcfg.get("accept_variants", ["standard"]))
        self.think_divisor = bcfg.get("think_divisor", 40)
        self.max_think = bcfg.get("max_think_seconds", 10.0)
        self.games: dict[str, _Game] = {}
        # Défi automatique d'autres bots (cooldown par adversaire pour ne pas spammer).
        self.challenge_bots = bool(bcfg.get("challenge_bots", False))
        self.challenge_modes = self._build_challenge_modes(bcfg)
        self._recent_challenges: dict[str, float] = {}
        # --- Chat & courtoisie -------------------------------------------------
        self.chat_enabled = bool(bcfg.get("chat_enabled", True))
        self.greeting = bcfg.get(
            "greeting",
            "Bonjour {opp} ! Je suis un réseau de neurones maison. "
            "Bonne partie — tape !help pour les commandes.")
        self.goodbye = bcfg.get("goodbye", "gg, merci pour la partie ! {opp}")
        # --- Abandon / nulle automatiques -------------------------------------
        self.resign_enabled = bool(bcfg.get("resign_enabled", False))
        self.resign_score = float(bcfg.get("resign_score", -0.97))
        self.resign_streak = int(bcfg.get("resign_streak", 4))
        self.resign_min_move = int(bcfg.get("resign_min_move", 12))
        self.accept_draw = bool(bcfg.get("accept_draw", True))
        self.accept_draw_score = float(bcfg.get("accept_draw_score", -0.50))

    @property
    def active_games(self) -> set[str]:
        return set(self.games)

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

    # --- Actions de partie ---------------------------------------------------
    def _chat(self, game_id: str, text: str, room: str = "player"):
        """Envoie un message dans le chat de la partie (jamais fatal)."""
        if not self.chat_enabled or not text:
            return
        try:
            # Lichess plafonne les messages à 140 caractères.
            self._post(f"/api/bot/game/{game_id}/chat",
                       data={"room": room, "text": text[:140]})
        except Exception as e:  # noqa: BLE001
            print(f"[chat {game_id}] échec envoi: {e}")

    def _resign(self, game_id: str):
        try:
            self._post(f"/api/bot/game/{game_id}/resign")
        except Exception as e:  # noqa: BLE001
            print(f"[partie {game_id}] échec abandon: {e}")

    def _answer_draw(self, game_id: str, accept: bool):
        try:
            self._post(f"/api/bot/game/{game_id}/draw/{'yes' if accept else 'no'}")
        except Exception as e:  # noqa: BLE001
            print(f"[partie {game_id}] échec réponse nulle: {e}")

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
        if gid not in self.games:
            self.games[gid] = _Game(gid)
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

    def _build_challenge_modes(self, bcfg: dict) -> list[dict]:
        """Liste des cadences proposées aux autres bots (tirage aléatoire par défi).

        Chaque mode est un dict :
          - temps-réel : {clock_limit: secondes, clock_increment: secondes}
          - correspondance : {days: jours_par_coup}
          - `rated` optionnel par mode (sinon valeur globale `challenge_rated`).
        Si `bot.challenge_modes` est absent du config, on retombe sur l'ancien
        réglage unique (`challenge_correspondence`/`challenge_clock_*`)."""
        modes = bcfg.get("challenge_modes")
        if modes:
            return [m for m in modes if isinstance(m, dict)]
        if bcfg.get("challenge_correspondence", False):
            return [{"days": bcfg.get("challenge_days", 3)}]
        return [{"clock_limit": bcfg.get("challenge_clock_limit", 300),
                 "clock_increment": bcfg.get("challenge_clock_increment", 3)}]

    def _send_challenge(self, username: str, mode: dict):
        bcfg = self.cfg.get("bot", {})
        rated = bool(mode.get("rated", bcfg.get("challenge_rated", False)))
        data = {
            "rated": "true" if rated else "false",
            "color": "random",
            "variant": "standard",
        }
        if mode.get("days"):
            # Correspondance : délai en jours par coup, pas de pendule temps-réel.
            data["days"] = mode["days"]
            kind = f"correspondance {mode['days']}j/coup"
        else:
            lim = int(mode.get("clock_limit", 600))
            inc = int(mode.get("clock_increment", 5))
            data["clock.limit"] = lim
            data["clock.increment"] = inc
            # Affichage type Lichess : minutes+incrément (300+3 -> "5+3").
            mins = lim / 60
            mins = int(mins) if mins == int(mins) else round(mins, 1)
            kind = f"{mins}+{inc}"
        r = self._post(f"/api/challenge/{username}", data=data)
        label = "rated" if rated else "casual"
        if r.status_code == 200:
            print(f"[défi-bot] défi envoyé à {username} ({kind}, {label})")
        else:
            print(f"[défi-bot] échec défi {username}: {r.status_code} {r.text[:120]}")

    def _challenge_loop(self):
        """Défie périodiquement un bot en ligne tant qu'on est sous le quota de
        parties simultanées. Cooldown par adversaire pour ne pas spammer. La
        cadence est tirée au hasard parmi `self.challenge_modes` à chaque défi."""
        bcfg = self.cfg.get("bot", {})
        interval = bcfg.get("challenge_interval_sec", 60)
        cooldown = bcfg.get("challenge_cooldown_sec", 600)
        max_games = bcfg.get("max_concurrent_games", 1)
        print(f"[défi-bot] activé : défie un bot en ligne toutes les {interval}s "
              f"si < {max_games} partie(s) en cours. {len(self.challenge_modes)} cadence(s).")
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
                self._send_challenge(opp, random.choice(self.challenge_modes))
            except Exception as e:  # noqa: BLE001 — le thread ne doit jamais mourir
                print(f"[défi-bot] erreur: {e}")

    # --- Partie -------------------------------------------------------------
    def _play_game(self, game_id: str):
        self.engine.maybe_reload()        # prend les derniers poids appris
        print(f"[partie {game_id}] démarrée")
        g = self.games.get(game_id) or _Game(game_id)
        self.games[game_id] = g
        game_over = False
        # En correspondance, Lichess ferme les flux inactifs après une longue
        # attente : on se reconnecte tant que la partie n'est pas terminée, sinon
        # le bot « oublie » la partie et ne rejoue jamais.
        backoff = 2.0          # délai de reconnexion (croît si Lichess refuse)
        consec_empty = 0       # reconnexions sans aucune donnée reçue
        while not game_over:
            got_data = False
            try:
                resp = self._stream(f"/api/bot/game/stream/{game_id}")
                for line in resp.iter_lines():
                    msg = _parse_line(line)
                    if msg is None:
                        continue
                    got_data = True
                    t = msg.get("type")
                    if t == "chatLine":
                        self._on_chat(g, msg)
                        continue
                    if t == "gameFull":
                        g.initial_fen = msg.get("initialFen", "startpos")
                        g.my_color = (chess.WHITE if msg["white"].get("id") == self.bot_id
                                      else chess.BLACK)
                        opp = msg["black"] if g.my_color == chess.WHITE else msg["white"]
                        g.opponent = opp.get("name") or opp.get("id") or "?"
                        self._greet(g)
                        state = msg["state"]
                    elif t == "gameState":
                        state = msg
                    else:
                        continue
                    if state.get("status", "started") != "started":
                        print(f"[partie {game_id}] terminée ({state.get('status')})")
                        self._farewell(g, state.get("status"))
                        game_over = True
                        break
                    self._maybe_move(g, state)
            except Exception as e:
                print(f"[partie {game_id}] erreur stream: {e}")
            if game_over:
                break
            # Le flux s'est interrompu sans état terminal. Avant de boucler à
            # l'infini, on demande à Lichess si la partie est encore en cours :
            # une partie finie/avortée n'apparaît plus dans /api/account/playing
            # et Lichess ferme alors son flux « without response ».
            if not self._game_in_progress(game_id):
                print(f"[partie {game_id}] absente des parties en cours → suivi abandonné")
                break
            if got_data:
                backoff, consec_empty = 2.0, 0
            else:
                consec_empty += 1
                if consec_empty >= 5:
                    print(f"[partie {game_id}] flux injoignable ({consec_empty} essais vides) "
                          f"→ suivi abandonné")
                    break
            print(f"[partie {game_id}] flux interrompu, reconnexion dans {backoff:.0f}s…")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
        self.games.pop(game_id, None)

    def _game_in_progress(self, game_id: str) -> bool:
        """Vrai si Lichess liste encore la partie dans /api/account/playing.
        En cas d'erreur réseau on renvoie True : on ne veut pas abandonner une
        vraie partie en cours à cause d'un simple hoquet de connexion."""
        try:
            playing = self._get("/api/account/playing").get("nowPlaying", [])
            return any(p.get("gameId") == game_id for p in playing)
        except Exception:
            return True

    def _greet(self, g: _Game):
        if g.greeted or not self.chat_enabled:
            return
        g.greeted = True
        self._chat(g.id, self.greeting.format(opp=g.opponent, me=self.username))

    def _farewell(self, g: _Game, status: str | None):
        # Pas de politesse sur une partie avortée (personne n'a joué).
        if status == "aborted" or not self.chat_enabled:
            return
        self._chat(g.id, self.goodbye.format(opp=g.opponent, me=self.username))

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

    def _maybe_move(self, g: _Game, state):
        board = self._build_board(g.initial_fen, state.get("moves", ""))
        g.board = board                              # pour les commandes de chat (!fen…)
        if board.turn != g.my_color or board.is_game_over():
            return
        self.engine.maybe_reload()                   # derniers poids appris à chaud
        analysis = self.engine.analyze(board, self._think_seconds(state, g.my_color))
        g.last = analysis
        move = analysis.move
        if move is None:
            return

        # Accepter une nulle proposée par l'adversaire si on n'est pas mieux.
        if self.accept_draw and self._opponent_offers_draw(state, g.my_color):
            if analysis.score <= self.accept_draw_score:
                print(f"[partie {g.id}] nulle acceptée (éval {analysis.score:+.2f})")
                self._chat(g.id, "D'accord pour la nulle.")
                self._answer_draw(g.id, accept=True)
                return

        # Abandon si la position est jugée désespérée depuis plusieurs coups.
        g.bad_streak = g.bad_streak + 1 if analysis.score < self.resign_score else 0
        if (self.resign_enabled and g.bad_streak >= self.resign_streak
                and board.fullmove_number >= self.resign_min_move):
            print(f"[partie {g.id}] abandon (éval {analysis.score:+.2f})")
            self._chat(g.id, "Position perdue, j'abandonne. gg !")
            self._resign(g.id)
            return

        r = self._post(f"/api/bot/game/{g.id}/move/{move.uci()}")
        if r.status_code != 200:
            print(f"[partie {g.id}] coup refusé {move.uci()}: {r.status_code} {r.text}")
        else:
            g.moves_played += 1

    @staticmethod
    def _opponent_offers_draw(state, my_color) -> bool:
        # Lichess pose wdraw/bdraw=True quand le camp correspondant propose nulle.
        return bool(state.get("bdraw") if my_color == chess.WHITE
                    else state.get("wdraw"))

    # --- Chat de partie ------------------------------------------------------
    def _on_chat(self, g: _Game, msg):
        if not self.chat_enabled:
            return
        user = msg.get("username", "")
        room = msg.get("room", "player")
        text = (msg.get("text") or "").strip()
        # Ignorer nos propres messages et ceux du système.
        if not text or user in (self.username, self.bot_id, "lichess"):
            return
        cmd = text.lower().lstrip("!/").split()[0] if text else ""
        handler = self._commands().get(cmd)
        if handler is None:
            # Salutation amicale même sans commande explicite.
            if cmd in ("hi", "hello", "hey", "bonjour", "salut", "yo"):
                handler = self._cmd_hi
            else:
                return
        try:
            reply = handler(g, user)
        except Exception as e:  # noqa: BLE001 — une commande ne doit jamais crasher la partie
            print(f"[chat {g.id}] erreur commande '{cmd}': {e}")
            return
        if reply:
            self._chat(g.id, reply, room=room)

    def _commands(self) -> dict:
        return {
            "help": self._cmd_help, "commands": self._cmd_help, "h": self._cmd_help,
            "eval": self._cmd_eval, "score": self._cmd_eval, "ev": self._cmd_eval,
            "pv": self._cmd_pv, "line": self._cmd_pv,
            "top": self._cmd_top, "moves": self._cmd_top,
            "info": self._cmd_info, "engine": self._cmd_info,
            "name": self._cmd_name, "who": self._cmd_name,
            "rating": self._cmd_rating, "elo": self._cmd_rating,
            "fen": self._cmd_fen,
            "ping": lambda g, u: "pong",
            "gg": self._cmd_gg,
            "hi": self._cmd_hi, "hello": self._cmd_hi, "hey": self._cmd_hi,
            "source": self._cmd_source, "github": self._cmd_source, "code": self._cmd_source,
        }

    HELP_TEXT = ("Commandes : !eval !pv !top !info !rating !fen !name !source !ping !gg")

    def _cmd_help(self, g, user):
        return self.HELP_TEXT

    def _cmd_eval(self, g, user):
        if g.last is None:
            return "Pas encore réfléchi à un coup."
        a = g.last
        return f"Éval {a.eval_str()} — je jouais {a.move_san} ({a.visits} nœuds)"

    def _cmd_pv(self, g, user):
        if g.last is None or not g.last.pv_san:
            return "Pas encore de variante."
        return f"Variante prévue : {g.last.pv_san}"

    def _cmd_top(self, g, user):
        if g.last is None or not g.last.top:
            return "Pas encore de coups analysés."
        parts = [f"{san} ({n})" for san, n, _ in g.last.top]
        return "Top coups : " + ", ".join(parts)

    def _cmd_info(self, g, user):
        step = self.engine.step
        net = f"step {step}" if step is not None else "réseau aléatoire (non entraîné)"
        return (f"San-o1 — moteur neuronal maison (ResNet + MCTS), {net}, "
                f"{self.engine.default_nodes} nœuds/coup sur {self.engine.device}.")

    def _cmd_name(self, g, user):
        return f"Je suis {self.username}, un bot d'échecs à réseau de neurones (San-o1)."

    def _cmd_rating(self, g, user):
        perfs = self.account.get("perfs", {})
        wanted = ["bullet", "blitz", "rapid", "classical", "correspondence"]
        bits = []
        for p in wanted:
            d = perfs.get(p)
            if d and d.get("games"):
                prov = "?" if d.get("prov") else ""
                bits.append(f"{p} {d.get('rating','?')}{prov}")
        return "Mes Elo : " + ", ".join(bits) if bits else "Pas encore de classement."

    def _cmd_fen(self, g, user):
        return g.board.fen()

    def _cmd_gg(self, g, user):
        return "gg ! Bien joué 🙂"

    def _cmd_hi(self, g, user):
        return f"Salut {user} ! Bonne partie. (!help pour les commandes)"

    def _cmd_source(self, g, user):
        return "Moteur San-o1 — réseau de neurones + MCTS, entièrement maison."


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
