"""Encodage échiquier <-> tenseurs et coups <-> index de politique (style AlphaZero).

Tout est "canonicalisé" du point de vue du joueur au trait : si c'est aux Noirs
de jouer, l'échiquier est miroité verticalement (et les couleurs échangées) de
sorte que le camp au trait avance toujours vers le haut. Le réseau voit donc
toujours la même orientation.

Politique : 4672 = 64 cases x 73 types de coups.
  - planes 0..55  : coups "dame" = 8 directions x 7 distances (R/B/Q/K/poussées de pion)
  - planes 56..63 : 8 coups de cavalier
  - planes 64..72 : 9 sous-promotions = 3 directions x 3 pièces (C/F/T)
Les promotions Dame sont encodées comme un coup "dame" normal (distance 1).
"""

from __future__ import annotations

import chess
import numpy as np

# --- Dimensions ---------------------------------------------------------------
INPUT_PLANES = 19          # voir encode_board(features="base")
TACTICAL_INPUT_PLANES = 21 # base + cartes d'attaques nous/adversaire
POLICY_SIZE = 4672         # 64 * 73
PLANES_PER_SQUARE = 73

# --- Tables de directions (orientation canonique : le trait avance en +rank) --
# (delta_rank, delta_file) pour les 8 directions "dame".
_QUEEN_DIRS = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]
_QUEEN_DIR_INDEX = {d: i for i, d in enumerate(_QUEEN_DIRS)}

_KNIGHT_DELTAS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]
_KNIGHT_INDEX = {d: i for i, d in enumerate(_KNIGHT_DELTAS)}

# Sous-promotions : directions de fichier (df) et pièces.
_PROMO_DF = [0, -1, 1]                                   # tout droit, capture gauche, capture droite
_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_PROMO_DF_INDEX = {df: i for i, df in enumerate(_PROMO_DF)}
_PROMO_PIECE_INDEX = {p: i for i, p in enumerate(_PROMO_PIECES)}

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _canon_square(square: int, turn: bool) -> int:
    """Square dans le repère canonique (miroir vertical si trait aux Noirs)."""
    return square if turn == chess.WHITE else square ^ 56


def move_to_index(move: chess.Move, turn: bool) -> int:
    """Convertit un coup (dans le repère réel) en index de politique [0, 4672)."""
    from_sq = _canon_square(move.from_square, turn)
    to_sq = _canon_square(move.to_square, turn)

    fr, ff = divmod(from_sq, 8)
    tr, tf = divmod(to_sq, 8)
    dr, df = tr - fr, tf - ff

    # Sous-promotion (C/F/T) : plane dédié.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        dir_idx = _PROMO_DF_INDEX[df]
        piece_idx = _PROMO_PIECE_INDEX[move.promotion]
        plane = 64 + dir_idx * 3 + piece_idx
        return from_sq * PLANES_PER_SQUARE + plane

    # Coup de cavalier.
    if (dr, df) in _KNIGHT_INDEX:
        plane = 56 + _KNIGHT_INDEX[(dr, df)]
        return from_sq * PLANES_PER_SQUARE + plane

    # Coup "dame" (slides + poussées + promotion Dame).
    direction = (_sign(dr), _sign(df))
    dist = max(abs(dr), abs(df))
    dir_idx = _QUEEN_DIR_INDEX[direction]
    plane = dir_idx * 7 + (dist - 1)
    return from_sq * PLANES_PER_SQUARE + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Reconstruit un coup à partir d'un index, en s'appuyant sur le board pour
    lever les ambiguïtés (notamment la promotion Dame)."""
    turn = board.turn
    from_sq_c, plane = divmod(index, PLANES_PER_SQUARE)
    fr, ff = divmod(from_sq_c, 8)
    promotion = None

    if plane < 56:                       # coup "dame"
        dir_idx, dist_m1 = divmod(plane, 7)
        dr_u, df_u = _QUEEN_DIRS[dir_idx]
        dist = dist_m1 + 1
        tr, tf = fr + dr_u * dist, ff + df_u * dist
    elif plane < 64:                     # cavalier
        dr_u, df_u = _KNIGHT_DELTAS[plane - 56]
        tr, tf = fr + dr_u, ff + df_u
    else:                                # sous-promotion
        p = plane - 64
        dir_idx, piece_idx = divmod(p, 3)
        df = _PROMO_DF[dir_idx]
        tr, tf = fr + 1, ff + df
        promotion = _PROMO_PIECES[piece_idx]

    to_sq_c = tr * 8 + tf
    from_sq = _canon_square(from_sq_c, turn)
    to_sq = _canon_square(to_sq_c, turn)

    # Promotion Dame implicite : pion atteignant la dernière rangée via un plane "dame".
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            last_rank = 7 if turn == chess.WHITE else 0
            if chess.square_rank(to_sq) == last_rank:
                promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_move_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    """Retourne (liste des coups légaux, indices de politique correspondants)."""
    moves = list(board.legal_moves)
    idx = np.fromiter((move_to_index(m, board.turn) for m in moves),
                      dtype=np.int64, count=len(moves))
    return moves, idx


def legal_policy_mask(board: chess.Board) -> np.ndarray:
    """Masque dense (POLICY_SIZE,) : True uniquement pour les coups légaux."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    _, idx = legal_move_indices(board)
    mask[idx] = True
    return mask


def input_planes_for_features(features: str = "base") -> int:
    """Nombre de plans d'entrée pour un jeu de features."""
    if str(features).lower() == "tactical":
        return TACTICAL_INPUT_PLANES
    return INPUT_PLANES


def encode_board(board: chess.Board, features: str = "base") -> np.ndarray:
    """Encode la position en (C, 8, 8) float32, repère canonique.

    `features="base"` garde les 19 plans historiques. `features="tactical"`
    ajoute deux cartes d'attaque (cases attaquées par le trait / l'adversaire),
    dérivées de la FEN à la volée : aucun changement du format de samples.
    """
    turn = board.turn
    feature_mode = str(features).lower()
    planes = np.zeros((input_planes_for_features(feature_mode), 8, 8),
                      dtype=np.float32)

    for square, piece in board.piece_map().items():
        csq = _canon_square(square, turn)
        r, f = divmod(csq, 8)
        # python-chess : PAWN..KING == 1..6 -> index 0..5 sans recherche linéaire
        # (l'ancien _PIECE_TYPES.index() scannait la liste pour chaque pièce de
        # chaque position : pur surcoût Python sur le hot-path du self-play batché).
        ptype_idx = piece.piece_type - 1
        ours = piece.color == turn
        plane = ptype_idx + (0 if ours else 6)
        planes[plane, r, f] = 1.0

    # Plan 12 : couleur réelle du trait (1 = Blancs).
    planes[12, :, :] = 1.0 if turn == chess.WHITE else 0.0

    # Plans 13..16 : droits de roque (nous puis adversaire).
    planes[13, :, :] = float(board.has_kingside_castling_rights(turn))
    planes[14, :, :] = float(board.has_queenside_castling_rights(turn))
    planes[15, :, :] = float(board.has_kingside_castling_rights(not turn))
    planes[16, :, :] = float(board.has_queenside_castling_rights(not turn))

    # Plan 17 : compteur des 50 coups (normalisé).
    planes[17, :, :] = board.halfmove_clock / 100.0

    # Plan 18 : case en passant.
    if board.ep_square is not None:
        csq = _canon_square(board.ep_square, turn)
        r, f = divmod(csq, 8)
        planes[18, r, f] = 1.0

    if feature_mode == "tactical":
        for sq in chess.SQUARES:
            csq = _canon_square(sq, turn)
            r, f = divmod(csq, 8)
            if board.is_attacked_by(turn, sq):
                planes[19, r, f] = 1.0
            if board.is_attacked_by(not turn, sq):
                planes[20, r, f] = 1.0

    return planes
