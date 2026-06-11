"""Pertes et cibles AlphaZero, partagées par pretrain / online / distributed.

Politique : cross-entropy SOUPLE contre une distribution de probabilités (la
distribution de visites MCTS pour le self-play, ou un one-hot pour les samples
humains à un seul coup). La CE souple avec un one-hot redonne EXACTEMENT la CE
dure habituelle -> le pré-entraînement supervisé est numériquement inchangé.

Valeur : deux têtes possibles (cf. `model.value_head`) —
  - "scalar" : sortie tanh ∈ [-1,1], perte MSE contre z ∈ {-1,0,1} (legacy) ;
  - "wdl"    : 3 logits (défaite/nulle/victoire), perte cross-entropy ; le scalaire
    rendu au MCTS est l'espérance P(victoire) - P(défaite) ∈ [-1,1].
La cible WDL se déduit du même z ∈ {-1,0,1} : aucun changement du format de samples
n'est requis côté valeur.
"""

from __future__ import annotations

import chess
import numpy as np
import torch
import torch.nn.functional as F

from ..encoding import POLICY_SIZE, move_to_index


# --- Construction des cibles politiques (côté données) ------------------------

def dense_policy_target(board, move_uci: str, pi: dict | None) -> np.ndarray:
    """Distribution cible (POLICY_SIZE,) sommant à 1.

    `pi` = {coup_uci: poids} (visites MCTS) si disponible (self-play), sinon None
    -> one-hot sur le coup joué (samples humains). Les coups illégaux/non encodables
    sont simplement ignorés.
    """
    target = np.zeros(POLICY_SIZE, dtype=np.float32)
    turn = board.turn
    if pi:
        for uci, w in pi.items():
            try:
                idx = move_to_index(chess.Move.from_uci(uci), turn)
            except (ValueError, KeyError):
                continue
            target[idx] += float(w)
        s = target.sum()
        if s > 0:
            target /= s
            return target
        # pi vide/invalide -> repli one-hot
    target[move_to_index(chess.Move.from_uci(move_uci), turn)] = 1.0
    return target


# --- Pertes (côté modèle) -----------------------------------------------------

def policy_loss(logits: torch.Tensor, target: torch.Tensor,
                label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy souple. `target` : (B, POLICY_SIZE) sommant à 1."""
    if label_smoothing and label_smoothing > 0:
        k = target.shape[1]
        target = target * (1.0 - label_smoothing) + label_smoothing / k
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _wdl_class(z: torch.Tensor) -> torch.Tensor:
    """z ∈ {-1,0,1} -> classe {0:défaite, 1:nulle, 2:victoire}."""
    return torch.round(z).long() + 1


def value_loss(value_out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """MSE (tête scalaire (B,1)) ou cross-entropy (tête WDL (B,3))."""
    if value_out.dim() == 2 and value_out.shape[1] == 3:
        return F.cross_entropy(value_out, _wdl_class(z))
    return F.mse_loss(value_out.squeeze(1), z)


def value_to_scalar(value_out: torch.Tensor) -> torch.Tensor:
    """Scalaire ∈ [-1,1] pour le MCTS : tanh tel quel, ou E[résultat] = P(V)-P(D)."""
    if value_out.dim() == 2 and value_out.shape[1] == 3:
        p = F.softmax(value_out, dim=1)
        return p[:, 2] - p[:, 0]
    return value_out.reshape(-1)
