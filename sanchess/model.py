"""Réseau résiduel type AlphaZero/Lc0 : corps convolutif (+ Squeeze-Excitation
optionnel) + tête politique + tête valeur.

Améliorations vs. réseau de base :
  - blocs résiduels avec **Squeeze-and-Excitation** (recalibrage par canal,
    gain de force notable à coût quasi nul — cf. Leela Chess Zero) ;
  - tête valeur à largeur configurable ;
  - initialisation Kaiming explicite des convolutions/linéaires.

L'architecture reste pilotée par `config.yaml` (section `model`) et reste
compatible avec les anciens checkpoints via `utils.load_model_state` (chargement
tolérant).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import INPUT_PLANES, POLICY_SIZE


class SqueezeExcitation(nn.Module):
    """Recalibrage par canal : pool global -> MLP -> portes sigmoïdes."""

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))                 # (B, C) pooling spatial global
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se: bool = False, se_ratio: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio) if se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return F.relu(out + x)


class SanChessNet(nn.Module):
    """Entrée (B, INPUT_PLANES, 8, 8) -> (policy_logits (B, 4672), value (B, 1))."""

    def __init__(self, channels: int = 128, blocks: int = 10,
                 se: bool = False, se_ratio: int = 4, value_hidden: int = 256):
        super().__init__()
        self.cfg_meta = {"channels": channels, "blocks": blocks,
                         "se": se, "se_ratio": se_ratio,
                         "value_hidden": value_hidden}
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(channels, se=se, se_ratio=se_ratio) for _ in range(blocks)]
        )

        # Tête politique
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_SIZE)

        # Tête valeur
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_conv(x).flatten(1)
        policy_logits = self.policy_fc(p)

        v = self.value_conv(x).flatten(1)
        value = torch.tanh(self.value_fc(v))   # [-1, 1] du point de vue du trait

        return policy_logits, value


def _model_kwargs(m: dict) -> dict:
    return {
        "channels": m.get("channels", 128),
        "blocks": m.get("blocks", 10),
        "se": bool(m.get("se", False)),
        "se_ratio": int(m.get("se_ratio", 4)),
        "value_hidden": int(m.get("value_hidden", 256)),
    }


def build_model(cfg: dict) -> SanChessNet:
    """Construit le réseau d'après la section `model` de la config."""
    return SanChessNet(**_model_kwargs(cfg.get("model", {})))


def build_model_from_checkpoint(ckpt: dict, fallback_cfg: dict | None = None) -> SanChessNet:
    """Construit le réseau d'après l'archi enregistrée dans le checkpoint.

    Évite les divergences trainer/moteur lors du hot-reload : on respecte la
    forme avec laquelle les poids ont été entraînés plutôt que le `config.yaml`
    courant. Repli sur la config fournie si le checkpoint n'embarque pas d'archi.
    """
    mcfg = ckpt.get("model_cfg") or (fallback_cfg or {}).get("model", {})
    return SanChessNet(**_model_kwargs(mcfg))
