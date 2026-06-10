"""Réseau résiduel type AlphaZero : corps convolutif + tête politique + tête valeur."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import INPUT_PLANES, POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class SanChessNet(nn.Module):
    """Entrée (B, INPUT_PLANES, 8, 8) -> (policy_logits (B, 4672), value (B, 1))."""

    def __init__(self, channels: int = 128, blocks: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

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
            nn.Linear(8 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_conv(x).flatten(1)
        policy_logits = self.policy_fc(p)

        v = self.value_conv(x).flatten(1)
        value = torch.tanh(self.value_fc(v))   # [-1, 1] du point de vue du trait

        return policy_logits, value


def build_model(cfg: dict) -> SanChessNet:
    m = cfg.get("model", {})
    return SanChessNet(channels=m.get("channels", 128), blocks=m.get("blocks", 10))
