"""Réseau résiduel type AlphaZero/Lc0 : corps convolutif (+ Squeeze-Excitation
optionnel) + tête politique + tête valeur.

Améliorations vs. réseau de base (toutes éprouvées par Leela Chess Zero) :
  - blocs résiduels avec **Squeeze-and-Excitation** (recalibrage par canal,
    gain de force notable à coût quasi nul — cf. Leela Chess Zero) ;
  - **tête politique convolutive** (`policy_head: conv`) : sortie directe en
    73 plans 8x8 alignés sur l'encodage `move_to_index`, au lieu d'un FC géant
    32*64 -> 4672. Préserve la structure spatiale des coups, économise ~7 M de
    paramètres et apporte un gain d'Elo important (amélioration historique
    majeure des réseaux Lc0) ;
  - **activation configurable** (`activation: mish|silu|relu`) — Mish a donné
    un gain mesurable sur les réseaux Lc0 par rapport à ReLU ;
  - **init à zéro du gamma du dernier BatchNorm** de chaque bloc résiduel :
    chaque bloc démarre comme l'identité, ce qui stabilise l'entraînement des
    tours profondes (20 blocs et plus) ;
  - tête valeur à largeur configurable (canaux conv + couche cachée) ;
  - initialisation Kaiming explicite des convolutions/linéaires.

L'architecture reste pilotée par `config.yaml` (section `model`) et reste
compatible avec les anciens checkpoints : les nouvelles options ont des défauts
« legacy » (tête politique FC, ReLU) afin que `build_model_from_checkpoint`
reconstruise exactement la forme d'origine, et `utils.load_model_state` reste
tolérant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import INPUT_PLANES, PLANES_PER_SQUARE, POLICY_SIZE

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "mish": nn.Mish,
    "silu": nn.SiLU,
}


def _make_act(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()](inplace=True)
    except KeyError:
        raise ValueError(f"activation inconnue: {name!r} "
                         f"(choix: {sorted(_ACTIVATIONS)})") from None


def _policy_planes_to_logits(p: torch.Tensor) -> torch.Tensor:
    """(B, 73, 8, 8) -> (B, 4672) dans l'ordre `from_sq * 73 + plane`.

    Doit rester aligné sur `encoding.move_to_index` : index = (rank*8 + file)
    * PLANES_PER_SQUARE + plane, d'où l'ordre (rank, file, plane).
    """
    return p.permute(0, 2, 3, 1).flatten(1)


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
    def __init__(self, channels: int, se: bool = False, se_ratio: int = 4,
                 activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio) if se else None
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return self.act(out + x)


class SanChessNet(nn.Module):
    """Entrée (B, INPUT_PLANES, 8, 8) -> (policy_logits (B, 4672), value (B, 1))."""

    def __init__(self, channels: int = 128, blocks: int = 10,
                 se: bool = False, se_ratio: int = 4, value_hidden: int = 256,
                 policy_head: str = "flat", value_channels: int = 8,
                 activation: str = "relu", value_head: str = "scalar"):
        super().__init__()
        if policy_head not in ("flat", "conv"):
            raise ValueError(f"policy_head inconnu: {policy_head!r} "
                             f"(choix: 'flat', 'conv')")
        if value_head not in ("scalar", "wdl"):
            raise ValueError(f"value_head inconnu: {value_head!r} "
                             f"(choix: 'scalar', 'wdl')")
        self.cfg_meta = {"channels": channels, "blocks": blocks,
                         "se": se, "se_ratio": se_ratio,
                         "value_hidden": value_hidden,
                         "policy_head": policy_head,
                         "value_channels": value_channels,
                         "activation": activation,
                         "value_head": value_head}
        self.policy_head = policy_head
        self.value_head = value_head
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            _make_act(activation),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(channels, se=se, se_ratio=se_ratio,
                            activation=activation) for _ in range(blocks)]
        )

        # Tête politique
        if policy_head == "conv":
            # Style Lc0 : sortie spatiale (B, 73, 8, 8) alignée sur l'encodage.
            self.policy_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                _make_act(activation),
                nn.Conv2d(channels, PLANES_PER_SQUARE, 3, padding=1),
            )
            self.policy_fc = None
        else:
            self.policy_conv = nn.Sequential(
                nn.Conv2d(channels, 32, 1, bias=False),
                nn.BatchNorm2d(32),
                _make_act(activation),
            )
            self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_SIZE)

        # Tête valeur
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_channels, 1, bias=False),
            nn.BatchNorm2d(value_channels),
            _make_act(activation),
        )
        # Tête scalaire -> 1 sortie (tanh) ; tête WDL -> 3 logits (défaite/nulle/victoire).
        value_out = 3 if value_head == "wdl" else 1
        self.value_fc = nn.Sequential(
            nn.Linear(value_channels * 8 * 8, value_hidden),
            _make_act(activation),
            nn.Linear(value_hidden, value_out),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Branche résiduelle initialisée à zéro : chaque bloc démarre comme
        # l'identité, ce qui stabilise l'entraînement des tours profondes.
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)

        if self.policy_fc is None:
            policy_logits = _policy_planes_to_logits(self.policy_conv(x))
        else:
            p = self.policy_conv(x).flatten(1)
            policy_logits = self.policy_fc(p)

        v = self.value_conv(x).flatten(1)
        v = self.value_fc(v)
        # Scalaire : tanh ∈ [-1,1] (point de vue du trait). WDL : 3 logits bruts
        # (défaite/nulle/victoire) -> softmax + perte/scalaire gérés en aval.
        value = v if self.value_head == "wdl" else torch.tanh(v)

        return policy_logits, value


def _model_kwargs(m: dict) -> dict:
    # Les défauts correspondent à l'architecture historique afin que les
    # checkpoints antérieurs (sans ces clés) soient reconstruits à l'identique.
    return {
        "channels": m.get("channels", 128),
        "blocks": m.get("blocks", 10),
        "se": bool(m.get("se", False)),
        "se_ratio": int(m.get("se_ratio", 4)),
        "value_hidden": int(m.get("value_hidden", 256)),
        "policy_head": str(m.get("policy_head", "flat")),
        "value_channels": int(m.get("value_channels", 8)),
        "activation": str(m.get("activation", "relu")),
        "value_head": str(m.get("value_head", "scalar")),
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
