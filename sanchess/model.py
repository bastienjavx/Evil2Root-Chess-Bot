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

from .encoding import PLANES_PER_SQUARE, POLICY_SIZE, input_planes_for_features

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


class GlobalAttentionBlock(nn.Module):
    """Attention globale sur les 64 cases, injectée entre blocs résiduels v2."""

    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("attention_heads doit diviser channels")
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * 2
        self.ff = nn.Sequential(
            nn.Linear(channels, hidden),
            _make_act(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)       # (B, 64, C)
        y = self.norm1(tokens)
        y, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + y
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class GeometricAttentionBlock(nn.Module):
    """Attention globale EFFICIENTE (arch v3), optimisée pour Turing (pas de
    FlashAttention) et adaptée aux échecs.

    Deux idées :
      - **dimension réduite** : on projette `C -> attention_dim` (< C) avant
        l'attention puis on re-projette, donc le coût devient `~C·d` au lieu de
        `C²` — bien plus rapide que `nn.MultiheadAttention` à canaux égaux ;
      - **biais de position relatif appris** `(heads, 64, 64)` ajouté aux scores
        d'attention : les 64 cases ont une géométrie FIXE (distances cavalier,
        colonnes, diagonales), donc un biais appris par paire de cases capture
        ces relations case↔case SANS aucun coût par token ni par position. C'est
        l'innovation « chess-specific » de la v3, à coût quasi nul.

    Utilise `F.scaled_dot_product_attention` (kernel fusé mémoire-efficient).
    """

    def __init__(self, channels: int, heads: int = 8,
                 attention_dim: int | None = None, dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        d = attention_dim if attention_dim else max(channels // 2, heads)
        d = max((d // heads) * heads, heads)   # arrondi au multiple de heads
        self.heads = heads
        self.dim = d
        self.head_dim = d // heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * d)
        self.proj = nn.Linear(d, channels)
        # Biais relatif appris : un (64, 64) par tête, ajouté aux scores.
        self.rel_bias = nn.Parameter(torch.zeros(heads, 64, 64))
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * 2
        self.ff = nn.Sequential(
            nn.Linear(channels, hidden),
            _make_act(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w                                    # 64 (échiquier 8x8)
        tokens = x.flatten(2).transpose(1, 2)        # (B, N, C)
        y = self.norm1(tokens)
        qkv = self.qkv(y).reshape(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Biais additif (1, heads, N, N) -> broadcast sur le batch.
        attn_mask = self.rel_bias.unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        tokens = tokens + self.proj(out)
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class SanChessNet(nn.Module):
    """Entrée (B, C, 8, 8) -> (policy_logits (B, 4672), value)."""

    def __init__(self, channels: int = 128, blocks: int = 10,
                 se: bool = False, se_ratio: int = 4, value_hidden: int = 256,
                 policy_head: str = "flat", value_channels: int = 8,
                 activation: str = "relu", value_head: str = "scalar",
                 arch: str = "v1", input_features: str = "base",
                 attention_every: int = 0, attention_heads: int = 8,
                 attention_dropout: float = 0.0, attention_dim: int = 0,
                 moves_left_head: bool = False, moves_left_hidden: int = 128):
        super().__init__()
        if policy_head not in ("flat", "conv"):
            raise ValueError(f"policy_head inconnu: {policy_head!r} "
                             f"(choix: 'flat', 'conv')")
        if value_head not in ("scalar", "wdl"):
            raise ValueError(f"value_head inconnu: {value_head!r} "
                             f"(choix: 'scalar', 'wdl')")
        if arch not in ("v1", "v2", "v3"):
            raise ValueError(f"arch inconnue: {arch!r} (choix: 'v1', 'v2', 'v3')")
        if arch == "v1":
            attention_every = 0
            input_features = "base"
            moves_left_head = False
        input_planes = input_planes_for_features(input_features)
        self.cfg_meta = {"channels": channels, "blocks": blocks,
                         "se": se, "se_ratio": se_ratio,
                         "value_hidden": value_hidden,
                         "policy_head": policy_head,
                         "value_channels": value_channels,
                         "activation": activation,
                         "value_head": value_head,
                         "arch": arch,
                         "input_features": input_features,
                         "attention_every": attention_every,
                         "attention_heads": attention_heads,
                         "attention_dropout": attention_dropout,
                         "attention_dim": attention_dim,
                         "moves_left_head": moves_left_head,
                         "moves_left_hidden": moves_left_hidden}
        self.policy_head = policy_head
        self.value_head = value_head
        self.arch = arch
        self.input_features = input_features
        self.has_moves_left = bool(moves_left_head)
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            _make_act(activation),
        )
        tower: list[nn.Module] = []
        for i in range(blocks):
            tower.append(ResidualBlock(channels, se=se, se_ratio=se_ratio,
                                       activation=activation))
            if attention_every and (i + 1) % attention_every == 0:
                if arch == "v3":
                    # Attention géométrique efficiente (dim réduite + biais appris).
                    tower.append(GeometricAttentionBlock(
                        channels, attention_heads, attention_dim or None,
                        attention_dropout, activation))
                else:
                    tower.append(GlobalAttentionBlock(channels, attention_heads,
                                                      attention_dropout, activation))
        self.tower = nn.Sequential(*tower)

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

        # Tête « moves-left » (optionnelle) : prédit les demi-coups restants
        # avant la fin de partie (normalisés /100). Pooling global -> MLP -> 1,
        # softplus pour rester >= 0. Sert au MCTS à départager des coups de valeur
        # quasi égale (gagner vite / perdre lentement, anti-nulle 50 coups).
        if moves_left_head:
            self.moves_left_conv = nn.Sequential(
                nn.Conv2d(channels, value_channels, 1, bias=False),
                nn.BatchNorm2d(value_channels),
                _make_act(activation),
            )
            self.moves_left_fc = nn.Sequential(
                nn.Linear(value_channels * 8 * 8, moves_left_hidden),
                _make_act(activation),
                nn.Linear(moves_left_hidden, 1),
            )
        else:
            self.moves_left_conv = None
            self.moves_left_fc = None
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

    def forward(self, x: torch.Tensor):
        """-> (policy_logits (B,4672), value, moves_left).

        `moves_left` vaut `None` si la tête est désactivée (v1/v2 et v3 sans
        `moves_left_head`), sinon un tenseur (B,) de demi-coups restants
        normalisés (>= 0). Tous les appelants déballent les 3 valeurs.
        """
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

        moves_left = None
        if self.moves_left_fc is not None:
            ml = self.moves_left_conv(x).flatten(1)
            moves_left = F.softplus(self.moves_left_fc(ml)).squeeze(1)

        return policy_logits, value, moves_left


class SanChessNetV2(SanChessNet):
    """Réseau v2 : résidus SE + attention globale optionnelle + features riches."""

    def __init__(self, **kwargs):
        kwargs.setdefault("arch", "v2")
        kwargs.setdefault("policy_head", "conv")
        kwargs.setdefault("value_head", "wdl")
        kwargs.setdefault("input_features", "tactical")
        kwargs.setdefault("attention_every", 6)
        super().__init__(**kwargs)


class SanChessNetV3(SanChessNet):
    """Réseau v3 : tronc SE + **attention géométrique efficiente** (SDPA, dim
    réduite, biais de position appris) + **tête moves-left** — conçu pour être
    rapide sur Turing (RTX 2070S) sans perdre en force."""

    def __init__(self, **kwargs):
        kwargs.setdefault("arch", "v3")
        kwargs.setdefault("policy_head", "conv")
        kwargs.setdefault("value_head", "wdl")
        kwargs.setdefault("input_features", "tactical")
        kwargs.setdefault("se", True)
        kwargs.setdefault("attention_every", 5)
        kwargs.setdefault("moves_left_head", True)
        super().__init__(**kwargs)


def _model_kwargs(m: dict) -> dict:
    # Les défauts correspondent à l'architecture historique afin que les
    # checkpoints antérieurs (sans ces clés) soient reconstruits à l'identique.
    arch = str(m.get("arch", "v1"))
    is_v2 = arch == "v2"
    is_v3 = arch == "v3"
    rich = is_v2 or is_v3   # défauts « modernes » (conv/wdl/tactical) pour v2 et v3
    return {
        "channels": m.get("channels", 128),
        "blocks": m.get("blocks", 10),
        "se": bool(m.get("se", True if is_v3 else False)),
        "se_ratio": int(m.get("se_ratio", 4)),
        "value_hidden": int(m.get("value_hidden", 256)),
        "policy_head": str(m.get("policy_head", "conv" if rich else "flat")),
        "value_channels": int(m.get("value_channels", 8)),
        "activation": str(m.get("activation", "relu")),
        "value_head": str(m.get("value_head", "wdl" if rich else "scalar")),
        "arch": arch,
        "input_features": str(m.get("input_features", "tactical" if rich else "base")),
        "attention_every": int(m.get("attention_every", 5 if is_v3 else (6 if is_v2 else 0))),
        "attention_heads": int(m.get("attention_heads", 8)),
        "attention_dropout": float(m.get("attention_dropout", 0.0)),
        "attention_dim": int(m.get("attention_dim", 0)),
        "moves_left_head": bool(m.get("moves_left_head", True if is_v3 else False)),
        "moves_left_hidden": int(m.get("moves_left_hidden", 128)),
    }


def _build_for_arch(kwargs: dict) -> SanChessNet:
    arch = kwargs.get("arch")
    if arch == "v3":
        return SanChessNetV3(**kwargs)
    if arch == "v2":
        return SanChessNetV2(**kwargs)
    return SanChessNet(**kwargs)


def build_model(cfg: dict) -> SanChessNet:
    """Construit le réseau d'après la section `model` de la config."""
    return _build_for_arch(_model_kwargs(cfg.get("model", {})))


def build_model_from_checkpoint(ckpt: dict, fallback_cfg: dict | None = None) -> SanChessNet:
    """Construit le réseau d'après l'archi enregistrée dans le checkpoint.

    Évite les divergences trainer/moteur lors du hot-reload : on respecte la
    forme avec laquelle les poids ont été entraînés plutôt que le `config.yaml`
    courant. Repli sur la config fournie si le checkpoint n'embarque pas d'archi.
    """
    mcfg = ckpt.get("model_cfg") or (fallback_cfg or {}).get("model", {})
    return _build_for_arch(_model_kwargs(mcfg))
