"""Tests du réseau et des utilitaires d'appareil / distribué.

Exécuter :  python -m tests.test_model
"""

from __future__ import annotations

import torch

from sanchess.encoding import INPUT_PLANES, POLICY_SIZE, TACTICAL_INPUT_PLANES
from sanchess.model import (SanChessNet, SanChessNetV2, SanChessNetV3,
                            build_model, build_model_from_checkpoint)
from sanchess.utils import (amp_enabled, device_kind, load_model_state,
                            resolve_device)


def test_forward_shapes():
    """Sortie : politique (B, 4672) et valeur (B, 1) dans [-1, 1]."""
    for se in (False, True):
        for policy_head, activation in (("flat", "relu"), ("conv", "mish")):
            net = SanChessNet(channels=16, blocks=2, se=se,
                              policy_head=policy_head,
                              activation=activation).eval()
            x = torch.randn(4, INPUT_PLANES, 8, 8)
            with torch.no_grad():
                logits, value, moves_left = net(x)
            assert logits.shape == (4, POLICY_SIZE), logits.shape
            assert value.shape == (4, 1), value.shape
            assert torch.all(value >= -1) and torch.all(value <= 1)
            assert moves_left is None       # tête moves-left désactivée par défaut


def test_conv_policy_ordering():
    """L'aplatissement de la tête conv suit l'ordre `from_sq * 73 + plane`
    de encoding.move_to_index."""
    from sanchess.encoding import PLANES_PER_SQUARE
    from sanchess.model import _policy_planes_to_logits

    p = torch.empty(1, PLANES_PER_SQUARE, 8, 8)
    for plane in range(PLANES_PER_SQUARE):
        for r in range(8):
            for f in range(8):
                p[0, plane, r, f] = (r * 8 + f) * PLANES_PER_SQUARE + plane
    flat = _policy_planes_to_logits(p)
    assert torch.equal(flat[0], torch.arange(POLICY_SIZE, dtype=p.dtype))


def test_build_from_config_and_checkpoint():
    cfg = {"model": {"channels": 16, "blocks": 2, "se": True, "se_ratio": 2,
                     "value_hidden": 32, "policy_head": "conv",
                     "value_channels": 16, "activation": "mish"}}
    net = build_model(cfg)
    ckpt = {"model_cfg": cfg["model"], "model_state": net.state_dict()}
    rebuilt = build_model_from_checkpoint(ckpt, {})
    # Les poids du checkpoint doivent se charger sans clé manquante.
    rebuilt.load_state_dict(ckpt["model_state"])
    assert rebuilt.cfg_meta == net.cfg_meta


def test_v2_forward_and_checkpoint_arch():
    cfg = {"model": {"arch": "v2", "channels": 16, "blocks": 3, "se": True,
                     "policy_head": "conv", "value_head": "wdl",
                     "activation": "silu", "input_features": "tactical",
                     "attention_every": 2, "attention_heads": 4}}
    net = build_model(cfg).eval()
    assert isinstance(net, SanChessNetV2)
    x = torch.randn(2, TACTICAL_INPUT_PLANES, 8, 8)
    with torch.no_grad():
        logits, value, moves_left = net(x)
    assert logits.shape == (2, POLICY_SIZE)
    assert value.shape == (2, 3)
    assert moves_left is None       # v2 : pas de tête moves-left

    ckpt = {"model_cfg": cfg["model"], "model_state": net.state_dict()}
    rebuilt = build_model_from_checkpoint(ckpt, {})
    rebuilt.load_state_dict(ckpt["model_state"])
    assert rebuilt.cfg_meta["arch"] == "v2"


def test_v3_geometric_attention_and_moves_left_head():
    """v3 : attention géométrique efficiente + tête moves-left (B,) >= 0 ;
    round-trip d'archi via le checkpoint."""
    cfg = {"model": {"arch": "v3", "channels": 32, "blocks": 4, "se": True,
                     "policy_head": "conv", "value_head": "wdl",
                     "activation": "mish", "input_features": "tactical",
                     "attention_every": 2, "attention_heads": 4,
                     "attention_dim": 16}}
    net = build_model(cfg).eval()
    assert isinstance(net, SanChessNetV3)
    assert net.has_moves_left
    x = torch.randn(3, TACTICAL_INPUT_PLANES, 8, 8)
    with torch.no_grad():
        logits, value, moves_left = net(x)
    assert logits.shape == (3, POLICY_SIZE)
    assert value.shape == (3, 3)
    assert moves_left.shape == (3,)
    assert torch.all(moves_left >= 0)            # softplus

    ckpt = {"model_cfg": net.cfg_meta, "model_state": net.state_dict()}
    rebuilt = build_model_from_checkpoint(ckpt, {})
    rebuilt.load_state_dict(ckpt["model_state"])   # archi reconstruite à l'identique
    assert rebuilt.cfg_meta["arch"] == "v3"
    assert rebuilt.cfg_meta["moves_left_head"] is True


def test_v3_moves_left_head_can_be_disabled():
    """v3 sans tête moves-left -> forward renvoie moves_left=None."""
    cfg = {"model": {"arch": "v3", "channels": 16, "blocks": 2,
                     "attention_every": 0, "moves_left_head": False}}
    net = build_model(cfg).eval()
    assert not net.has_moves_left
    with torch.no_grad():
        _, _, moves_left = net(torch.randn(2, TACTICAL_INPUT_PLANES, 8, 8))
    assert moves_left is None


def test_tolerant_load_across_architectures():
    """Charger des poids non-SE dans un réseau SE ne doit pas planter."""
    small = SanChessNet(channels=16, blocks=2, se=False)
    big = SanChessNet(channels=16, blocks=2, se=True)
    load_model_state(big, small.state_dict(), strict=False)   # ne lève pas


def test_resolve_device_and_helpers():
    dev = resolve_device("auto")
    assert dev in ("cuda", "mps", "cpu")
    assert resolve_device("cpu") == "cpu"
    assert device_kind("cuda:0") == "cuda"
    assert device_kind("mps") == "mps"
    assert device_kind("cpu") == "cpu"
    # AMP n'est activé que sur CUDA.
    assert amp_enabled("cpu", True) is False
    assert amp_enabled("mps", True) is False


def test_flatten_roundtrip():
    """Les helpers de (un)flatten du trainer distribué préservent les valeurs."""
    from sanchess.train.distributed import _flatten, _unflatten_into

    src = [torch.randn(3, 4), torch.randn(5)]
    dst = [torch.zeros(3, 4), torch.zeros(5)]
    _unflatten_into(_flatten(src), dst)
    for a, b in zip(src, dst):
        assert torch.allclose(a, b, atol=1e-6)


def test_masked_policy_loss_ignores_illegal_logits():
    from sanchess.train.losses import policy_loss

    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False]])
    logits_a = torch.tensor([[2.0, 0.5, 100.0, -100.0]])
    logits_b = torch.tensor([[2.0, 0.5, -100.0, 100.0]])
    loss_a = policy_loss(logits_a, target, legal_mask=mask)
    loss_b = policy_loss(logits_b, target, legal_mask=mask)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_masked_label_smoothing_only_uses_legal_moves():
    from sanchess.train.losses import policy_loss

    logits = torch.tensor([[0.0, 0.0, 50.0, 50.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False]])
    loss = policy_loss(logits, target, label_smoothing=0.2, legal_mask=mask)
    loss.backward()
    assert logits.grad[0, 2].abs() == 0
    assert logits.grad[0, 3].abs() == 0


if __name__ == "__main__":
    test_forward_shapes()
    test_conv_policy_ordering()
    test_build_from_config_and_checkpoint()
    test_v2_forward_and_checkpoint_arch()
    test_v3_geometric_attention_and_moves_left_head()
    test_v3_moves_left_head_can_be_disabled()
    test_tolerant_load_across_architectures()
    test_resolve_device_and_helpers()
    test_flatten_roundtrip()
    test_masked_policy_loss_ignores_illegal_logits()
    test_masked_label_smoothing_only_uses_legal_moves()
    print("OK — tous les tests du modèle / distribué passent.")
