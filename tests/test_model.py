"""Tests du réseau et des utilitaires d'appareil / distribué.

Exécuter :  python -m tests.test_model
"""

from __future__ import annotations

import torch

from sanchess.encoding import INPUT_PLANES, POLICY_SIZE
from sanchess.model import (SanChessNet, build_model,
                            build_model_from_checkpoint)
from sanchess.utils import (amp_enabled, device_kind, load_model_state,
                            resolve_device)


def test_forward_shapes():
    """Sortie : politique (B, 4672) et valeur (B, 1) dans [-1, 1]."""
    for se in (False, True):
        net = SanChessNet(channels=16, blocks=2, se=se).eval()
        x = torch.randn(4, INPUT_PLANES, 8, 8)
        with torch.no_grad():
            logits, value = net(x)
        assert logits.shape == (4, POLICY_SIZE), logits.shape
        assert value.shape == (4, 1), value.shape
        assert torch.all(value >= -1) and torch.all(value <= 1)


def test_build_from_config_and_checkpoint():
    cfg = {"model": {"channels": 16, "blocks": 2, "se": True, "se_ratio": 2,
                     "value_hidden": 32}}
    net = build_model(cfg)
    ckpt = {"model_cfg": cfg["model"], "model_state": net.state_dict()}
    rebuilt = build_model_from_checkpoint(ckpt, {})
    # Les poids du checkpoint doivent se charger sans clé manquante.
    rebuilt.load_state_dict(ckpt["model_state"])
    assert rebuilt.cfg_meta == net.cfg_meta


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


if __name__ == "__main__":
    test_forward_shapes()
    test_build_from_config_and_checkpoint()
    test_tolerant_load_across_architectures()
    test_resolve_device_and_helpers()
    test_flatten_roundtrip()
    print("OK — tous les tests du modèle / distribué passent.")
