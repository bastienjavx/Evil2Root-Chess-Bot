"""Export ONNX (`sanchess.export`) : le graphe exporté est valide, accepte un
batch dynamique, et reproduit numériquement la sortie PyTorch (policy + value).

Le chemin TensorRT (`compile_trt`) exige un GPU + torch_tensorrt -> non testé ici ;
on vérifie seulement qu'il échoue avec un message clair hors GPU, et que
`make_inference_model` retombe proprement sur le modèle PyTorch.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sanchess.encoding import INPUT_PLANES
from sanchess.export import compile_trt, export_onnx, make_inference_model
from sanchess.model import SanChessNet

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _model():
    torch.manual_seed(0)
    return SanChessNet(channels=16, blocks=2, se=True, policy_head="conv",
                       value_head="wdl", activation="mish").eval()


def test_onnx_export_valid_and_matches_torch(tmp_path):
    model = _model()
    out = export_onnx(model, tmp_path / "m.onnx", "cpu", sample_batch=8)
    onnx.checker.check_model(onnx.load(str(out)))      # graphe structurellement valide

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch in (1, 5, 16):                            # axe de batch dynamique
        x = np.random.randn(batch, INPUT_PLANES, 8, 8).astype(np.float32)
        pol_ort, val_ort = sess.run(["policy", "value"], {"board": x})
        with torch.no_grad():
            pol_t, val_t = model(torch.from_numpy(x))
        assert pol_ort.shape == (batch, 4672)
        assert val_ort.shape == (batch, 3)             # tête WDL
        np.testing.assert_allclose(pol_ort, pol_t.numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(val_ort, val_t.numpy(), rtol=1e-3, atol=1e-4)


def test_compile_trt_clear_error_on_cpu():
    with pytest.raises(RuntimeError):
        compile_trt(_model(), "cpu")


def test_make_inference_model_fallback():
    model = _model()
    # Pas de trt_engine -> renvoie le modèle inchangé.
    assert make_inference_model({"model": {}}, model, "cpu") is model
    # Chemin inexistant -> repli sur le modèle PyTorch.
    cfg = {"model": {"trt_engine": "/nope/does_not_exist.ts"}}
    assert make_inference_model(cfg, model, "cuda") is model
