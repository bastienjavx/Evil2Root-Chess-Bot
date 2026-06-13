"""Export du réseau pour une inférence PLUS RAPIDE sur la 2070S (Turing).

En blitz, la force vient du nombre de nœuds MCTS à temps fixe, donc du débit
d'évaluation du réseau. PyTorch eager + autocast laisse de la performance sur la
table ; TensorRT fp16 tourne typiquement 2-3× plus vite sur Turing -> ~2× de
nœuds -> gain d'Elo réel, et permet de garder un réseau plus gros (entraîné sur
le cloud) jouable en temps réel ici.

Deux cibles :

  1. ONNX (`export_onnx`) — format portable. Sert à construire un moteur TensorRT
     hors-ligne avec `trtexec --onnx=... --fp16 --saveEngine=...`, ou à servir via
     onnxruntime-gpu. Batch dynamique (la racine = batch 1, les feuilles = batch
     `eval_batch_size`, plus un dernier lot partiel).

  2. TorchScript+TensorRT (`compile_trt`) — `torch_tensorrt.compile` renvoie un
     module à l'interface IDENTIQUE au modèle (`logits, value = module(x)`), donc
     l'`Evaluator` du MCTS le consomme SANS aucune modification. On le sérialise
     (`torch.jit.save`) et on le recharge via `load_compiled`.

Intégration au jeu : renseigner `model.trt_engine: chemin.ts` dans config.yaml.
`make_inference_model` charge alors le moteur compilé à la place du modèle eager ;
en cas d'absence de torch_tensorrt / de fichier, on retombe proprement sur le
modèle PyTorch (aucune régression). La COMPILATION exige un GPU (à faire sur la
2070S cible, pas sur le cloud : un moteur TensorRT est spécifique au GPU/driver).

Usage :
  # ONNX (pour trtexec / onnxruntime)
  python -m sanchess.export --ckpt checkpoints/latest.pt --onnx checkpoints/sano1.onnx
  # TorchScript+TensorRT fp16 (à lancer SUR la 2070S)
  python -m sanchess.export --ckpt checkpoints/latest.pt --trt checkpoints/sano1.ts --fp16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .encoding import INPUT_PLANES
from .model import build_model, build_model_from_checkpoint
from .utils import load_checkpoint, load_config, load_model_state, resolve_device


def load_model_for_export(ckpt_path: Path, cfg: dict, device: str):
    """Reconstruit le réseau depuis l'archi EMBARQUÉE dans le checkpoint (et non
    la config courante) et charge les poids. Renvoie le modèle en eval()."""
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, map_location=device)
        model = build_model_from_checkpoint(ck, cfg)
        load_model_state(model, ck["model_state"])
    else:
        model = build_model(cfg)
        sys.stderr.write("⚠ aucun checkpoint : export d'un réseau aléatoire.\n")
    return model.to(device).eval()


def export_onnx(model, out_path: Path, device: str,
                opset: int = 17, sample_batch: int = 16) -> Path:
    """Exporte en ONNX avec un axe de batch DYNAMIQUE (batch 1..N supportés)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(sample_batch, INPUT_PLANES, 8, 8, device=device)
    dyn = {"board": {0: "batch"},
           "policy": {0: "batch"}, "value": {0: "batch"}}
    # dynamo=False : exporteur TorchScript historique (pas de dépendance onnxscript).
    # Suffisant pour ce CNN statique et compatible trtexec / onnxruntime-gpu.
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["board"], output_names=["policy", "value"],
        dynamic_axes=dyn, opset_version=opset,
        do_constant_folding=True, dynamo=False,
    )
    return out_path


def compile_trt(model, device: str, fp16: bool = True,
                min_batch: int = 1, opt_batch: int = 16, max_batch: int = 64):
    """Compile en TensorRT via torch_tensorrt (batch dynamique). Renvoie un module
    callable comme le modèle. Lève une erreur claire si torch_tensorrt/GPU absent."""
    try:
        import torch_tensorrt
    except ImportError as e:  # pragma: no cover - dépend de l'install GPU
        raise RuntimeError(
            "torch_tensorrt introuvable. Installer la pile TensorRT sur la machine "
            "GPU cible (cf. docs NVIDIA / `pip install torch-tensorrt`), ou utiliser "
            "l'export ONNX + trtexec à la place."
        ) from e
    if not str(device).startswith("cuda"):
        raise RuntimeError("La compilation TensorRT exige un GPU CUDA (device=cuda).")

    inputs = [torch_tensorrt.Input(
        min_shape=(min_batch, INPUT_PLANES, 8, 8),
        opt_shape=(opt_batch, INPUT_PLANES, 8, 8),
        max_shape=(max_batch, INPUT_PLANES, 8, 8),
        dtype=torch.float32,
    )]
    enabled = {torch.float32}
    if fp16:
        enabled.add(torch.float16)
    return torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled)


def load_compiled(path: Path, device: str):
    """Recharge un moteur TorchScript+TensorRT (torch_tensorrt requis à l'import)."""
    try:
        import torch_tensorrt  # noqa: F401 — enregistre les ops TRT au load
    except ImportError:
        pass
    return torch.jit.load(str(path), map_location=device).eval()


def make_inference_model(cfg: dict, base_model, device: str):
    """Renvoie le moteur compilé TensorRT si `model.trt_engine` est renseigné et
    chargeable, sinon le modèle PyTorch fourni (repli sans régression). Utilisé par
    le moteur UCI / le bot / le web pour accélérer l'inférence de façon optionnelle."""
    path = (cfg.get("model", {}) or {}).get("trt_engine")
    if not path:
        return base_model
    p = Path(path)
    if not p.exists():
        sys.stderr.write(f"info string trt_engine introuvable ({p}) -> modèle PyTorch.\n")
        return base_model
    if not str(device).startswith("cuda"):
        sys.stderr.write("info string trt_engine ignoré (device != cuda) -> PyTorch.\n")
        return base_model
    try:
        m = load_compiled(p, device)
        sys.stderr.write(f"info string moteur TensorRT chargé : {p}\n")
        return m
    except Exception as e:  # noqa: BLE001 — repli robuste
        sys.stderr.write(f"info string échec chargement TensorRT ({e}) -> PyTorch.\n")
        return base_model


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=cfg["paths"]["latest"], help="checkpoint .pt")
    ap.add_argument("--onnx", default=None, help="chemin de sortie ONNX")
    ap.add_argument("--trt", default=None, help="chemin de sortie TorchScript+TensorRT")
    ap.add_argument("--fp16", action="store_true", help="activer fp16 pour TensorRT")
    ap.add_argument("--device", default=None, help="cuda / cpu (défaut: auto)")
    ap.add_argument("--opt-batch", type=int,
                    default=int(cfg.get("mcts", {}).get("eval_batch_size", 16)),
                    dest="opt_batch", help="batch d'optimisation TensorRT")
    ap.add_argument("--max-batch", type=int, default=64, dest="max_batch")
    args = ap.parse_args()
    if not args.onnx and not args.trt:
        ap.error("préciser au moins --onnx et/ou --trt")

    device = resolve_device(args.device or cfg.get("train", {}).get("device", "auto"))
    model = load_model_for_export(Path(args.ckpt), cfg, device)

    if args.onnx:
        out = export_onnx(model, Path(args.onnx), device, sample_batch=args.opt_batch)
        print(f"ONNX écrit : {out}")
        print("  -> moteur TensorRT : "
              f"trtexec --onnx={out} --fp16 --saveEngine={out.with_suffix('.plan')} "
              f"--minShapes=board:1x{INPUT_PLANES}x8x8 "
              f"--optShapes=board:{args.opt_batch}x{INPUT_PLANES}x8x8 "
              f"--maxShapes=board:{args.max_batch}x{INPUT_PLANES}x8x8")

    if args.trt:
        compiled = compile_trt(model, device, fp16=args.fp16,
                               opt_batch=args.opt_batch, max_batch=args.max_batch)
        torch.jit.save(compiled, str(args.trt))
        print(f"Moteur TensorRT écrit : {args.trt}")
        print("  -> activer dans config.yaml :  model.trt_engine: " + str(args.trt))


if __name__ == "__main__":
    main()
