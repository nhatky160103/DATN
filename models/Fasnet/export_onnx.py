"""
Export MiniFASNet weights (.pth) → ONNX

Key design decisions:
- Model type and input size are parsed directly from the weight filename
  (same convention as the original repo's src/utility.py::parse_model_name)
- conv6_kernel is derived from input size via get_kernel(), not hardcoded
- Dummy input shape matches the actual model input (H, W from filename)

Weight filename convention:
    <scale>_<H>x<W>_<ModelType>.pth
    e.g. 2.7_80x80_MiniFASNetV2.pth
         4_0_0_80x80_MiniFASNetV1SE.pth

Usage:
    # Export a single weight file (model type + input size auto-detected from filename)
    python -m models.Anti_spoof.export_onnx --weights models/Anti_spoof/weights/2.7_80x80_MiniFASNetV2.pth
    python -m models.Anti_spoof.export_onnx --weights models/Anti_spoof/weights/4_0_0_80x80_MiniFASNetV1SE.pth

    # Export both default weights at once
    python -m models.Anti_spoof.export_onnx --all

    # Verify ONNX output after export
    python -m models.Anti_spoof.export_onnx --all --verify

    # Include softmax in the exported graph
    python -m models.Anti_spoof.export_onnx --all --with-softmax
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

from models.Fasnet.FasNetBackbone import MiniFASNetV1SE, MiniFASNetV2

# ── Constants ─────────────────────────────────────────────────────────────────

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

# Default weight files shipped with the project (names encode meta-data)
DEFAULT_WEIGHTS = [
    "2.7_80x80_MiniFASNetV2.pth",
    "4_0_0_80x80_MiniFASNetV1SE.pth",
]

MODEL_MAPPING = {
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2":   MiniFASNetV2,
}


# ── Helpers (mirrors src/utility.py from the original repo) ───────────────────

def get_kernel(height: int, width: int) -> tuple[int, int]:
    """
    Compute conv6_kernel from input resolution.
    Formula from original repo: utility.py::get_kernel()
        kernel = ((H + 15) // 16, (W + 15) // 16)
    Examples:
        80x80  → (5, 5)
        128x128 → (8, 8)
    """
    return ((height + 15) // 16, (width + 15) // 16)


def parse_model_name(filename: str) -> tuple[int, int, str]:
    """
    Extract (H, W, model_type) from weight filename.
    Mirrors src/utility.py::parse_model_name() from the original repo.

    Examples:
        '2.7_80x80_MiniFASNetV2.pth'     → (80, 80, 'MiniFASNetV2')
        '4_0_0_80x80_MiniFASNetV1SE.pth' → (80, 80, 'MiniFASNetV1SE')
    """
    stem = Path(filename).stem          # strip .pth
    parts = stem.split("_")

    # Last part before extension is model type; second-to-last is HxW
    model_type = parts[-1]
    h_str, w_str = parts[-2].split("x")

    if model_type not in MODEL_MAPPING:
        raise ValueError(
            f"Unknown model type '{model_type}' in filename '{filename}'.\n"
            f"Supported: {list(MODEL_MAPPING)}"
        )

    return int(h_str), int(w_str), model_type


# ── Weight loading ────────────────────────────────────────────────────────────

def load_state(model: nn.Module, weights_path: str) -> None:
    """Load state dict, stripping 'module.' prefix if present (DataParallel)."""
    state = torch.load(weights_path, map_location="cpu")
    first_key = next(iter(state))
    if first_key.startswith("module."):
        state = OrderedDict((k[7:], v) for k, v in state.items())
    model.load_state_dict(state)


# ── Optional softmax wrapper ──────────────────────────────────────────────────

class _WithSoftmax(nn.Module):
    """Thin wrapper that appends softmax(dim=1) to the model output."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(x), dim=1)


# ── Core export ───────────────────────────────────────────────────────────────

def export(weights_path: str, output_path: str | None = None,
           opset: int = 13, with_softmax: bool = False) -> str:
    """
    Export a single .pth → .onnx.

    Args:
        weights_path:  Path to the .pth weight file.
        output_path:   Destination .onnx path. Defaults to same dir as weights,
                       same stem + '.onnx'.
        opset:         ONNX opset version (default 13).
        with_softmax:  If True, wrap model with softmax before export.

    Returns:
        Absolute path of the exported .onnx file.
    """
    weights_path = os.path.abspath(weights_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    # ── Parse filename to get model meta-data ─────────────────────────────────
    filename = os.path.basename(weights_path)
    h_input, w_input, model_type = parse_model_name(filename)
    conv6_kernel = get_kernel(h_input, w_input)

    print(f"[export] {filename}")
    print(f"  model_type   : {model_type}")
    print(f"  input size   : {h_input}x{w_input}")
    print(f"  conv6_kernel : {conv6_kernel}")
    print(f"  with_softmax : {with_softmax}")

    # ── Build model ───────────────────────────────────────────────────────────
    model_cls = MODEL_MAPPING[model_type]
    model = model_cls(conv6_kernel=conv6_kernel)
    load_state(model, weights_path)
    model.eval()

    # ── Wrap with softmax if requested ────────────────────────────────────────
    export_model = _WithSoftmax(model) if with_softmax else model

    # ── Determine output path ─────────────────────────────────────────────────
    if output_path is None:
        output_path = weights_path.replace(".pth", ".onnx")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Export ────────────────────────────────────────────────────────────────
    # Input shape: (batch, channels, H, W) — channels=3 (RGB)
    dummy = torch.zeros(1, 3, h_input, w_input)

    output_node = "probs" if with_softmax else "logits"

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=[output_node],
        dynamic_axes={"input": {0: "batch"}, output_node: {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        dynamo=False,
        external_data=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✓ Saved → {output_path}  ({size_mb:.2f} MB)\n")
    return output_path


# ── Verify ────────────────────────────────────────────────────────────────────

def verify(onnx_path: str) -> None:
    """Run a dummy forward pass through the exported ONNX model."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
        return

    print(f"[verify] {os.path.basename(onnx_path)}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inp  = session.get_inputs()[0]
    out  = session.get_outputs()[0]
    print(f"  Input  : name='{inp.name}'  shape={inp.shape}  dtype={inp.type}")
    print(f"  Output : name='{out.name}'  shape={out.shape}  dtype={out.type}")

    # Build dummy input matching the exported shape
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    dummy = np.random.rand(*shape).astype(np.float32)
    result = session.run(None, {inp.name: dummy})[0]

    print(f"  Result : shape={result.shape}  values={result}")
    print(f"  ✓ ONNX model works correctly!\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MiniFASNet .pth → .onnx"
    )
    parser.add_argument(
        "--weights",
        help="Path to a single .pth weight file"
    )
    parser.add_argument(
        "--output",
        help="Output .onnx path (default: same dir as weights, same stem)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help=f"Export all default weights: {DEFAULT_WEIGHTS}"
    )
    parser.add_argument(
        "--opset", type=int, default=13,
        help="ONNX opset version (default: 13)"
    )
    parser.add_argument(
        "--with-softmax", action="store_true",
        help="Append softmax(dim=1) to the exported graph (output node: 'probs')"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a test forward pass after export"
    )
    args = parser.parse_args()

    if not args.all and not args.weights:
        parser.error("Provide --weights <path> or --all")

    targets = (
        [os.path.join(WEIGHTS_DIR, w) for w in DEFAULT_WEIGHTS]
        if args.all
        else [args.weights]
    )

    for weights_path in targets:
        onnx_path = export(
            weights_path=weights_path,
            output_path=args.output if not args.all else None,
            opset=args.opset,
            with_softmax=args.with_softmax,
        )
        if args.verify:
            verify(onnx_path)
