from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch

from models.Anti_spoof import FasNetBackbone


def _load_state(model: torch.nn.Module, weights: str) -> None:
    state = torch.load(weights, map_location="cpu")
    first_key = next(iter(state))
    if first_key.startswith("module."):
        state = OrderedDict((key[7:], value) for key, value in state.items())
    model.load_state_dict(state)


def export(model_type: str, weights: str, output: str, opset: int = 13) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "v1se":
        model = FasNetBackbone.MiniFASNetV1SE(conv6_kernel=(5, 5))
    elif model_type == "v2":
        model = FasNetBackbone.MiniFASNetV2(conv6_kernel=(5, 5))
    else:
        raise ValueError("model_type must be one of: v1se, v2")

    _load_state(model, weights)
    model.eval()

    dummy = torch.randn(1, 3, 80, 80)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported FASNet {model_type} ONNX to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["v1se", "v2"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()
    export(args.model_type, args.weights, args.output, args.opset)


if __name__ == "__main__":
    main()
