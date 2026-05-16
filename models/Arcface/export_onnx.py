from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnx

from models.Arcface.backbones import get_model


def export_arcface(weights: str, network: str, output: str, opset: int = 13) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = get_model(network, dropout=0.0, fp16=False, num_features=512)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        dynamo=False,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"Exported ArcFace ONNX to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/Arcface/weights/backbone.pth")
    parser.add_argument("--network", default="r18")
    parser.add_argument("--output", default="triton_model_repository/arcface/1/model.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()
    export_arcface(args.weights, args.network, args.output, args.opset)


if __name__ == "__main__":
    main()
