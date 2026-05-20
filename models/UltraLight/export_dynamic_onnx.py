from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx


def _set_first_dim_dynamic(value_info, name: str = "batch") -> None:
    dims = value_info.type.tensor_type.shape.dim
    if not dims:
        return
    dims[0].ClearField("dim_value")
    dims[0].dim_param = name


def export_dynamic_batch_onnx(
    input_path: str,
    output_path: str,
    verify_batch: int = 2,
    input_width: int = 320,
    input_height: int = 240,
) -> None:
    model = onnx.load(input_path)

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    runtime_inputs = [item for item in model.graph.input if item.name not in initializer_names]
    del model.graph.input[:]
    model.graph.input.extend(runtime_inputs)

    if len(model.graph.input) != 1:
        names = [item.name for item in model.graph.input]
        raise ValueError(f"Expected one runtime input after cleanup, got: {names}")

    _set_first_dim_dynamic(model.graph.input[0])
    for output in model.graph.output:
        _set_first_dim_dynamic(output)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    onnx.checker.check_model(str(output))

    if verify_batch > 0:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("onnxruntime is required for --verify-batch") from exc

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        dummy = np.zeros((verify_batch, 3, input_height, input_width), dtype=np.float32)
        scores, boxes = session.run(None, {input_name: dummy})
        if scores.shape[0] != verify_batch or boxes.shape[0] != verify_batch:
            raise RuntimeError(f"Unexpected batch output shapes: scores={scores.shape}, boxes={boxes.shape}")
        if scores.shape[2:] != (2,) or boxes.shape[2:] != (4,) or scores.shape[1] != boxes.shape[1]:
            raise RuntimeError(f"Unexpected output shapes: scores={scores.shape}, boxes={boxes.shape}")

    print(f"Exported dynamic-batch UltraLight ONNX to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch UltraLight ONNX detector to dynamic batch.")
    parser.add_argument("--input", default="models/UltraLight/weights/version-slim-320.onnx")
    parser.add_argument("--output", default="models/UltraLight/weights/version-slim-320-dynamic.onnx")
    parser.add_argument("--verify-batch", type=int, default=2)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=240)
    args = parser.parse_args()
    export_dynamic_batch_onnx(args.input, args.output, args.verify_batch, args.input_width, args.input_height)


if __name__ == "__main__":
    main()
