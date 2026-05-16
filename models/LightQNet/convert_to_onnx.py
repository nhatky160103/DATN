"""
Convert LightQNet TF1 frozen graph (.pb) → ONNX

Requirements:
    pip install tf2onnx tensorflow onnxruntime

Usage:
    python -m models.lightqnet.convert_to_onnx                         # dm050 (default)
    python -m models.lightqnet.convert_to_onnx --model lightqnet-dm100.pb
    python -m models.lightqnet.convert_to_onnx --all                   # convert all 3 variants
    python -m models.lightqnet.convert_to_onnx --verify                # convert + verify
"""

import os
import sys
import argparse
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_NODE  = "input:0"
OUTPUT_NODE = "confidence_st:0"
ALL_MODELS  = ["lightqnet-dm025.pb", "lightqnet-dm050.pb", "lightqnet-dm100.pb"]


def convert(pb_path: str, onnx_path: str):
    try:
        import tf2onnx.convert
    except ImportError:
        print("[ERROR] tf2onnx not installed. Run: pip install tf2onnx")
        sys.exit(1)

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    print(f"[convert] {os.path.basename(pb_path)}  →  {os.path.basename(onnx_path)}")

    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    inp_tensor = graph.get_tensor_by_name(INPUT_NODE)
    out_tensor = graph.get_tensor_by_name(OUTPUT_NODE)
    print(f"  Input shape  : {inp_tensor.shape}")
    print(f"  Output shape : {out_tensor.shape}")

    tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=[INPUT_NODE],
        output_names=[OUTPUT_NODE],
        opset=13,
        output_path=onnx_path,
    )

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ✓ Done — {size_mb:.2f} MB\n")


def verify(onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    print(f"[verify] {os.path.basename(onnx_path)}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"  Input  : name='{inp.name}'  shape={inp.shape}  type={inp.type}")
    print(f"  Output : name='{out.name}'  shape={out.shape}")

    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    dummy = np.random.rand(*shape).astype(np.float32)
    result = session.run(None, {inp.name: dummy})[0]
    print(f"  Output : {result}  shape={result.shape}")
    print(f"  ✓ ONNX model works correctly!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="lightqnet-dm050.pb", help=".pb filename")
    parser.add_argument("--all",    action="store_true", help="Convert all 3 variants")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX after convert")
    args = parser.parse_args()

    targets = ALL_MODELS if args.all else [args.model]

    for model_name in targets:
        pb_path   = os.path.join(CURRENT_DIR, model_name)
        onnx_path = pb_path.replace(".pb", ".onnx")

        if not os.path.exists(pb_path):
            print(f"[ERROR] File not found: {pb_path}")
            print(f"  → Download from: https://github.com/KaenChan/lightqnet/raw/master/{model_name}")
            continue

        convert(pb_path, onnx_path)

        if args.verify:
            verify(onnx_path)
