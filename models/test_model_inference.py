from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:
    np = None


def _status(name: str, ok: bool, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {"name": name, "ok": ok, "detail": detail}
    if extra:
        result.update(extra)
    print(json.dumps(result, ensure_ascii=False))
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=1, keepdims=True)


def test_onnx_model(
    name: str,
    model_path: str,
    dummy: np.ndarray,
    expected_last_dim: int | None = None,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    if np is None:
        return _status(name, False, "numpy import failed")

    path = Path(model_path)
    if not path.exists():
        return _status(name, False, f"missing model file: {path}")

    try:
        import onnxruntime as ort
    except Exception as exc:
        return _status(name, False, f"onnxruntime import failed: {exc}")

    try:
        session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()
        started = time.perf_counter()
        outputs = session.run(None, {input_meta.name: dummy.astype(np.float32)})
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        shapes = [list(output.shape) for output in outputs]
        if expected_last_dim is not None and outputs[0].reshape(outputs[0].shape[0], -1).shape[-1] != expected_last_dim:
            return _status(
                name,
                False,
                f"unexpected output shape: {shapes}",
                {"input_name": input_meta.name, "outputs": [out.name for out in output_meta]},
            )

        return _status(
            name,
            True,
            "onnx load + inference ok",
            {
                "model_path": str(path),
                "input_name": input_meta.name,
                "input_shape": list(dummy.shape),
                "output_names": [out.name for out in output_meta],
                "output_shapes": shapes,
                "elapsed_ms": round(elapsed_ms, 3),
            },
        )
    except Exception as exc:
        return _status(name, False, f"onnx inference failed: {exc}", {"model_path": str(path)})


def test_arcface_onnx(path: str) -> dict[str, Any]:
    if np is None:
        return _status("arcface_r18_onnx", False, "numpy import failed")
    dummy = np.random.uniform(-1.0, 1.0, size=(1, 3, 112, 112)).astype(np.float32)
    return test_onnx_model("arcface_r18_onnx", path, dummy, expected_last_dim=512)


def test_fasnet_onnx(name: str, path: str) -> dict[str, Any]:
    if np is None:
        return _status(name, False, "numpy import failed")
    dummy = np.random.uniform(0.0, 255.0, size=(1, 3, 80, 80)).astype(np.float32)
    result = test_onnx_model(name, path, dummy, expected_last_dim=3)
    if result["ok"]:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            logits = session.run(None, {input_name: dummy})[0]
            probs = _softmax(logits.reshape(1, -1))
            result["probabilities"] = probs.reshape(-1).round(6).tolist()
            result["predicted_label"] = int(np.argmax(probs))
        except Exception as exc:
            result["postprocess_error"] = str(exc)
    return result


def test_lightqnet_onnx(path: str) -> dict[str, Any]:
    if np is None:
        return _status("lightqnet_onnx", False, "numpy import failed")
    dummy = np.random.uniform(-1.0, 1.0, size=(1, 96, 96, 3)).astype(np.float32)
    return test_onnx_model("lightqnet_onnx", path, dummy, expected_last_dim=1)


def test_lightqnet_graphdef(path: str) -> dict[str, Any]:
    if np is None:
        return _status("lightqnet_graphdef", False, "numpy import failed")

    graph_path = Path(path)
    if not graph_path.exists():
        return _status("lightqnet_graphdef", False, f"missing graphdef file: {graph_path}")

    try:
        import tensorflow.compat.v1 as tf

        tf.disable_eager_execution()
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_path.read_bytes())
            tf.import_graph_def(graph_def, name="")
            image_ph = graph.get_tensor_by_name("input:0")
            score = graph.get_tensor_by_name("confidence_st:0")

        dummy = np.random.uniform(-1.0, 1.0, size=(1, 96, 96, 3)).astype(np.float32)
        started = time.perf_counter()
        with tf.Session(graph=graph) as session:
            output = session.run(score, feed_dict={image_ph: dummy})
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return _status(
            "lightqnet_graphdef",
            True,
            "graphdef load + inference ok",
            {"model_path": str(graph_path), "output_shape": list(output.shape), "elapsed_ms": round(elapsed_ms, 3)},
        )
    except Exception as exc:
        return _status("lightqnet_graphdef", False, f"graphdef inference failed: {exc}", {"model_path": str(graph_path)})


def test_mtcnn(image_path: str | None, max_side: int, min_face_size: int, factor: float, torch_num_threads: int) -> dict[str, Any]:
    if np is None:
        return _status("mtcnn_pytorch", False, "numpy import failed", {"onnx": False})

    try:
        import cv2
        import numpy as np

        from attendance_pipeline.stages.detection import MTCNNFaceDetector
    except Exception as exc:
        return _status("mtcnn_pytorch", False, f"imports failed: {exc}")

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            return _status("mtcnn_pytorch", False, f"cannot read image: {image_path}")
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        detector = MTCNNFaceDetector(
            threshold=0.7,
            max_side=max_side,
            min_face_size=min_face_size,
            factor=factor,
            keep_all=True,
            torch_num_threads=torch_num_threads,
        )
        started = time.perf_counter()
        detections = detector.detect(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _status(
            "mtcnn_pytorch",
            True,
            "mtcnn load + detect ok",
            {"detections": len(detections), "elapsed_ms": round(elapsed_ms, 3), "onnx": False},
        )
    except Exception as exc:
        return _status("mtcnn_pytorch", False, f"mtcnn failed: {exc}", {"onnx": False})


def test_mtcnn_onnx(
    onnx_dir: str,
    image_path: str | None,
    min_face_size: int,
    factor: float,
    intra_op_num_threads: int,
) -> dict[str, Any]:
    if np is None:
        return _status("mtcnn_onnx", False, "numpy import failed", {"onnx": True})

    try:
        from PIL import Image

        from models.Detection.onnx_runtime import ONNXMTCNN
    except Exception as exc:
        return _status("mtcnn_onnx", False, f"imports failed: {exc}", {"onnx": True})

    path = Path(onnx_dir)
    missing = [name for name in ("pnet.onnx", "rnet.onnx", "onet.onnx") if not (path / name).exists()]
    if missing:
        return _status("mtcnn_onnx", False, f"missing ONNX files in {path}: {missing}", {"onnx": True})

    try:
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))

        detector = ONNXMTCNN(
            onnx_dir=path,
            keep_all=True,
            min_face_size=min_face_size,
            factor=factor,
            post_process=False,
            providers=["CPUExecutionProvider"],
            intra_op_num_threads=intra_op_num_threads,
        )
        started = time.perf_counter()
        boxes, probs = detector.detect(image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detections = 0 if boxes is None else len(boxes)
        scores = [] if probs is None or probs[0] is None else np.asarray(probs, dtype=np.float32).round(6).tolist()
        return _status(
            "mtcnn_onnx",
            True,
            "onnx mtcnn load + detect ok",
            {"detections": detections, "scores": scores, "elapsed_ms": round(elapsed_ms, 3), "onnx": True},
        )
    except Exception as exc:
        return _status("mtcnn_onnx", False, f"onnx mtcnn failed: {exc}", {"onnx": True})


def test_triton_model(
    name: str,
    triton_url: str,
    model_name: str,
    input_name: str,
    dummy: np.ndarray,
    output_names: list[str],
) -> dict[str, Any]:
    if np is None:
        return _status(name, False, "numpy import failed")

    try:
        import tritonclient.http as httpclient
    except Exception as exc:
        return _status(name, False, f"tritonclient import failed: {exc}")

    try:
        client = httpclient.InferenceServerClient(url=triton_url)
        infer_input = httpclient.InferInput(input_name, dummy.shape, "FP32")
        infer_input.set_data_from_numpy(dummy.astype(np.float32))
        outputs = [httpclient.InferRequestedOutput(output_name) for output_name in output_names]
        started = time.perf_counter()
        result = client.infer(model_name, [infer_input], outputs=outputs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        arrays = [result.as_numpy(output_name) for output_name in output_names]
        return _status(
            name,
            True,
            "triton inference ok",
            {
                "triton_url": triton_url,
                "model_name": model_name,
                "input_shape": list(dummy.shape),
                "output_shapes": [list(array.shape) for array in arrays],
                "elapsed_ms": round(elapsed_ms, 3),
            },
        )
    except Exception as exc:
        return _status(name, False, f"triton inference failed: {exc}", {"triton_url": triton_url, "model_name": model_name})


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test local ONNX/GraphDef/MTCNN and optional Triton inference.")
    parser.add_argument("--arcface-onnx", default="triton_model_repository/arcface/1/model.onnx")
    parser.add_argument("--lightqnet-onnx", default="triton_model_repository/lightqnet/1/model.onnx")
    parser.add_argument("--lightqnet-pb", default="triton_model_repository/lightqnet/1/model.graphdef")
    parser.add_argument("--fasnet-v1-onnx", default="triton_model_repository/fasnet_v1se/1/model.onnx")
    parser.add_argument("--fasnet-v2-onnx", default="triton_model_repository/fasnet_v2/1/model.onnx")
    parser.add_argument("--mtcnn-image", default="image_resources/mtcnn_test.png")
    parser.add_argument("--mtcnn-max-side", type=int, default=640)
    parser.add_argument("--mtcnn-min-face-size", type=int, default=40)
    parser.add_argument("--mtcnn-factor", type=float, default=0.8)
    parser.add_argument("--mtcnn-onnx-dir", default="models/Detection/onnx")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--onnxruntime-num-threads", type=int, default=2)
    parser.add_argument("--triton-url", default=None)
    parser.add_argument("--skip-mtcnn", action="store_true")
    parser.add_argument("--skip-mtcnn-onnx", action="store_true")
    args = parser.parse_args()

    results = [
        test_arcface_onnx(args.arcface_onnx),
        test_lightqnet_onnx(args.lightqnet_onnx),
        test_lightqnet_graphdef(args.lightqnet_pb),
        test_fasnet_onnx("fasnet_v1se_onnx", args.fasnet_v1_onnx),
        test_fasnet_onnx("fasnet_v2_onnx", args.fasnet_v2_onnx),
    ]
    if not args.skip_mtcnn:
        results.append(
            test_mtcnn(
                args.mtcnn_image,
                args.mtcnn_max_side,
                args.mtcnn_min_face_size,
                args.mtcnn_factor,
                args.torch_num_threads,
            )
        )
    if not args.skip_mtcnn_onnx:
        results.append(
            test_mtcnn_onnx(
                args.mtcnn_onnx_dir,
                args.mtcnn_image,
                args.mtcnn_min_face_size,
                args.mtcnn_factor,
                args.onnxruntime_num_threads,
            )
        )

    if args.triton_url:
        results.extend(
            [
                test_triton_model(
                    "triton_arcface",
                    args.triton_url,
                    "arcface",
                    "input",
                    np.random.uniform(-1.0, 1.0, size=(1, 3, 112, 112)).astype(np.float32),
                    ["embedding"],
                ),
                test_triton_model(
                    "triton_lightqnet",
                    args.triton_url,
                    "lightqnet",
                    "input",
                    np.random.uniform(-1.0, 1.0, size=(1, 96, 96, 3)).astype(np.float32),
                    ["confidence_st"],
                ),
                test_triton_model(
                    "triton_fasnet_v1se",
                    args.triton_url,
                    "fasnet_v1se",
                    "input",
                    np.random.uniform(0.0, 255.0, size=(1, 3, 80, 80)).astype(np.float32),
                    ["logits"],
                ),
                test_triton_model(
                    "triton_fasnet_v2",
                    args.triton_url,
                    "fasnet_v2",
                    "input",
                    np.random.uniform(0.0, 255.0, size=(1, 3, 80, 80)).astype(np.float32),
                    ["logits"],
                ),
            ]
        )

    failed = [result for result in results if not result["ok"]]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
