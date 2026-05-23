from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from pipeline.camera_registry import build_camera_registry
from pipeline.config import load_pipeline_config
from pipeline.stages.detection import FaceDetectionStage
from pipeline.triton_client import TritonInferenceClient


def _camera_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def _draw_status(frame, text: str) -> None:
    cv2.rectangle(frame, (8, 8), (520, 42), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def _draw_detection(frame, bbox: list[int], score: float) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{score:.3f}  {x2 - x1}x{y2 - y1}"
    cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + 150, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def _draw_crop_box(frame, bbox: list[int] | None) -> None:
    if not bbox:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview UltraLight detector realtime with pipeline Triton model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera-id", default="camera-01")
    parser.add_argument("--source", default="", help="Override camera source: RTSP URL, video path, or camera index.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--crop-margin", type=float, default=None)
    parser.add_argument("--crop-margin-x", type=float, default=None)
    parser.add_argument("--crop-margin-y", type=float, default=None)
    parser.add_argument("--triton-url", default="")
    parser.add_argument("--window", default="UltraLight Detector Preview")
    parser.add_argument("--snapshot-dir", default="", help="Save frames with boxes here instead of opening a GUI window.")
    parser.add_argument("--save-crops", action="store_true", help="Also save expanded detection crops.")
    parser.add_argument("--reconnect-delay-sec", type=float, default=2.0)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    source = args.source
    if not source:
        registry = build_camera_registry(cfg)
        source = registry.get(args.camera_id).source

    triton = TritonInferenceClient(args.triton_url or cfg.triton.url, cfg.triton.timeout_sec, cfg.triton.enabled)
    detector = FaceDetectionStage(
        triton,
        model_name=cfg.triton.detector_model,
        threshold=cfg.bbox_threshold if args.threshold is None else args.threshold,
        iou_threshold=cfg.detection.iou_threshold,
        input_width=cfg.detection.input_width,
        input_height=cfg.detection.input_height,
        crop_margin=cfg.detection.crop_margin if args.crop_margin is None else args.crop_margin,
        crop_margin_x=cfg.detection.crop_margin_x if args.crop_margin_x is None else args.crop_margin_x,
        crop_margin_y=cfg.detection.crop_margin_y if args.crop_margin_y is None else args.crop_margin_y,
    )

    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    source_value = _camera_source(source)
    frame_index = 0

    while True:
        cap = cv2.VideoCapture(source_value)
        if not cap.isOpened():
            print(f"Cannot open source: {source}. Retrying...")
            cap.release()
            time.sleep(args.reconnect_delay_sec)
            continue

        print("Opened source. Press q or Esc to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Cannot read frame. Reconnecting...")
                break

            started = time.perf_counter()
            detections = detector.predict(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            vis = frame.copy()
            for detection in detections:
                _draw_detection(vis, detection.bbox, detection.score)

            fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
            _draw_status(
                vis,
                f"faces={len(detections)}  infer={elapsed_ms:.1f}ms  fps={fps:.1f}  threshold={detector.threshold:.2f}",
            )

            if snapshot_dir:
                output_path = snapshot_dir / f"detector_{frame_index:06d}.jpg"
                cv2.imwrite(str(output_path), vis)
                if args.save_crops:
                    for index, detection in enumerate(detections):
                        x1, y1, x2, y2 = detection.bbox
                        crop = frame[y1:y2, x1:x2]
                        cv2.imwrite(str(snapshot_dir / f"crop_{frame_index:06d}_{index}.jpg"), crop)
                if frame_index % 30 == 0:
                    print(f"Saved {output_path}")
                frame_index += 1
                continue

            try:
                cv2.imshow(args.window, vis)
            except cv2.error as exc:
                output_path = Path("detector_preview_snapshot.jpg")
                cv2.imwrite(str(output_path), vis)
                print(f"Cannot open GUI preview window: {exc}")
                print(f"Saved one frame instead: {output_path}")
                cap.release()
                return

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        time.sleep(args.reconnect_delay_sec)


if __name__ == "__main__":
    main()
