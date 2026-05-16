from __future__ import annotations

import argparse
import time
import uuid

import cv2

from .config import load_pipeline_config
from .redis_queue import RedisStreamQueue
from .schemas import FrameMessage, to_json


def _camera_source(raw: str) -> str | int:
    return int(raw) if raw.isdigit() else raw


def run_frame_reader(config_path: str = "config.yaml", camera_id: str = "camera-01") -> None:
    cfg = load_pipeline_config(config_path)
    queue = RedisStreamQueue(cfg.redis.url, cfg.redis.frame_queue, cfg.redis.max_queue_size)
    interval_sec = cfg.camera.sample_interval_ms / 1000.0

    while True:
        cap = cv2.VideoCapture(_camera_source(cfg.camera.source))
        if not cap.isOpened():
            print(f"Cannot open camera source: {cfg.camera.source}. Retrying...")
            time.sleep(cfg.camera.reconnect_delay_sec)
            continue

        last_emit = 0.0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if now - last_emit < interval_sec:
                continue
            last_emit = now

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), cfg.camera.jpeg_quality],
            )
            if not ok:
                continue

            message = FrameMessage.from_jpeg(camera_id, str(uuid.uuid4()), encoded.tobytes())
            queue.push(to_json(message))

        cap.release()
        time.sleep(cfg.camera.reconnect_delay_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera-id", default="camera-01")
    args = parser.parse_args()
    run_frame_reader(args.config, args.camera_id)


if __name__ == "__main__":
    main()
