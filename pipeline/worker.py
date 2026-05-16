from __future__ import annotations

import argparse

import cv2
import numpy as np

from .config import PipelineConfig, load_pipeline_config
from .orchestrator import RecognitionOrchestrator
from .redis_queue import RedisListQueue
from .response import ResponseWriter
from .schemas import FrameMessage, from_json


class AttendancePipelineWorker:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.frame_queue = RedisListQueue(cfg.redis.url, cfg.redis.frame_queue, cfg.redis.max_queue_size)
        self.response_writer = ResponseWriter(cfg.redis.url, cfg.redis.result_queue, cfg.redis.max_queue_size)
        self.orchestrator = RecognitionOrchestrator(cfg)

    def run_forever(self) -> None:
        while True:
            payload = self.frame_queue.pop(timeout_sec=1)
            if payload is None:
                continue
            try:
                self.process_frame(from_json(payload, FrameMessage))
            except Exception as exc:
                print(f"Pipeline worker failed to process frame: {exc}")

    def process_frame(self, message: FrameMessage) -> None:
        frame_bytes = np.frombuffer(message.image_bytes(), dtype=np.uint8)
        frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return

        response = self.orchestrator.recognize(
            frame,
            camera_id=message.camera_id,
            frame_id=message.frame_id,
            use_tracking=True,
            use_voting=True,
        )
        for result in self.orchestrator.recognition_results(response):
            if result.status == "pending":
                continue
            self.response_writer.write(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    AttendancePipelineWorker(load_pipeline_config(args.config)).run_forever()


if __name__ == "__main__":
    main()
