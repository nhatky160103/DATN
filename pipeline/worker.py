from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from .config import PipelineConfig, load_pipeline_config
from .orchestrator import RecognitionOrchestrator
from .redis_queue import RedisStreamQueue
from .response import ResponseWriter
from .schemas import FrameMessage, from_json


class AttendancePipelineWorker:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.frame_queue = RedisStreamQueue(
            cfg.redis.url,
            cfg.redis.frame_queue,
            cfg.redis.max_queue_size,
            group=cfg.redis.consumer_group,
            consumer=cfg.redis.consumer_name,
        )
        self.response_writer = ResponseWriter(
            cfg.redis.url,
            cfg.redis.result_queue,
            cfg.redis.max_queue_size,
            database_url=cfg.database.url,
        )
        self.orchestrator = RecognitionOrchestrator(cfg)

    def run_forever(self) -> None:
        while True:
            try:
                message = self.frame_queue.pop(timeout_ms=self.cfg.redis.stream_block_ms)
                if message is None:
                    continue
                self.process_frame(from_json(message.payload, FrameMessage))
                self.frame_queue.ack(message.stream_id)
            except Exception as exc:
                print(f"Pipeline worker failed to process frame: {exc}")
                time.sleep(1.0)

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
