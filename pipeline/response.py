from __future__ import annotations

from .redis_queue import RedisStreamQueue
from .schemas import RecognitionResult, to_json


class ResponseWriter:
    def __init__(self, redis_url: str, result_queue: str, max_size: int = 128):
        self.queue = RedisStreamQueue(redis_url, result_queue, max_size)

    def write(self, result: RecognitionResult, update_firebase: bool = True) -> None:
        if update_firebase:
            try:
                from database.firebase import push_recognition_event

                push_recognition_event(
                    result.bucket_name,
                    {
                        "employee_id": result.employee_id,
                        "track_id": result.track_id,
                        "score": result.score,
                        "status": result.status,
                        "timestamp": result.timestamp,
                        "metadata": result.metadata,
                    },
                )
            except Exception as exc:
                result.metadata["firebase_error"] = str(exc)
        self.queue.push(to_json(result))

    def latest(self) -> dict | None:
        return self.queue.peek_latest()
