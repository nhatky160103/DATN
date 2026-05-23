from __future__ import annotations

from .event_store import PostgresEventStore
from .redis_queue import RedisStreamQueue
from .schemas import RecognitionResult, to_json


class ResponseWriter:
    def __init__(
        self,
        redis_url: str,
        result_queue: str,
        max_size: int = 128,
        database_url: str | None = None,
    ):
        self.queue = RedisStreamQueue(redis_url, result_queue, max_size)
        self.event_store = None
        if database_url:
            try:
                self.event_store = PostgresEventStore(database_url)
            except Exception as exc:
                print(f"Cannot initialize Postgres attendance event store: {exc}")

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
        if self.event_store is not None:
            try:
                self.event_store.write(result)
            except Exception as exc:
                result.metadata["postgres_error"] = str(exc)
        self.queue.push(to_json(result))

    def latest(self) -> dict | None:
        return self.queue.peek_latest()

    def recent(self, count: int = 20) -> list[dict]:
        return self.queue.recent(count)
