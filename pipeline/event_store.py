from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schemas import RecognitionResult


class PostgresEventStore:
    def __init__(self, database_url: str):
        try:
            import psycopg
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise RuntimeError("PostgresEventStore requires the 'psycopg[binary]' package") from exc

        self._psycopg = psycopg
        self._jsonb = Jsonb
        self.database_url = database_url
        self._ensure_table()

    def _connect(self):
        return self._psycopg.connect(self.database_url)

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS attendance_events (
                        id BIGSERIAL PRIMARY KEY,
                        bucket_name TEXT NOT NULL,
                        employee_id TEXT NOT NULL,
                        camera_id TEXT,
                        track_id INTEGER,
                        frame_id TEXT,
                        status TEXT NOT NULL,
                        score DOUBLE PRECISION,
                        quality_score DOUBLE PRECISION,
                        det_score DOUBLE PRECISION,
                        liveness_score DOUBLE PRECISION,
                        bbox JSONB,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_attendance_events_occurred_at
                    ON attendance_events (occurred_at DESC)
                    """
                )

    def write(self, result: RecognitionResult) -> None:
        metadata = dict(result.metadata or {})
        occurred_at = datetime.fromtimestamp(float(result.timestamp), tz=timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO attendance_events (
                        bucket_name,
                        employee_id,
                        camera_id,
                        track_id,
                        frame_id,
                        status,
                        score,
                        quality_score,
                        det_score,
                        liveness_score,
                        bbox,
                        metadata,
                        occurred_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result.bucket_name,
                        result.employee_id,
                        metadata.get("camera_id"),
                        result.track_id,
                        metadata.get("frame_id"),
                        result.status,
                        result.score,
                        metadata.get("quality_score"),
                        metadata.get("det_score"),
                        metadata.get("liveness_score"),
                        self._jsonb(metadata.get("bbox")) if metadata.get("bbox") is not None else None,
                        self._jsonb(metadata),
                        occurred_at,
                    ),
                )

    @staticmethod
    def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        (
            event_id,
            bucket_name,
            employee_id,
            camera_id,
            track_id,
            frame_id,
            status,
            score,
            quality_score,
            det_score,
            liveness_score,
            bbox,
            metadata,
            occurred_at,
        ) = row
        metadata = dict(metadata or {})
        metadata.setdefault("camera_id", camera_id)
        metadata.setdefault("frame_id", frame_id)
        metadata.setdefault("bbox", bbox)
        metadata.setdefault("quality_score", quality_score)
        metadata.setdefault("det_score", det_score)
        metadata.setdefault("liveness_score", liveness_score)
        return {
            "id": event_id,
            "bucket_name": bucket_name,
            "employee_id": employee_id,
            "track_id": track_id,
            "score": score,
            "status": status,
            "timestamp": occurred_at.timestamp(),
            "metadata": metadata,
        }

    def recent(self, count: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        bucket_name,
                        employee_id,
                        camera_id,
                        track_id,
                        frame_id,
                        status,
                        score,
                        quality_score,
                        det_score,
                        liveness_score,
                        bbox,
                        metadata,
                        occurred_at
                    FROM attendance_events
                    ORDER BY occurred_at DESC, id DESC
                    LIMIT %s
                    """,
                    (int(count),),
                )
                return [self._row_to_dict(row) for row in cur.fetchall()]

    def latest(self) -> dict[str, Any] | None:
        rows = self.recent(1)
        return rows[0] if rows else None
