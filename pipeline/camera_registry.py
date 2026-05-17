from __future__ import annotations

from typing import Protocol

from .config import CameraConfig, PipelineConfig


class CameraRegistry(Protocol):
    def list_enabled(self) -> tuple[CameraConfig, ...]:
        ...

    def get(self, camera_id: str) -> CameraConfig:
        ...

    def mark_online(self, camera_id: str) -> None:
        ...

    def mark_error(self, camera_id: str, error: str) -> None:
        ...


class YamlCameraRegistry:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def list_enabled(self) -> tuple[CameraConfig, ...]:
        return tuple(camera for camera in self.cfg.cameras if camera.enabled)

    def get(self, camera_id: str) -> CameraConfig:
        return self.cfg.get_camera(camera_id)

    def mark_online(self, camera_id: str) -> None:
        return None

    def mark_error(self, camera_id: str, error: str) -> None:
        return None


class PostgresCameraRegistry:
    def __init__(self, database_url: str):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("PostgreSQL camera registry requires the 'psycopg[binary]' package") from exc

        self._psycopg = psycopg
        self.database_url = database_url

    def _connect(self):
        return self._psycopg.connect(self.database_url)

    def list_enabled(self) -> tuple[CameraConfig, ...]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, rtsp_url, enabled, sample_interval_ms, jpeg_quality, reconnect_delay_sec
                    FROM cameras
                    WHERE enabled = TRUE
                    ORDER BY id
                    """
                )
                return tuple(self._row_to_camera(row) for row in cur.fetchall())

    def get(self, camera_id: str) -> CameraConfig:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, rtsp_url, enabled, sample_interval_ms, jpeg_quality, reconnect_delay_sec
                    FROM cameras
                    WHERE id = %s
                    """,
                    (camera_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise ValueError(f"Unknown camera_id={camera_id!r} in camera registry")
        return self._row_to_camera(row)

    def mark_online(self, camera_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE cameras
                    SET status = 'online', last_seen_at = now(), last_error = NULL, updated_at = now()
                    WHERE id = %s
                    """,
                    (camera_id,),
                )

    def mark_error(self, camera_id: str, error: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE cameras
                    SET status = 'error', last_error = %s, updated_at = now()
                    WHERE id = %s
                    """,
                    (error[:1000], camera_id),
                )

    @staticmethod
    def _row_to_camera(row: tuple) -> CameraConfig:
        camera_id, name, source, enabled, sample_interval_ms, jpeg_quality, reconnect_delay_sec = row
        return CameraConfig(
            id=str(camera_id),
            name=str(name),
            source=str(source),
            enabled=bool(enabled),
            sample_interval_ms=int(sample_interval_ms),
            jpeg_quality=int(jpeg_quality),
            reconnect_delay_sec=float(reconnect_delay_sec),
        )


def build_camera_registry(cfg: PipelineConfig, prefer_postgres: bool = True) -> CameraRegistry:
    if prefer_postgres and cfg.database.url:
        try:
            registry = PostgresCameraRegistry(cfg.database.url)
            cameras = registry.list_enabled()
        except Exception as exc:
            print(f"Cannot use PostgreSQL camera registry, falling back to config.yaml cameras: {exc}")
        else:
            if cameras:
                return registry
            print("PostgreSQL camera registry has no enabled cameras, falling back to config.yaml cameras")

    return YamlCameraRegistry(cfg)
