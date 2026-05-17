from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RedisConfig:
    url: str = "redis://redis:6379/0"
    frame_queue: str = "attendance:frames"
    detection_queue: str = "attendance:detections"
    result_queue: str = "attendance:results"
    max_queue_size: int = 128
    consumer_group: str = "attendance-workers"
    consumer_name: str = socket.gethostname()
    stream_block_ms: int = 1000


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = "postgresql://camera_app:camera_app_dev_password@postgres:5432/camera_db"
    camera_poll_interval_sec: float = 10.0


@dataclass(frozen=True)
class CameraConfig:
    id: str = "camera-01"
    name: str = "Default camera"
    source: str = "0"
    enabled: bool = True
    sample_interval_ms: int = 500
    jpeg_quality: int = 85
    reconnect_delay_sec: float = 3.0


@dataclass(frozen=True)
class TritonConfig:
    url: str = "triton:8000"
    enabled: bool = True
    detector_model: str = "ultralight"
    arcface_model: str = "arcface"
    quality_model: str = "lightqnet"
    fasnet_v1_model: str = "fasnet_v1se"
    fasnet_v2_model: str = "fasnet_v2"
    timeout_sec: float = 5.0


@dataclass(frozen=True)
class DetectionConfig:
    provider: str = "triton_ultralight"
    input_width: int = 320
    input_height: int = 240
    iou_threshold: float = 0.4


@dataclass(frozen=True)
class TrackingConfig:
    track_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30


@dataclass(frozen=True)
class PipelineConfig:
    bucket_name: str = "Hust"
    use_voting: bool = True
    required_images: int = 4
    max_track_buffer: int = 10
    validation_threshold: float = 0.7
    bbox_threshold: float = 0.7
    min_face_area: float = 0.02
    qscore_threshold: float = 0.4
    anti_spoof_enabled: bool = False
    anti_spoof_threshold: float = 0.9
    distance_mode: str = "cosine"
    l2_threshold: float = 27.5
    cosine_threshold: float = 0.78
    redis: RedisConfig = RedisConfig()
    database: DatabaseConfig = DatabaseConfig()
    camera: CameraConfig = CameraConfig()
    cameras: tuple[CameraConfig, ...] = (CameraConfig(),)
    triton: TritonConfig = TritonConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()

    def get_camera(self, camera_id: str) -> CameraConfig:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        available = ", ".join(camera.id for camera in self.cameras) or "<none>"
        raise ValueError(f"Unknown camera_id={camera_id!r}. Available cameras: {available}")


def _get(mapping: dict[str, Any], path: str, default: Any) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _as_bool(value: Any) -> bool:
    return str(value).lower() not in {"0", "false", "no"}


def _load_camera_config(raw_camera: dict[str, Any], default_id: str = "camera-01") -> CameraConfig:
    return CameraConfig(
        id=str(raw_camera.get("id", default_id)),
        name=str(raw_camera.get("name", raw_camera.get("id", default_id))),
        source=str(raw_camera.get("source", CameraConfig.source)),
        enabled=_as_bool(raw_camera.get("enabled", CameraConfig.enabled)),
        sample_interval_ms=int(raw_camera.get("sample_interval_ms", CameraConfig.sample_interval_ms)),
        jpeg_quality=int(raw_camera.get("jpeg_quality", CameraConfig.jpeg_quality)),
        reconnect_delay_sec=float(raw_camera.get("reconnect_delay_sec", CameraConfig.reconnect_delay_sec)),
    )


def load_pipeline_config(path: str | Path = "config.yaml") -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    redis = raw.get("redis", {})
    database = raw.get("database", {})
    camera = raw.get("camera", {})
    cameras = raw.get("cameras", [])
    triton = raw.get("triton", {})
    detection = raw.get("detection", {})
    tracking = raw.get("tracking", {})
    infer_video = raw.get("infer_video", {})
    identity = raw.get("vector_search", raw.get("identity_person", {}))
    pipeline = raw.get("pipeline", {})

    redis_cfg = RedisConfig(
        url=os.getenv("REDIS_URL", redis.get("url", RedisConfig.url)),
        frame_queue=redis.get("frame_queue", RedisConfig.frame_queue),
        detection_queue=redis.get("detection_queue", RedisConfig.detection_queue),
        result_queue=redis.get("result_queue", RedisConfig.result_queue),
        max_queue_size=int(redis.get("max_queue_size", RedisConfig.max_queue_size)),
        consumer_group=os.getenv("REDIS_CONSUMER_GROUP", redis.get("consumer_group", RedisConfig.consumer_group)),
        consumer_name=os.getenv("REDIS_CONSUMER_NAME", redis.get("consumer_name", RedisConfig.consumer_name)),
        stream_block_ms=int(redis.get("stream_block_ms", RedisConfig.stream_block_ms)),
    )
    database_cfg = DatabaseConfig(
        url=os.getenv("DATABASE_URL", database.get("url", DatabaseConfig.url)),
        camera_poll_interval_sec=float(
            database.get("camera_poll_interval_sec", DatabaseConfig.camera_poll_interval_sec)
        ),
    )
    if cameras:
        camera_list = tuple(_load_camera_config(item, f"camera-{index + 1:02d}") for index, item in enumerate(cameras))
    else:
        camera_list = (_load_camera_config(camera),)

    camera_cfg = camera_list[0]
    triton_cfg = TritonConfig(
        url=os.getenv("TRITON_URL", triton.get("url", TritonConfig.url)),
        enabled=_as_bool(os.getenv("TRITON_ENABLED", triton.get("enabled", TritonConfig.enabled))),
        detector_model=triton.get("detector_model", TritonConfig.detector_model),
        arcface_model=triton.get("arcface_model", TritonConfig.arcface_model),
        quality_model=triton.get("quality_model", TritonConfig.quality_model),
        fasnet_v1_model=triton.get("fasnet_v1_model", TritonConfig.fasnet_v1_model),
        fasnet_v2_model=triton.get("fasnet_v2_model", TritonConfig.fasnet_v2_model),
        timeout_sec=float(triton.get("timeout_sec", TritonConfig.timeout_sec)),
    )
    detection_cfg = DetectionConfig(
        provider=detection.get("provider", DetectionConfig.provider),
        input_width=int(detection.get("input_width", DetectionConfig.input_width)),
        input_height=int(detection.get("input_height", DetectionConfig.input_height)),
        iou_threshold=float(detection.get("iou_threshold", DetectionConfig.iou_threshold)),
    )
    tracking_cfg = TrackingConfig(
        track_thresh=float(tracking.get("track_thresh", TrackingConfig.track_thresh)),
        match_thresh=float(tracking.get("match_thresh", TrackingConfig.match_thresh)),
        track_buffer=int(tracking.get("track_buffer", TrackingConfig.track_buffer)),
        frame_rate=int(tracking.get("frame_rate", TrackingConfig.frame_rate)),
    )

    return PipelineConfig(
        bucket_name=os.getenv("BUCKET_NAME", pipeline.get("bucket_name", PipelineConfig.bucket_name)),
        use_voting=_as_bool(infer_video.get("use_voting", PipelineConfig.use_voting)),
        required_images=int(_get(raw, "infer_video.required_images", PipelineConfig.required_images)),
        max_track_buffer=int(infer_video.get("max_track_buffer", PipelineConfig.max_track_buffer)),
        validation_threshold=float(infer_video.get("validation_threshold", PipelineConfig.validation_threshold)),
        bbox_threshold=float(infer_video.get("bbox_threshold", PipelineConfig.bbox_threshold)),
        min_face_area=float(infer_video.get("min_face_area", PipelineConfig.min_face_area)),
        qscore_threshold=float(infer_video.get("qscore_threshold", PipelineConfig.qscore_threshold)),
        anti_spoof_enabled=_as_bool(infer_video.get("is_anti_spoof", PipelineConfig.anti_spoof_enabled)),
        anti_spoof_threshold=float(infer_video.get("anti_spoof_threshold", PipelineConfig.anti_spoof_threshold)),
        distance_mode=identity.get("distance_mode", PipelineConfig.distance_mode),
        l2_threshold=float(identity.get("l2_threshold", PipelineConfig.l2_threshold)),
        cosine_threshold=float(identity.get("cosine_threshold", PipelineConfig.cosine_threshold)),
        redis=redis_cfg,
        database=database_cfg,
        camera=camera_cfg,
        cameras=camera_list,
        triton=triton_cfg,
        detection=detection_cfg,
        tracking=tracking_cfg,
    )
