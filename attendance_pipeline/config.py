from __future__ import annotations

import os
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


@dataclass(frozen=True)
class CameraConfig:
    source: str = "0"
    sample_interval_ms: int = 500
    jpeg_quality: int = 85
    reconnect_delay_sec: float = 3.0


@dataclass(frozen=True)
class TritonConfig:
    url: str = "triton:8000"
    enabled: bool = True
    arcface_model: str = "arcface"
    quality_model: str = "lightqnet"
    fasnet_v1_model: str = "fasnet_v1se"
    fasnet_v2_model: str = "fasnet_v2"
    timeout_sec: float = 5.0


@dataclass(frozen=True)
class DetectionConfig:
    provider: str = "python_mtcnn"
    mtcnn_max_side: int = 640
    mtcnn_min_face_size: int = 40
    mtcnn_factor: float = 0.8
    mtcnn_keep_all: bool = True
    torch_num_threads: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    bucket_name: str = "Hust"
    required_images: int = 4
    validation_threshold: float = 0.7
    bbox_threshold: float = 0.7
    min_face_area: float = 0.1
    qscore_threshold: float = 0.4
    anti_spoof_enabled: bool = False
    anti_spoof_threshold: float = 0.9
    distance_mode: str = "cosine"
    l2_threshold: float = 27.5
    cosine_threshold: float = 0.78
    redis: RedisConfig = RedisConfig()
    camera: CameraConfig = CameraConfig()
    triton: TritonConfig = TritonConfig()
    detection: DetectionConfig = DetectionConfig()


def _get(mapping: dict[str, Any], path: str, default: Any) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def load_pipeline_config(path: str | Path = "config.yaml") -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    redis = raw.get("redis", {})
    camera = raw.get("camera", {})
    triton = raw.get("triton", {})
    detection = raw.get("detection", {})
    infer_video = raw.get("infer_video", {})
    identity = raw.get("vector_search", raw.get("identity_person", {}))
    pipeline = raw.get("pipeline", {})

    redis_cfg = RedisConfig(
        url=os.getenv("REDIS_URL", redis.get("url", RedisConfig.url)),
        frame_queue=redis.get("frame_queue", RedisConfig.frame_queue),
        detection_queue=redis.get("detection_queue", RedisConfig.detection_queue),
        result_queue=redis.get("result_queue", RedisConfig.result_queue),
        max_queue_size=int(redis.get("max_queue_size", RedisConfig.max_queue_size)),
    )
    camera_cfg = CameraConfig(
        source=os.getenv("CAMERA_SOURCE", str(camera.get("source", CameraConfig.source))),
        sample_interval_ms=int(camera.get("sample_interval_ms", CameraConfig.sample_interval_ms)),
        jpeg_quality=int(camera.get("jpeg_quality", CameraConfig.jpeg_quality)),
        reconnect_delay_sec=float(camera.get("reconnect_delay_sec", CameraConfig.reconnect_delay_sec)),
    )
    triton_cfg = TritonConfig(
        url=os.getenv("TRITON_URL", triton.get("url", TritonConfig.url)),
        enabled=str(os.getenv("TRITON_ENABLED", triton.get("enabled", TritonConfig.enabled))).lower()
        not in {"0", "false", "no"},
        arcface_model=triton.get("arcface_model", TritonConfig.arcface_model),
        quality_model=triton.get("quality_model", TritonConfig.quality_model),
        fasnet_v1_model=triton.get("fasnet_v1_model", TritonConfig.fasnet_v1_model),
        fasnet_v2_model=triton.get("fasnet_v2_model", TritonConfig.fasnet_v2_model),
        timeout_sec=float(triton.get("timeout_sec", TritonConfig.timeout_sec)),
    )
    detection_cfg = DetectionConfig(
        provider=detection.get("provider", DetectionConfig.provider),
        mtcnn_max_side=int(detection.get("mtcnn_max_side", DetectionConfig.mtcnn_max_side)),
        mtcnn_min_face_size=int(detection.get("mtcnn_min_face_size", DetectionConfig.mtcnn_min_face_size)),
        mtcnn_factor=float(detection.get("mtcnn_factor", DetectionConfig.mtcnn_factor)),
        mtcnn_keep_all=bool(detection.get("mtcnn_keep_all", DetectionConfig.mtcnn_keep_all)),
        torch_num_threads=int(detection.get("torch_num_threads", DetectionConfig.torch_num_threads)),
    )

    return PipelineConfig(
        bucket_name=os.getenv("BUCKET_NAME", pipeline.get("bucket_name", PipelineConfig.bucket_name)),
        required_images=int(_get(raw, "infer_video.required_images", PipelineConfig.required_images)),
        validation_threshold=float(infer_video.get("validation_threshold", PipelineConfig.validation_threshold)),
        bbox_threshold=float(infer_video.get("bbox_threshold", PipelineConfig.bbox_threshold)),
        min_face_area=float(infer_video.get("min_face_area", PipelineConfig.min_face_area)),
        qscore_threshold=float(infer_video.get("qscore_threshold", PipelineConfig.qscore_threshold)),
        anti_spoof_enabled=bool(infer_video.get("is_anti_spoof", PipelineConfig.anti_spoof_enabled)),
        anti_spoof_threshold=float(infer_video.get("anti_spoof_threshold", PipelineConfig.anti_spoof_threshold)),
        distance_mode=identity.get("distance_mode", PipelineConfig.distance_mode),
        l2_threshold=float(identity.get("l2_threshold", PipelineConfig.l2_threshold)),
        cosine_threshold=float(identity.get("cosine_threshold", PipelineConfig.cosine_threshold)),
        redis=redis_cfg,
        camera=camera_cfg,
        triton=triton_cfg,
        detection=detection_cfg,
    )
