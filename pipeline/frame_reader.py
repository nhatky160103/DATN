from __future__ import annotations

import argparse
import threading
import time
import uuid

import cv2
import numpy as np

from .camera_registry import build_camera_registry
from .config import CameraConfig, load_pipeline_config
from .redis_queue import RedisStreamQueue
from .schemas import FrameMessage, to_json


def _camera_source(raw: str) -> str | int:
    return int(raw) if raw.isdigit() else raw


def _gstreamer_pipeline(source: str) -> str:
    if source.isdigit():
        return (
            f"v4l2src device=/dev/video{source} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
        )
    if source.startswith("rtsp://"):
        return (
            f"rtspsrc location={source} latency=200 protocols=tcp ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
        )
    return source


class GStreamerCapture:
    def __init__(self, source: str):
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            raise RuntimeError(f"GStreamer Python bindings are unavailable: {exc}") from exc

        self.Gst = Gst
        Gst.init(None)
        self.pipeline = Gst.parse_launch(_gstreamer_pipeline(source))
        self.sink = self.pipeline.get_by_name("sink")
        if self.sink is None:
            raise RuntimeError("GStreamer pipeline does not contain an appsink named 'sink'")
        self.pipeline.set_state(Gst.State.PLAYING)

    def isOpened(self) -> bool:
        state = self.pipeline.get_state(0).state
        return state in {self.Gst.State.PLAYING, self.Gst.State.PAUSED}

    def read(self) -> tuple[bool, np.ndarray | None]:
        sample = self.sink.emit("try-pull-sample", 1_000_000_000)
        if sample is None:
            return False, None

        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        buffer = sample.get_buffer()
        ok, map_info = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(map_info)
        return True, frame

    def release(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)


def _open_capture(camera: CameraConfig, use_gstreamer: bool) -> cv2.VideoCapture:
    if use_gstreamer:
        try:
            cap = GStreamerCapture(camera.source)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception as exc:
            print(f"Cannot open camera {camera.id} with GStreamer, falling back to OpenCV: {exc}")

    return cv2.VideoCapture(_camera_source(camera.source))


def _read_camera(
    camera: CameraConfig,
    queue: RedisStreamQueue,
    stop_event: threading.Event,
    registry,
    use_gstreamer: bool,
) -> None:
    if not camera.enabled:
        raise ValueError(f"Camera {camera.id!r} is disabled")

    interval_sec = camera.sample_interval_ms / 1000.0

    while not stop_event.is_set():
        cap = _open_capture(camera, use_gstreamer)
        if not cap.isOpened():
            error = f"Cannot open camera {camera.id} source: {camera.source}"
            print(f"{error}. Retrying...")
            registry.mark_error(camera.id, error)
            stop_event.wait(camera.reconnect_delay_sec)
            continue

        registry.mark_online(camera.id)
        print(f"Opened camera {camera.id}: {camera.name}")
        last_emit = 0.0
        while cap.isOpened() and not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                registry.mark_error(camera.id, "Cannot read frame")
                break

            now = time.time()
            if now - last_emit < interval_sec:
                continue
            last_emit = now

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), camera.jpeg_quality],
            )
            if not ok:
                continue

            message = FrameMessage.from_jpeg(camera.id, str(uuid.uuid4()), encoded.tobytes())
            queue.push(to_json(message))
            registry.mark_online(camera.id)

        cap.release()
        stop_event.wait(camera.reconnect_delay_sec)


def run_frame_reader(
    config_path: str = "config.yaml",
    camera_id: str = "camera-01",
    use_gstreamer: bool = True,
) -> None:
    cfg = load_pipeline_config(config_path)
    registry = build_camera_registry(cfg)
    camera = registry.get(camera_id)
    queue = RedisStreamQueue(cfg.redis.url, cfg.redis.frame_queue, cfg.redis.max_queue_size)
    _read_camera(camera, queue, threading.Event(), registry, use_gstreamer)


def run_camera_agent(config_path: str = "config.yaml", use_gstreamer: bool = True) -> None:
    cfg = load_pipeline_config(config_path)
    registry = build_camera_registry(cfg)
    queue = RedisStreamQueue(cfg.redis.url, cfg.redis.frame_queue, cfg.redis.max_queue_size)
    workers: dict[str, tuple[threading.Thread, threading.Event]] = {}

    while True:
        cameras = {camera.id: camera for camera in registry.list_enabled()}

        for camera_id, (thread, stop_event) in list(workers.items()):
            if camera_id not in cameras or not thread.is_alive():
                stop_event.set()
                workers.pop(camera_id, None)

        for camera_id, camera in cameras.items():
            if camera_id in workers:
                continue
            stop_event = threading.Event()
            thread = threading.Thread(
                target=_read_camera,
                args=(camera, queue, stop_event, registry, use_gstreamer),
                name=f"frame-reader-{camera_id}",
                daemon=True,
            )
            thread.start()
            workers[camera_id] = (thread, stop_event)
            print(f"Started camera reader: {camera_id}")

        time.sleep(cfg.database.camera_poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera-id", default="camera-01")
    parser.add_argument("--all-enabled", action="store_true")
    parser.add_argument("--no-gstreamer", action="store_true")
    args = parser.parse_args()
    if args.all_enabled:
        run_camera_agent(args.config, use_gstreamer=not args.no_gstreamer)
    else:
        run_frame_reader(args.config, args.camera_id, use_gstreamer=not args.no_gstreamer)


if __name__ == "__main__":
    main()
