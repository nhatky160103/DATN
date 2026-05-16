from __future__ import annotations

import base64
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, request

from .config import load_pipeline_config
from .orchestrator import RecognitionOrchestrator
from .response import ResponseWriter


def create_app() -> Flask:
    cfg = load_pipeline_config()
    app = Flask(__name__)
    responses = ResponseWriter(cfg.redis.url, cfg.redis.result_queue, cfg.redis.max_queue_size)
    orchestrator = RecognitionOrchestrator(cfg)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "bucket_name": cfg.bucket_name})

    @app.get("/results/latest")
    def latest_result():
        return jsonify(responses.latest() or {"status": "no_results"})

    @app.post("/recognize")
    def recognize():
        json_body = request.get_json(silent=True) if request.is_json else {}
        camera_id = request.form.get("camera_id") or json_body.get("camera_id", "api")
        frame_id = request.form.get("frame_id") or str(uuid.uuid4())

        if "image" in request.files:
            raw = request.files["image"].read()
        elif json_body.get("image_base64"):
            raw = base64.b64decode(json_body["image_base64"])
        else:
            return jsonify({"status": "error", "error": "missing image file or image_base64"}), 400

        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "error": "invalid image"}), 400

        result = orchestrator.recognize(
            frame,
            camera_id=camera_id,
            frame_id=frame_id,
            use_tracking=True,
            use_voting=False,
        )
        return jsonify(result)

    return app


app = create_app()
