from __future__ import annotations

from flask import Flask, jsonify

from .config import load_pipeline_config
from .response import ResponseWriter


def create_app() -> Flask:
    cfg = load_pipeline_config()
    app = Flask(__name__)
    responses = ResponseWriter(cfg.redis.url, cfg.redis.result_queue, cfg.redis.max_queue_size)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "bucket_name": cfg.bucket_name})

    @app.get("/results/latest")
    def latest_result():
        return jsonify(responses.latest() or {"status": "no_results"})

    return app


app = create_app()

