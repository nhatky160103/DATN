from __future__ import annotations

import base64
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from .config import load_pipeline_config
from .orchestrator import RecognitionOrchestrator
from .response import ResponseWriter


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attendance Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #687385;
      --line: #dfe5ec;
      --ok: #147a50;
      --warn: #9a5b00;
      --bad: #b42318;
      --chip: #edf2f7;
      --accent: #135f7a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }

    .page {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 32px;
    }

    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: 26px;
      font-weight: 700;
    }

    .subtle {
      color: var(--muted);
      font-size: 14px;
    }

    .status-bar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .chip {
      border: 1px solid var(--line);
      background: var(--chip);
      border-radius: 6px;
      padding: 6px 9px;
      font-size: 13px;
      white-space: nowrap;
    }

    .grid {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }

    .panel h2 {
      margin: 0 0 12px;
      font-size: 16px;
      font-weight: 700;
    }

    .latest {
      min-height: 230px;
      display: grid;
      gap: 14px;
    }

    .employee {
      font-size: 30px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 28px;
      border-radius: 6px;
      padding: 4px 9px;
      font-weight: 700;
      font-size: 13px;
      text-transform: uppercase;
      background: var(--chip);
    }

    .badge.recognized { background: #e2f7ed; color: var(--ok); }
    .badge.unknown { background: #fff3d6; color: var(--warn); }
    .badge.rejected, .badge.error { background: #fde7e3; color: var(--bad); }

    .facts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .fact {
      border-top: 1px solid var(--line);
      padding-top: 9px;
      min-width: 0;
    }

    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }

    .value {
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }

    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    td {
      overflow-wrap: anywhere;
    }

    tr:last-child td { border-bottom: 0; }

    .time { width: 116px; }
    .camera { width: 130px; }
    .score { width: 86px; }
    .empty {
      color: var(--muted);
      padding: 28px 0;
      text-align: center;
    }

    @media (max-width: 820px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      .status-bar {
        justify-content: flex-start;
      }

      .grid {
        grid-template-columns: 1fr;
      }

      .page {
        width: min(100% - 20px, 1180px);
        padding-top: 14px;
      }

      th, td {
        padding: 9px 6px;
        font-size: 13px;
      }

      .camera, .score {
        display: none;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div>
        <h1>Attendance Monitor</h1>
        <div class="subtle">Live recognition results from Redis stream</div>
      </div>
      <div class="status-bar">
        <span class="chip" id="api-status">API checking</span>
        <span class="chip" id="bucket">Bucket: {{ bucket_name }}</span>
        <span class="chip" id="total">Results: 0</span>
      </div>
    </header>

    <section class="grid">
      <section class="panel latest">
        <div>
          <h2>Latest Result</h2>
          <span class="badge" id="latest-status">waiting</span>
        </div>
        <div>
          <div class="label">Employee</div>
          <div class="employee" id="latest-employee">No results</div>
        </div>
        <div class="facts">
          <div class="fact">
            <div class="label">Camera</div>
            <div class="value" id="latest-camera">-</div>
          </div>
          <div class="fact">
            <div class="label">Score</div>
            <div class="value" id="latest-score">-</div>
          </div>
          <div class="fact">
            <div class="label">Quality</div>
            <div class="value" id="latest-quality">-</div>
          </div>
          <div class="fact">
            <div class="label">Track</div>
            <div class="value" id="latest-track">-</div>
          </div>
          <div class="fact">
            <div class="label">Time</div>
            <div class="value" id="latest-time">-</div>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>Recent Events</h2>
        <table>
          <thead>
            <tr>
              <th class="time">Time</th>
              <th class="camera">Camera</th>
              <th>Employee</th>
              <th>Status</th>
              <th class="score">Quality</th>
              <th class="score">Score</th>
            </tr>
          </thead>
          <tbody id="events">
            <tr><td colspan="6" class="empty">Waiting for results</td></tr>
          </tbody>
        </table>
      </section>
    </section>
  </main>

  <script>
    const statusClass = (status) => {
      if (status === "recognized") return "recognized";
      if (status === "unknown") return "unknown";
      if (status && (status.includes("rejected") || status.includes("error") || status.includes("too_small"))) {
        return "rejected";
      }
      return "";
    };

    const fmtTime = (seconds) => {
      if (!seconds) return "-";
      return new Date(seconds * 1000).toLocaleTimeString();
    };

    const fmtScore = (score) => {
      if (score === null || score === undefined) return "-";
      return Number(score).toFixed(3);
    };

    const esc = (value) => String(value ?? "-").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]));

    const cameraOf = (item) => item.metadata?.camera_id || "-";
    const qualityOf = (item) => item?.metadata?.quality_score;

    function renderLatest(item) {
      const badge = document.getElementById("latest-status");
      badge.textContent = item?.status || "waiting";
      badge.className = `badge ${statusClass(item?.status)}`;
      document.getElementById("latest-employee").textContent = item?.employee_id || "No results";
      document.getElementById("latest-camera").textContent = item ? cameraOf(item) : "-";
      document.getElementById("latest-score").textContent = item ? fmtScore(item.score) : "-";
      document.getElementById("latest-quality").textContent = item ? fmtScore(qualityOf(item)) : "-";
      document.getElementById("latest-track").textContent = item?.track_id ?? "-";
      document.getElementById("latest-time").textContent = item ? fmtTime(item.timestamp) : "-";
    }

    function renderRows(items) {
      const body = document.getElementById("events");
      if (!items.length) {
        body.innerHTML = '<tr><td colspan="6" class="empty">No recognition results yet</td></tr>';
        return;
      }

      body.innerHTML = items.map((item) => `
        <tr>
          <td class="time">${esc(fmtTime(item.timestamp))}</td>
          <td class="camera">${esc(cameraOf(item))}</td>
          <td>${esc(item.employee_id || "UNKNOWN")}</td>
          <td><span class="badge ${esc(statusClass(item.status))}">${esc(item.status || "-")}</span></td>
          <td class="score">${esc(fmtScore(qualityOf(item)))}</td>
          <td class="score">${esc(fmtScore(item.score))}</td>
        </tr>
      `).join("");
    }

    async function refresh() {
      try {
        const response = await fetch("/results/recent?count=20", { cache: "no-store" });
        const data = await response.json();
        const items = data.results || [];
        document.getElementById("api-status").textContent = "API online";
        document.getElementById("total").textContent = `Results: ${data.count || 0}`;
        renderLatest(items[0]);
        renderRows(items);
      } catch (error) {
        document.getElementById("api-status").textContent = "API error";
        document.getElementById("api-status").className = "chip";
      }
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


def create_app() -> Flask:
    cfg = load_pipeline_config()
    app = Flask(__name__)
    responses = ResponseWriter(cfg.redis.url, cfg.redis.result_queue, cfg.redis.max_queue_size)
    orchestrator = RecognitionOrchestrator(cfg)

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML, bucket_name=cfg.bucket_name)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "bucket_name": cfg.bucket_name})

    @app.get("/results/latest")
    def latest_result():
        return jsonify(responses.latest() or {"status": "no_results"})

    @app.get("/results/recent")
    def recent_results():
        count = request.args.get("count", default=20, type=int)
        count = min(max(count, 1), 100)
        results = responses.recent(count)
        return jsonify({"status": "ok", "count": len(results), "results": results})

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
