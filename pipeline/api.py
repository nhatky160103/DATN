from __future__ import annotations

import base64
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

from .config import load_pipeline_config
from .enroll_qdrant_identity_store import SUPPORTED_EXTS, enroll_dataset
from .orchestrator import RecognitionOrchestrator
from .response import ResponseWriter

DEFAULT_UPLOAD_DATASET_ROOT = "FacenetDataset"


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


ADMIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attendance Admin</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d9e0e7;
      --ok: #087443;
      --warn: #946200;
      --bad: #b42318;
      --accent: #0f5f73;
      --soft: #eef3f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 14px;
    }
    .page {
      width: min(1440px, calc(100% - 28px));
      margin: 0 auto;
      padding: 18px 0 28px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }
    h1 { margin: 0; font-size: 24px; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .chip {
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 6px;
      padding: 6px 9px;
      white-space: nowrap;
    }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }
    .tab-button {
      border-color: var(--line);
      background: transparent;
      color: var(--muted);
      border-bottom-left-radius: 0;
      border-bottom-right-radius: 0;
      margin-bottom: -1px;
    }
    .tab-button.active {
      border-bottom-color: var(--panel);
      background: var(--panel);
      color: var(--accent);
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .layout {
      display: grid;
      grid-template-columns: minmax(360px, 0.9fr) minmax(520px, 1.4fr);
      gap: 14px;
      align-items: start;
    }
    .stack { display: grid; gap: 14px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .grid-form {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .grid-form.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      color: var(--text);
      background: white;
      font: inherit;
      min-height: 36px;
    }
    button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 8px 10px;
      font-weight: 700;
      cursor: pointer;
      min-height: 36px;
    }
    button.secondary {
      background: white;
      color: var(--accent);
    }
    button.danger {
      border-color: var(--bad);
      background: var(--bad);
    }
    button:disabled {
      opacity: 0.55;
      cursor: default;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 7px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    tr:last-child td { border-bottom: 0; }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 7px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      background: var(--soft);
    }
    .badge.online, .badge.recognized { color: var(--ok); background: #e5f6ee; }
    .badge.error, .badge.rejected { color: var(--bad); background: #fde8e5; }
    .badge.unknown { color: var(--warn); background: #fff3d6; }
    .muted { color: var(--muted); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }
    .metric .value { font-size: 20px; font-weight: 700; margin-top: 3px; }
    .actions { display: flex; gap: 6px; flex-wrap: wrap; }
    .table-scroll { overflow-x: auto; }
    .table-scroll.bounded {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .identity-scroll { max-height: 360px; }
    .events-scroll { max-height: 430px; }
    .table-scroll.bounded table {
      border-collapse: separate;
      border-spacing: 0;
    }
    .table-scroll.bounded thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--panel);
      box-shadow: inset 0 -1px 0 var(--line);
    }
    .message { min-height: 20px; color: var(--muted); margin-top: 8px; }
    .form-block {
      border-bottom: 1px solid var(--line);
      margin-bottom: 14px;
      padding-bottom: 14px;
    }
    @media (max-width: 980px) {
      header { align-items: flex-start; flex-direction: column; }
      .chips { justify-content: flex-start; }
      .layout { grid-template-columns: 1fr; }
      .grid-form, .grid-form.three, .metrics { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div>
        <h1>Attendance Admin</h1>
        <div class="muted">Bucket {{ bucket_name }}</div>
      </div>
      <div class="chips">
        <span class="chip" id="api-chip">API checking</span>
        <span class="chip" id="event-chip">Events 0</span>
        <span class="chip" id="identity-chip">Identities 0</span>
      </div>
    </header>

    <nav class="tabs" aria-label="Admin sections">
      <button type="button" class="tab-button active" data-tab-target="overview" aria-selected="true">Overview</button>
      <button type="button" class="tab-button" data-tab-target="cameras-tab" aria-selected="false">Cameras</button>
      <button type="button" class="tab-button" data-tab-target="identities-tab" aria-selected="false">Identities</button>
      <button type="button" class="tab-button" data-tab-target="settings-tab" aria-selected="false">Settings</button>
    </nav>

    <section id="overview" class="tab-panel active">
      <div class="stack">
        <section class="panel">
          <div class="metrics">
            <div class="metric"><div class="muted">Cameras</div><div class="value" id="m-cameras">-</div></div>
            <div class="metric"><div class="muted">Online</div><div class="value" id="m-online">-</div></div>
            <div class="metric"><div class="muted">Qdrant Points</div><div class="value" id="m-points">-</div></div>
            <div class="metric"><div class="muted">Redis Frames</div><div class="value" id="m-frames">-</div></div>
          </div>
        </section>

        <section class="panel">
          <div class="toolbar">
            <h2>Recent Events</h2>
            <button class="secondary" onclick="refreshEvents()">Refresh</button>
          </div>
          <div class="table-scroll bounded events-scroll">
            <table>
              <thead><tr><th>Time</th><th>Camera</th><th>Employee</th><th>Status</th><th>Quality</th><th>Score</th></tr></thead>
              <tbody id="events"><tr><td colspan="6" class="muted">Loading</td></tr></tbody>
            </table>
          </div>
        </section>
      </div>
    </section>

    <section id="cameras-tab" class="tab-panel">
      <section class="panel">
        <div class="toolbar">
          <h2>Cameras</h2>
          <button class="secondary" onclick="refreshCameras()">Refresh</button>
        </div>
        <form id="camera-form" class="grid-form">
          <label>ID<input name="id" placeholder="camera-02" required></label>
          <label>Name<input name="name" placeholder="Laptop camera"></label>
          <label style="grid-column:1/-1">Source<input name="source" placeholder="rtsp://... or 0" required></label>
          <label>Rotate<input name="rotate" type="number" value="0" step="90"></label>
          <label>Enabled
            <select name="enabled"><option value="true">true</option><option value="false">false</option></select>
          </label>
          <div class="actions" style="grid-column:1/-1">
            <button type="submit">Add Camera</button>
          </div>
        </form>
        <div class="message" id="camera-message"></div>
        <div class="table-scroll">
          <table>
            <thead><tr><th>ID</th><th>Status</th><th>Source</th><th>Actions</th></tr></thead>
            <tbody id="cameras"><tr><td colspan="4" class="muted">Loading</td></tr></tbody>
          </table>
        </div>
      </section>
    </section>

    <section id="identities-tab" class="tab-panel">
      <section class="panel">
        <div class="toolbar">
          <h2>Identities</h2>
          <button class="secondary" onclick="refreshIdentities()">Refresh</button>
        </div>
        <form id="upload-enroll-form" class="grid-form form-block" enctype="multipart/form-data">
          <label>Employee ID<input name="employee_id" placeholder="employee-01" required></label>
          <label>Images<input name="images" type="file" accept="image/*" multiple required></label>
          <label>Dataset Root<input name="dataset_root" value="FacenetDataset"></label>
          <label>Min Quality<input name="min_quality" type="number" step="0.01" placeholder="0.4"></label>
          <div class="actions" style="grid-column:1/-1"><button type="submit">Upload & Enroll</button></div>
        </form>
        <form id="enroll-form" class="grid-form">
          <label style="grid-column:1/-1">Dataset Root<input name="dataset_root" placeholder="FacenetDataset" required></label>
          <label>Min Quality<input name="min_quality" type="number" step="0.01" placeholder="0.4"></label>
          <div class="actions"><button type="submit">Enroll</button></div>
        </form>
        <div class="message" id="identity-message"></div>
        <div class="table-scroll bounded identity-scroll">
          <table>
            <thead><tr><th>Employee</th><th>Points</th><th>Active</th><th>Actions</th></tr></thead>
            <tbody id="identities"><tr><td colspan="4" class="muted">Loading</td></tr></tbody>
          </table>
        </div>
      </section>
    </section>

    <section id="settings-tab" class="tab-panel">
      <section class="panel">
        <div class="toolbar">
          <h2>Thresholds</h2>
          <button class="secondary" onclick="refreshSettings()">Reload</button>
        </div>
        <form id="settings-form" class="grid-form three">
          <label>Match Threshold<input name="qdrant.match_threshold" type="number" step="0.01"></label>
          <label>Quality Threshold<input name="infer_video.qscore_threshold" type="number" step="0.01"></label>
          <label>BBox Threshold<input name="infer_video.bbox_threshold" type="number" step="0.01"></label>
          <label>Validation Threshold<input name="infer_video.validation_threshold" type="number" step="0.01"></label>
          <label>Required Images<input name="infer_video.required_images" type="number" step="1"></label>
          <label>Max Track Buffer<input name="infer_video.max_track_buffer" type="number" step="1"></label>
          <label>Min Face Area<input name="infer_video.min_face_area" type="number" step="0.001"></label>
          <label>Crop Margin X<input name="detection.crop_margin_x" type="number" step="0.01"></label>
          <label>Crop Margin Y<input name="detection.crop_margin_y" type="number" step="0.01"></label>
          <label>Track Threshold<input name="tracking.track_thresh" type="number" step="0.01"></label>
          <label>Track Match<input name="tracking.match_thresh" type="number" step="0.01"></label>
          <label>Track Buffer<input name="tracking.track_buffer" type="number" step="1"></label>
          <div class="actions" style="grid-column:1/-1"><button type="submit">Save Settings</button></div>
        </form>
        <div class="message" id="settings-message"></div>
      </section>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const esc = (value) => String(value ?? "-").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[char]));
    const score = (value) => value === null || value === undefined ? "-" : Number(value).toFixed(3);
    const time = (seconds) => seconds ? new Date(seconds * 1000).toLocaleTimeString() : "-";
    const badgeClass = (status) => {
      if (status === "online" || status === "recognized") return "online";
      if (status === "unknown") return "unknown";
      if (status && (status.includes("error") || status.includes("rejected") || status.includes("too_small"))) return "error";
      return "";
    };
    async function api(path, options) {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok || data.status === "error") throw new Error(data.error || response.statusText);
      return data;
    }
    document.querySelectorAll("[data-tab-target]").forEach((button) => {
      button.addEventListener("click", () => {
        const targetId = button.dataset.tabTarget;
        document.querySelectorAll("[data-tab-target]").forEach((item) => {
          const active = item === button;
          item.classList.toggle("active", active);
          item.setAttribute("aria-selected", active ? "true" : "false");
        });
        document.querySelectorAll(".tab-panel").forEach((panel) => {
          panel.classList.toggle("active", panel.id === targetId);
        });
      });
    });
    function nestFromForm(form) {
      const out = {};
      for (const [name, raw] of new FormData(form).entries()) {
        if (raw === "") continue;
        const parts = name.split(".");
        const key = parts.pop();
        let target = out;
        for (const part of parts) target = target[part] ||= {};
        const value = raw === "true" ? true : raw === "false" ? false : Number.isNaN(Number(raw)) ? raw : Number(raw);
        target[key] = value;
      }
      return out;
    }
    async function refreshSystem() {
      try {
        const data = await api("/system/status");
        const cameras = data.cameras || [];
        $("api-chip").textContent = "API online";
        $("m-cameras").textContent = cameras.length;
        $("m-online").textContent = cameras.filter((item) => item.status === "online").length;
        $("m-points").textContent = data.qdrant?.points_count ?? "-";
        $("m-frames").textContent = data.redis?.frame_queue ?? "-";
      } catch (error) {
        $("api-chip").textContent = "API error";
      }
    }
    async function refreshEvents() {
      const data = await api("/results/recent?count=30");
      const items = data.results || [];
      $("event-chip").textContent = `Events ${data.count || 0}`;
      $("events").innerHTML = items.length ? items.map((item) => `
        <tr>
          <td>${esc(time(item.timestamp))}</td>
          <td>${esc(item.metadata?.camera_id)}</td>
          <td>${esc(item.employee_id)}</td>
          <td><span class="badge ${badgeClass(item.status)}">${esc(item.status)}</span></td>
          <td>${esc(score(item.metadata?.quality_score))}</td>
          <td>${esc(score(item.score))}</td>
        </tr>`).join("") : '<tr><td colspan="6" class="muted">No events</td></tr>';
    }
    async function refreshCameras() {
      const data = await api("/cameras");
      const cameras = data.cameras || [];
      $("cameras").innerHTML = cameras.length ? cameras.map((camera) => `
        <tr>
          <td>${esc(camera.id)}<br><span class="muted">${esc(camera.name)}</span></td>
          <td><span class="badge ${badgeClass(camera.status)}">${esc(camera.status)}</span><br><span class="muted">${esc(camera.last_error || "")}</span></td>
          <td>${esc(camera.source)}</td>
          <td><div class="actions">
            <button class="secondary" data-action="toggle-camera" data-camera-id="${esc(camera.id)}" data-enabled="${camera.enabled ? "false" : "true"}">${camera.enabled ? "Disable" : "Enable"}</button>
            <button class="danger" data-action="delete-camera" data-camera-id="${esc(camera.id)}">Delete</button>
          </div></td>
        </tr>`).join("") : '<tr><td colspan="4" class="muted">No cameras</td></tr>';
    }
    async function refreshIdentities() {
      const data = await api("/identities");
      const identities = data.identities || [];
      $("identity-chip").textContent = `Identities ${data.count || 0}`;
      $("identities").innerHTML = identities.length ? identities.map((identity) => `
        <tr>
          <td>${esc(identity.employee_id)}<br><span class="muted">${esc(identity.sample_image_path)}</span></td>
          <td>${identity.points}</td>
          <td>${identity.active_points}</td>
          <td><div class="actions">
            <button class="secondary" data-action="activate-identity" data-employee-id="${esc(identity.employee_id)}">Activate</button>
            <button class="secondary" data-action="deactivate-identity" data-employee-id="${esc(identity.employee_id)}">Deactivate</button>
            <button class="danger" data-action="delete-identity" data-employee-id="${esc(identity.employee_id)}">Delete</button>
          </div></td>
        </tr>`).join("") : '<tr><td colspan="4" class="muted">No identities</td></tr>';
    }
    function setSettingsForm(settings) {
      const flat = {
        "qdrant.match_threshold": settings.qdrant?.match_threshold,
        "infer_video.qscore_threshold": settings.infer_video?.qscore_threshold,
        "infer_video.bbox_threshold": settings.infer_video?.bbox_threshold,
        "infer_video.validation_threshold": settings.infer_video?.validation_threshold,
        "infer_video.required_images": settings.infer_video?.required_images,
        "infer_video.max_track_buffer": settings.infer_video?.max_track_buffer,
        "infer_video.min_face_area": settings.infer_video?.min_face_area,
        "detection.crop_margin_x": settings.detection?.crop_margin_x,
        "detection.crop_margin_y": settings.detection?.crop_margin_y,
        "tracking.track_thresh": settings.tracking?.track_thresh,
        "tracking.match_thresh": settings.tracking?.match_thresh,
        "tracking.track_buffer": settings.tracking?.track_buffer,
      };
      for (const [name, value] of Object.entries(flat)) {
        const input = document.querySelector(`[name="${name}"]`);
        if (input && value !== null && value !== undefined) input.value = value;
      }
    }
    async function refreshSettings() {
      const data = await api("/settings");
      setSettingsForm(data.settings || {});
    }
    async function toggleCamera(id, enabled) {
      await api(`/cameras/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({enabled})
      });
      refreshCameras(); refreshSystem();
    }
    async function deleteCamera(id) {
      if (!confirm(`Delete camera ${id}?`)) return;
      await api(`/cameras/${encodeURIComponent(id)}`, {method: "DELETE"});
      refreshCameras(); refreshSystem();
    }
    async function setIdentity(id, action) {
      await api(`/identities/${encodeURIComponent(id)}/${action}`, {method: "POST"});
      refreshIdentities();
    }
    async function deleteIdentity(id) {
      if (!confirm(`Delete identity ${id}?`)) return;
      await api(`/identities/${encodeURIComponent(id)}`, {method: "DELETE"});
      refreshIdentities(); refreshSystem();
    }
    $("cameras").addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      if (button.dataset.action === "toggle-camera") {
        await toggleCamera(button.dataset.cameraId, button.dataset.enabled === "true");
      }
      if (button.dataset.action === "delete-camera") {
        await deleteCamera(button.dataset.cameraId);
      }
    });
    $("identities").addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      if (button.dataset.action === "activate-identity") {
        await setIdentity(button.dataset.employeeId, "activate");
      }
      if (button.dataset.action === "deactivate-identity") {
        await setIdentity(button.dataset.employeeId, "deactivate");
      }
      if (button.dataset.action === "delete-identity") {
        await deleteIdentity(button.dataset.employeeId);
      }
    });
    $("camera-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const body = nestFromForm(form);
      body.metadata = {rotate: Number(form.rotate.value || 0)};
      try {
        await api("/cameras", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        $("camera-message").textContent = "Camera saved";
        form.reset();
        refreshCameras(); refreshSystem();
      } catch (error) {
        $("camera-message").textContent = error.message;
      }
    });
    $("enroll-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const body = nestFromForm(event.currentTarget);
      $("identity-message").textContent = "Enroll running";
      try {
        const data = await api("/identities/enroll", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        $("identity-message").textContent = `Enroll started: ${data.job?.id || "-"}`;
        refreshIdentities(); refreshSystem();
      } catch (error) {
        $("identity-message").textContent = error.message;
      }
    });
    $("upload-enroll-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const body = new FormData(form);
      $("identity-message").textContent = "Upload running";
      try {
        const data = await api("/identities/enroll/upload", {method: "POST", body});
        $("identity-message").textContent = `Upload saved ${data.saved_images || 0} image(s), enroll started: ${data.job?.id || "-"}`;
        form.reset();
        form.elements.dataset_root.value = "FacenetDataset";
        refreshIdentities(); refreshSystem();
      } catch (error) {
        $("identity-message").textContent = error.message;
      }
    });
    $("settings-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        const body = nestFromForm(event.currentTarget);
        await api("/settings", {method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        $("settings-message").textContent = "Settings saved. Restart worker/api to apply.";
      } catch (error) {
        $("settings-message").textContent = error.message;
      }
    });
    async function refreshAll() {
      await Promise.allSettled([refreshSystem(), refreshEvents(), refreshCameras(), refreshIdentities(), refreshSettings()]);
    }
    refreshAll();
    setInterval(() => { refreshSystem(); refreshEvents(); }, 1500);
  </script>
</body>
</html>
"""


def create_app() -> Flask:
    cfg = load_pipeline_config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
    responses = ResponseWriter(
        cfg.redis.url,
        cfg.redis.result_queue,
        cfg.redis.max_queue_size,
        database_url=cfg.database.url,
    )
    orchestrator = RecognitionOrchestrator(cfg)
    enroll_jobs: dict[str, dict] = {}

    def upload_dataset_root(raw_root: str | None) -> Path:
        root_value = str(raw_root or DEFAULT_UPLOAD_DATASET_ROOT).strip() or DEFAULT_UPLOAD_DATASET_ROOT
        root = Path(root_value)
        root = root.resolve() if root.is_absolute() else (Path.cwd() / root).resolve()
        workspace = Path.cwd().resolve()
        if not root.is_relative_to(workspace):
            raise ValueError("dataset_root must be inside the project workspace")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def employee_upload_dir(root: Path, employee_id: str) -> Path:
        employee_id = str(employee_id or "").strip()
        if not employee_id:
            raise ValueError("employee_id is required")
        if employee_id in {".", ".."} or "/" in employee_id or "\\" in employee_id:
            raise ValueError("employee_id must be a single folder name")
        target = (root / employee_id).resolve()
        if not target.is_relative_to(root):
            raise ValueError("employee_id points outside dataset_root")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_upload_images(dataset_root: Path, employee_id: str, files) -> list[Path]:
        employee_dir = employee_upload_dir(dataset_root, employee_id)
        saved_paths: list[Path] = []
        for upload in files:
            original_name = upload.filename or ""
            if not original_name:
                continue
            suffix = Path(original_name).suffix.lower()
            if suffix not in SUPPORTED_EXTS:
                raise ValueError(f"Unsupported image extension: {original_name}")
            filename = secure_filename(original_name)
            if not filename:
                filename = f"image{suffix}"
            output_path = employee_dir / f"{uuid.uuid4().hex}_{filename}"
            upload.save(output_path)
            saved_paths.append(output_path)
        if not saved_paths:
            raise ValueError("at least one image file is required")
        return saved_paths

    def db_connect():
        import psycopg

        return psycopg.connect(cfg.database.url)

    def camera_row(row):
        (
            camera_id,
            name,
            rtsp_url,
            enabled,
            location,
            sample_interval_ms,
            jpeg_quality,
            reconnect_delay_sec,
            status,
            last_seen_at,
            last_error,
            metadata,
            created_at,
            updated_at,
        ) = row
        return {
            "id": camera_id,
            "name": name,
            "source": rtsp_url,
            "enabled": enabled,
            "location": location,
            "sample_interval_ms": sample_interval_ms,
            "jpeg_quality": jpeg_quality,
            "reconnect_delay_sec": reconnect_delay_sec,
            "status": status,
            "last_seen_at": None if last_seen_at is None else last_seen_at.isoformat(),
            "last_error": last_error,
            "metadata": metadata or {},
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
        }

    def fetch_cameras():
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        name,
                        rtsp_url,
                        enabled,
                        location,
                        sample_interval_ms,
                        jpeg_quality,
                        reconnect_delay_sec,
                        status,
                        last_seen_at,
                        last_error,
                        metadata,
                        created_at,
                        updated_at
                    FROM cameras
                    ORDER BY id
                    """
                )
                return [camera_row(row) for row in cur.fetchall()]

    def qdrant_client():
        from qdrant_client import QdrantClient

        api_key = cfg.qdrant.api_key or None
        return QdrantClient(url=cfg.qdrant.url, api_key=api_key, timeout=10.0)

    def qdrant_scroll(client, scroll_filter=None, limit: int = 256, offset=None):
        result = client.scroll(
            collection_name=cfg.qdrant.collection,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if isinstance(result, tuple):
            return result
        return result.points, result.next_page_offset

    def load_raw_settings() -> dict:
        with open("config.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def write_raw_settings(data: dict) -> None:
        with open("config.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)

    def current_settings() -> dict:
        current = load_pipeline_config("config.yaml")
        return {
            "pipeline": {
                "bucket_name": current.bucket_name,
            },
            "infer_video": {
                "use_voting": current.use_voting,
                "min_face_area": current.min_face_area,
                "bbox_threshold": current.bbox_threshold,
                "required_images": current.required_images,
                "max_track_buffer": current.max_track_buffer,
                "validation_threshold": current.validation_threshold,
                "is_anti_spoof": current.anti_spoof_enabled,
                "anti_spoof_threshold": current.anti_spoof_threshold,
                "qscore_threshold": current.qscore_threshold,
            },
            "qdrant": {
                "url": current.qdrant.url,
                "collection": current.qdrant.collection,
                "match_threshold": current.match_threshold,
            },
            "detection": {
                "input_width": current.detection.input_width,
                "input_height": current.detection.input_height,
                "iou_threshold": current.detection.iou_threshold,
                "crop_margin": current.detection.crop_margin,
                "crop_margin_x": current.detection.crop_margin_x,
                "crop_margin_y": current.detection.crop_margin_y,
            },
            "tracking": {
                "track_thresh": current.tracking.track_thresh,
                "match_thresh": current.tracking.match_thresh,
                "track_buffer": current.tracking.track_buffer,
                "frame_rate": current.tracking.frame_rate,
            },
        }

    def set_nested(raw: dict, section: str, key: str, value):
        raw.setdefault(section, {})[key] = value

    def as_api_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

    def update_settings_file(body: dict) -> dict:
        raw = load_raw_settings()
        allowed = {
            "pipeline": {"bucket_name": str},
            "infer_video": {
                "use_voting": bool,
                "min_face_area": float,
                "bbox_threshold": float,
                "required_images": int,
                "max_track_buffer": int,
                "validation_threshold": float,
                "is_anti_spoof": bool,
                "anti_spoof_threshold": float,
                "qscore_threshold": float,
            },
            "qdrant": {
                "match_threshold": float,
            },
            "detection": {
                "input_width": int,
                "input_height": int,
                "iou_threshold": float,
                "crop_margin": float,
                "crop_margin_x": float,
                "crop_margin_y": float,
            },
            "tracking": {
                "track_thresh": float,
                "match_thresh": float,
                "track_buffer": int,
                "frame_rate": int,
            },
        }
        for section, keys in allowed.items():
            incoming = body.get(section)
            if not isinstance(incoming, dict):
                continue
            for key, caster in keys.items():
                if key not in incoming:
                    continue
                value = incoming[key]
                if caster is bool:
                    value = as_api_bool(value)
                else:
                    value = caster(value)
                set_nested(raw, section, key, value)
        write_raw_settings(raw)
        return current_settings()

    @app.get("/")
    def index():
        return render_template_string(ADMIN_HTML, bucket_name=cfg.bucket_name)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "bucket_name": cfg.bucket_name})

    @app.get("/system/status")
    def system_status():
        status = {
            "status": "ok",
            "bucket_name": cfg.bucket_name,
            "config_path": str(Path("config.yaml").resolve()),
        }
        try:
            status["cameras"] = fetch_cameras()
        except Exception as exc:
            status["camera_error"] = str(exc)
        try:
            info = qdrant_client().get_collection(cfg.qdrant.collection)
            status["qdrant"] = {
                "collection": cfg.qdrant.collection,
                "points_count": getattr(info, "points_count", None),
                "status": str(getattr(info, "status", "")),
            }
        except Exception as exc:
            status["qdrant_error"] = str(exc)
        try:
            status["redis"] = {
                "frame_queue": responses.queue.client.xlen(cfg.redis.frame_queue),
                "result_queue": responses.queue.client.xlen(cfg.redis.result_queue),
            }
        except Exception as exc:
            status["redis_error"] = str(exc)
        return jsonify(status)

    @app.get("/settings")
    def get_settings():
        return jsonify({"status": "ok", "settings": current_settings()})

    @app.patch("/settings")
    def patch_settings():
        body = request.get_json(silent=True) or {}
        try:
            settings = update_settings_file(body)
        except Exception as exc:
            return jsonify({"status": "error", "error": str(exc)}), 400
        return jsonify({"status": "ok", "settings": settings, "restart_required": ["worker", "api"]})

    @app.get("/results/latest")
    def latest_result():
        if responses.event_store is not None:
            try:
                return jsonify(responses.event_store.latest() or {"status": "no_results"})
            except Exception:
                pass
        return jsonify(responses.latest() or {"status": "no_results"})

    @app.get("/results/recent")
    def recent_results():
        count = request.args.get("count", default=20, type=int)
        count = min(max(count, 1), 100)
        if responses.event_store is not None:
            try:
                results = responses.event_store.recent(count)
                return jsonify({"status": "ok", "source": "postgres", "count": len(results), "results": results})
            except Exception:
                pass
        results = responses.recent(count)
        return jsonify({"status": "ok", "source": "redis", "count": len(results), "results": results})

    @app.get("/cameras")
    def list_cameras():
        return jsonify({"status": "ok", "cameras": fetch_cameras()})

    @app.post("/cameras")
    def create_camera():
        body = request.get_json(silent=True) or {}
        camera_id = str(body.get("id") or "").strip()
        name = str(body.get("name") or camera_id).strip()
        source = str(body.get("source") or body.get("rtsp_url") or "").strip()
        if not camera_id or not source:
            return jsonify({"status": "error", "error": "id and source are required"}), 400

        from psycopg.types.json import Jsonb

        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cameras (
                        id,
                        name,
                        rtsp_url,
                        enabled,
                        location,
                        sample_interval_ms,
                        jpeg_quality,
                        reconnect_delay_sec,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING
                        id,
                        name,
                        rtsp_url,
                        enabled,
                        location,
                        sample_interval_ms,
                        jpeg_quality,
                        reconnect_delay_sec,
                        status,
                        last_seen_at,
                        last_error,
                        metadata,
                        created_at,
                        updated_at
                    """,
                    (
                        camera_id,
                        name,
                        source,
                        as_api_bool(body.get("enabled", True)),
                        body.get("location"),
                        int(body.get("sample_interval_ms", 500)),
                        int(body.get("jpeg_quality", 85)),
                        float(body.get("reconnect_delay_sec", 3)),
                        Jsonb(body.get("metadata") or {}),
                    ),
                )
                camera = camera_row(cur.fetchone())
        return jsonify({"status": "ok", "camera": camera}), 201

    @app.patch("/cameras/<camera_id>")
    def update_camera(camera_id: str):
        body = request.get_json(silent=True) or {}
        allowed = {
            "name": "name",
            "source": "rtsp_url",
            "rtsp_url": "rtsp_url",
            "enabled": "enabled",
            "location": "location",
            "sample_interval_ms": "sample_interval_ms",
            "jpeg_quality": "jpeg_quality",
            "reconnect_delay_sec": "reconnect_delay_sec",
            "metadata": "metadata",
        }
        values = []
        assignments = []
        from psycopg.types.json import Jsonb

        for key, column in allowed.items():
            if key not in body:
                continue
            value = body[key]
            if column == "metadata":
                value = Jsonb(value or {})
            if column == "enabled":
                value = as_api_bool(value)
            assignments.append(f"{column} = %s")
            values.append(value)
        if not assignments:
            return jsonify({"status": "error", "error": "no camera fields to update"}), 400

        values.append(camera_id)
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE cameras
                    SET {", ".join(assignments)}, updated_at = now()
                    WHERE id = %s
                    RETURNING
                        id,
                        name,
                        rtsp_url,
                        enabled,
                        location,
                        sample_interval_ms,
                        jpeg_quality,
                        reconnect_delay_sec,
                        status,
                        last_seen_at,
                        last_error,
                        metadata,
                        created_at,
                        updated_at
                    """,
                    tuple(values),
                )
                row = cur.fetchone()
        if row is None:
            return jsonify({"status": "error", "error": "camera not found"}), 404
        return jsonify({"status": "ok", "camera": camera_row(row)})

    @app.delete("/cameras/<camera_id>")
    def delete_camera(camera_id: str):
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM cameras WHERE id = %s RETURNING id", (camera_id,))
                row = cur.fetchone()
        if row is None:
            return jsonify({"status": "error", "error": "camera not found"}), 404
        return jsonify({"status": "ok", "deleted": camera_id})

    @app.get("/identities")
    def list_identities():
        from qdrant_client import models

        client = qdrant_client()
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(key="bucket_name", match=models.MatchValue(value=cfg.bucket_name)),
            ]
        )
        identities = {}
        offset = None
        while True:
            points, offset = qdrant_scroll(client, scroll_filter=scroll_filter, offset=offset)
            for point in points:
                payload = point.payload or {}
                employee_id = str(payload.get("employee_id") or "")
                if not employee_id:
                    continue
                item = identities.setdefault(
                    employee_id,
                    {
                        "employee_id": employee_id,
                        "bucket_name": payload.get("bucket_name"),
                        "points": 0,
                        "active_points": 0,
                        "sample_image_path": payload.get("image_path"),
                    },
                )
                item["points"] += 1
                if payload.get("active") is True:
                    item["active_points"] += 1
                if not item.get("sample_image_path"):
                    item["sample_image_path"] = payload.get("image_path")
            if offset is None:
                break
        data = sorted(identities.values(), key=lambda item: item["employee_id"])
        return jsonify({"status": "ok", "count": len(data), "identities": data})

    @app.post("/identities/enroll")
    def enroll_identities():
        body = request.get_json(silent=True) or {}
        dataset_root = body.get("dataset_root")
        if not dataset_root:
            return jsonify({"status": "error", "error": "dataset_root is required"}), 400
        min_quality = body.get("min_quality")
        job_id = str(uuid.uuid4())
        enroll_jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "dataset_root": str(dataset_root),
            "error": None,
        }

        def run_job() -> None:
            try:
                enroll_dataset("config.yaml", str(dataset_root), None if min_quality is None else float(min_quality))
            except Exception as exc:
                enroll_jobs[job_id]["status"] = "error"
                enroll_jobs[job_id]["error"] = str(exc)
            else:
                enroll_jobs[job_id]["status"] = "complete"

        threading.Thread(target=run_job, name=f"enroll-{job_id}", daemon=True).start()
        return jsonify({"status": "ok", "job": enroll_jobs[job_id]}), 202

    @app.post("/identities/enroll/upload")
    def upload_and_enroll_identity():
        employee_id = str(request.form.get("employee_id") or "").strip()
        min_quality_raw = request.form.get("min_quality")
        try:
            min_quality = None if min_quality_raw in {None, ""} else float(min_quality_raw)
            dataset_root = upload_dataset_root(request.form.get("dataset_root"))
            saved_paths = save_upload_images(dataset_root, employee_id, request.files.getlist("images"))
        except Exception as exc:
            return jsonify({"status": "error", "error": str(exc)}), 400

        job_id = str(uuid.uuid4())
        enroll_jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "dataset_root": dataset_root.as_posix(),
            "employee_id": employee_id,
            "saved_images": len(saved_paths),
            "error": None,
        }

        def run_job() -> None:
            try:
                enroll_dataset("config.yaml", dataset_root.as_posix(), min_quality, employee_ids=[employee_id])
            except Exception as exc:
                enroll_jobs[job_id]["status"] = "error"
                enroll_jobs[job_id]["error"] = str(exc)
            else:
                enroll_jobs[job_id]["status"] = "complete"

        threading.Thread(target=run_job, name=f"enroll-upload-{job_id}", daemon=True).start()
        return jsonify(
            {
                "status": "ok",
                "saved_images": len(saved_paths),
                "paths": [path.relative_to(Path.cwd()).as_posix() for path in saved_paths],
                "job": enroll_jobs[job_id],
            }
        ), 202

    @app.get("/identities/enroll/<job_id>")
    def enroll_job(job_id: str):
        job = enroll_jobs.get(job_id)
        if job is None:
            return jsonify({"status": "error", "error": "enroll job not found"}), 404
        return jsonify({"status": "ok", "job": job})

    @app.post("/identities/<employee_id>/deactivate")
    def deactivate_identity(employee_id: str):
        from qdrant_client import models

        client = qdrant_client()
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(key="bucket_name", match=models.MatchValue(value=cfg.bucket_name)),
                models.FieldCondition(key="employee_id", match=models.MatchValue(value=employee_id)),
            ]
        )
        ids = []
        offset = None
        while True:
            points, offset = qdrant_scroll(client, scroll_filter=scroll_filter, offset=offset)
            ids.extend(point.id for point in points)
            if offset is None:
                break
        if not ids:
            return jsonify({"status": "error", "error": "identity not found"}), 404
        client.set_payload(
            collection_name=cfg.qdrant.collection,
            payload={"active": False},
            points=ids,
            wait=True,
        )
        return jsonify({"status": "ok", "employee_id": employee_id, "deactivated_points": len(ids)})

    @app.post("/identities/<employee_id>/activate")
    def activate_identity(employee_id: str):
        from qdrant_client import models

        client = qdrant_client()
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(key="bucket_name", match=models.MatchValue(value=cfg.bucket_name)),
                models.FieldCondition(key="employee_id", match=models.MatchValue(value=employee_id)),
            ]
        )
        ids = []
        offset = None
        while True:
            points, offset = qdrant_scroll(client, scroll_filter=scroll_filter, offset=offset)
            ids.extend(point.id for point in points)
            if offset is None:
                break
        if not ids:
            return jsonify({"status": "error", "error": "identity not found"}), 404
        client.set_payload(
            collection_name=cfg.qdrant.collection,
            payload={"active": True},
            points=ids,
            wait=True,
        )
        return jsonify({"status": "ok", "employee_id": employee_id, "activated_points": len(ids)})

    @app.delete("/identities/<employee_id>")
    def delete_identity(employee_id: str):
        from qdrant_client import models

        client = qdrant_client()
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(key="bucket_name", match=models.MatchValue(value=cfg.bucket_name)),
                models.FieldCondition(key="employee_id", match=models.MatchValue(value=employee_id)),
            ]
        )
        ids = []
        offset = None
        while True:
            points, offset = qdrant_scroll(client, scroll_filter=scroll_filter, offset=offset)
            ids.extend(point.id for point in points)
            if offset is None:
                break
        if not ids:
            return jsonify({"status": "error", "error": "identity not found"}), 404
        client.delete(
            collection_name=cfg.qdrant.collection,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )
        return jsonify({"status": "ok", "employee_id": employee_id, "deleted_points": len(ids)})

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
