# FARS

<div align="center">

**Face Attendance Recognition System**

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Docker](https://img.shields.io/badge/runtime-docker%20compose-2496ED.svg)
![Triton](https://img.shields.io/badge/inference-NVIDIA%20Triton-76B900.svg)
![Qdrant](https://img.shields.io/badge/vector%20search-Qdrant-DC244C.svg)

**A production-oriented face attendance recognition system with realtime camera ingestion, AI inference, vector search, event storage, and an Admin Dashboard.**

[Documentation](docs/) • [Architecture](docs/architecture.md) • [Installation](docs/installation.md) • [Usage](docs/usage.md) • [Operations](docs/operations.md) • [Research Results](docs/results.md)

<br/>

<img src="docs/assets/FARS_logo.jpg" alt="FARS illustration" width="100%">

</div>

---

## Overview

FARS builds an end-to-end attendance system based on face recognition. It continuously reads frames from cameras, detects and validates faces, extracts ArcFace embeddings, searches identities in Qdrant, stores attendance events, and exposes a web dashboard for system administration.

The platform is designed around independent services so that camera ingestion, AI inference, vector search, storage, and administration can be operated and debugged separately.

**Author:** Dinh Nhat Ky  
**Institution:** School of Information and Communication Technology, Hanoi University of Science and Technology

## Core Capabilities

- Multi-camera RTSP/webcam ingestion.
- PostgreSQL camera registry and attendance event storage.
- Redis Streams for frame buffering and realtime result delivery.
- Triton-served AI models for detection, quality, liveness, and embedding.
- Qdrant vector search for identity recognition.
- Track-level multi-frame validation to reduce unstable single-frame decisions.
- Admin Dashboard for cameras, identities, thresholds, and system status.

## System Architecture

<p align="center">
  <img src="docs/assets/FARS_architecture.jpg" alt="FARS system architecture" width="100%">
</p>

## Tech Stack

| Layer | Technology |
| --- | --- |
| Runtime | Docker Compose |
| API/Dashboard | Flask, Gunicorn |
| Camera input | RTSP / webcam |
| Camera ingestion | OpenCV VideoCapture / optional GStreamer |
| Queue | Redis Streams |
| Database | PostgreSQL |
| Vector search | Qdrant |
| Model serving | NVIDIA Triton Inference Server |
| Detection | Ultra-Light Face Detector |
| Tracking | ByteTrack |
| Quality | LightQNet |
| Liveness | MiniFASNet |
| Embedding | ArcFace-compatible ONNX model |

## Quick Start

```bash
cp .env.example .env
docker compose --profile pipeline up -d --build
```

Open the Admin Dashboard:

```text
http://localhost:5000
```

Enroll identities:

```bash
docker compose --profile pipeline run --rm worker \
  python -m pipeline.enroll_qdrant_identity_store \
  --config config.yaml \
  --dataset-root FacenetDataset
```

Check status:

```bash
curl http://localhost:5000/health
curl http://localhost:5000/system/status
curl "http://localhost:5000/results/recent?count=20"
```

## Documentation

| Document | Purpose |
| --- | --- |
| [Architecture](docs/architecture.md) | Detailed service architecture, data contracts, storage model, and diagrams |
| [Installation](docs/installation.md) | Environment setup, Docker Compose startup, and model checks |
| [Usage](docs/usage.md) | Dashboard usage, camera management, identity enrollment, API examples |
| [Operations](docs/operations.md) | Logs, health checks, Redis/PostgreSQL/Qdrant/Triton debug commands |
| [Triton Models](triton_model_repository/README.md) | Triton model repository layout and model input/output contracts |
| [Training](docs/training.md) | ArcFace/CDML training direction and production export path |
| [Results](docs/results.md) | Research accuracy, latency, threshold, and error analysis |
| [References](docs/references.md) | Bibliography and related work |
| [Future Work](docs/future-work.md) | Production and model roadmap |

## Repository Structure

```text
DATN/
├── config.yaml
├── docker-compose.yml
├── database/
│   └── init/
├── docs/
├── FacenetDataset/
├── models/
├── pipeline/
│   ├── api.py
│   ├── frame_reader.py
│   ├── worker.py
│   ├── orchestrator.py
│   ├── qdrant_identity_store.py
│   └── stages/
├── scripts/
└── triton_model_repository/
```

## Citation

```bibtex
@mastersthesis{dinh2026facerecognition,
  title  = {Building an Optimized Deep Learning Model for Face Recognition in Corporate Attendance Systems},
  author = {Dinh, Nhat Ky},
  year   = {2026},
  school = {Hanoi University of Science and Technology},
  note   = {School of Information and Communication Technology}
}
```
