# Installation Guide

This guide covers local Docker-based setup for the production pipeline.

## Prerequisites

- Docker Engine
- Docker Compose plugin
- Python 3.10+ for local helper scripts
- Model files under `triton_model_repository/`

## 1. Configure Environment

Create `.env`:

```bash
cp .env.example .env
```

Recommended local values:

```dotenv
POSTGRES_DB=camera_db
POSTGRES_USER=camera_app
POSTGRES_PASSWORD=camera_app_dev_password
DATABASE_URL=postgresql://camera_app:camera_app_dev_password@postgres:5432/camera_db
BUCKET_NAME=Hust
QDRANT_API_KEY=
```

## 2. Verify Triton Model Repository

Expected files:

```text
triton_model_repository/
  ultralight/1/version-RFB-640-dynamic.onnx
  arcface/1/backbone_r18.onnx
  lightqnet/1/lightqnet-dm100.onnx
  fasnet_v1se/1/4_0_0_80x80_MiniFASNetV1SE.onnx
  fasnet_v2/1/2.7_80x80_MiniFASNetV2.onnx
```

See [../triton_model_repository/README.md](../triton_model_repository/README.md).

## 3. Start Infrastructure

For infrastructure only:

```bash
docker compose up -d redis postgres qdrant triton
```

Check readiness:

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:6333/collections
```

## 4. Start Full Pipeline

```bash
docker compose --profile pipeline up -d --build
```

Services started by the profile:

- `frame-reader`
- `worker`
- `api`

## 5. Open Dashboard

```text
http://localhost:5000
```

## 6. Local Python Environment

Use this only for local scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Enroll Identities

Dataset layout:

```text
FacenetDataset/
  employee_a/
    image_01.jpg
    image_02.jpg
  employee_b/
    image_01.jpg
```

Run:

```bash
docker compose --profile pipeline run --rm worker \
  python -m pipeline.enroll_qdrant_identity_store \
  --config config.yaml \
  --dataset-root FacenetDataset
```

Check Qdrant:

```bash
curl http://localhost:6333/collections/face_embeddings
```

## 8. Register a Camera

Use the dashboard or run:

```bash
docker compose --profile pipeline run --rm worker \
  python -m scripts.register_camera \
  --id camera-01 \
  --name "Main camera" \
  --source "rtsp://user:password@192.168.1.10:8080/h264.sdp"
```

Restart frame reader after camera changes if needed:

```bash
docker compose --profile pipeline restart frame-reader
```

## 9. Common Commands

```bash
docker compose --profile pipeline ps
docker compose logs -f frame-reader
docker compose logs -f worker
docker compose logs -f api
docker compose --profile pipeline restart worker api
docker compose --profile pipeline down
```
