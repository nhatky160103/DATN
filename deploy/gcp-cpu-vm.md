# Deploy on a CPU VM

This guide describes a CPU-oriented deployment for the current Docker Compose runtime.

## Runtime Stack

```mermaid
flowchart LR
    Camera[RTSP Camera] --> Reader[frame-reader]
    Reader --> Redis[(Redis Streams)]
    Redis --> Worker[worker]
    Worker --> Triton[Triton CPU Inference]
    Worker --> Qdrant[(Qdrant)]
    Worker --> Postgres[(PostgreSQL)]
    API[Admin Dashboard/API] --> Postgres
    API --> Qdrant
```

## Recommended VM

Minimum for functional testing:

- 4 vCPU
- 16 GB RAM
- 50 GB disk

Recommended for multiple cameras:

- 8+ vCPU
- 32 GB RAM
- SSD disk

Open only the ports you need:

| Port | Service | Exposure |
| --- | --- | --- |
| `5000` | Admin Dashboard/API | Private network, VPN, or reverse proxy |
| `8000` | Triton HTTP | Private only |
| `6333` | Qdrant HTTP | Private only |
| `5432` | PostgreSQL | Private only |
| `6379` | Redis | Private only |

## Install Docker

Ubuntu example:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl docker.io docker-compose-plugin
sudo usermod -aG docker "$USER"
newgrp docker
```

## Configure Environment

Create `.env`:

```bash
cp .env.example .env
```

Set production credentials:

```dotenv
POSTGRES_DB=camera_db
POSTGRES_USER=camera_app
POSTGRES_PASSWORD=change_this_password
DATABASE_URL=postgresql://camera_app:change_this_password@postgres:5432/camera_db
BUCKET_NAME=Hust
QDRANT_API_KEY=
```

## Prepare Models

Verify:

```bash
find triton_model_repository -maxdepth 3 -type f | sort
```

Expected model files are documented in [../triton_model_repository/README.md](../triton_model_repository/README.md).

## Start Services

```bash
docker compose --profile pipeline up -d --build
```

Check:

```bash
docker compose --profile pipeline ps
curl http://localhost:5000/health
curl http://localhost:8000/v2/health/ready
curl http://localhost:6333/collections
```

## Register Cameras

Use the dashboard:

```text
http://VM_PRIVATE_OR_PUBLIC_IP:5000
```

Or CLI:

```bash
docker compose --profile pipeline run --rm worker \
  python -m scripts.register_camera \
  --id camera-01 \
  --name "Main entrance" \
  --source "rtsp://user:password@camera-host:554/stream1"
```

## Enroll Identities

Copy the dataset to the server:

```text
FacenetDataset/
  employee_id_1/*.jpg
  employee_id_2/*.jpg
```

Run:

```bash
docker compose --profile pipeline run --rm worker \
  python -m pipeline.enroll_qdrant_identity_store \
  --config config.yaml \
  --dataset-root FacenetDataset
```

## Operational Checks

Logs:

```bash
docker compose logs -f frame-reader
docker compose logs -f worker
docker compose logs -f api
```

Recent events:

```bash
docker compose exec postgres psql \
  "postgresql://camera_app:change_this_password@localhost:5432/camera_db" \
  -c "SELECT occurred_at, camera_id, employee_id, status, quality_score, score FROM attendance_events ORDER BY occurred_at DESC LIMIT 20;"
```

Qdrant collection:

```bash
curl http://localhost:6333/collections/face_embeddings
```

Redis queue:

```bash
docker compose exec redis redis-cli XINFO GROUPS attendance:frames
```

## Production Hardening

- Put the dashboard behind a reverse proxy with TLS.
- Add authentication before exposing the dashboard.
- Keep PostgreSQL, Redis, Qdrant, and Triton on private networking.
- Back up Docker volumes:
  - `postgres-data`
  - `qdrant-data`
- Monitor disk usage and container restarts.
-- Pin Docker image versions before final production release.
