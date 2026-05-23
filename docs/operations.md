# Operations Runbook

This document contains operational checks and debug commands.

## Service Status

```bash
docker compose --profile pipeline ps
```

## Logs

```bash
docker compose logs -f frame-reader
docker compose logs -f worker
docker compose logs -f api
docker compose logs -f triton
docker compose logs -f qdrant
docker compose logs -f postgres
docker compose logs -f redis
```

## Restart Services

```bash
docker compose --profile pipeline restart api
docker compose --profile pipeline restart worker
docker compose --profile pipeline restart frame-reader
```

## Redis Checks

Frame queue length:

```bash
docker compose exec redis redis-cli XLEN attendance:frames
```

Consumer group state:

```bash
docker compose exec redis redis-cli XINFO GROUPS attendance:frames
```

Recent recognition results:

```bash
docker compose exec redis redis-cli XREVRANGE attendance:results + - COUNT 5
```

## PostgreSQL Checks

Open psql:

```bash
docker compose exec postgres psql \
  "postgresql://camera_app:camera_app_dev_password@localhost:5432/camera_db"
```

List tables:

```sql
\dt
```

Inspect cameras:

```sql
SELECT id, name, rtsp_url, enabled, status, last_seen_at, last_error, metadata
FROM cameras;
```

Inspect recent events:

```sql
SELECT id, occurred_at, camera_id, employee_id, status, quality_score, score
FROM attendance_events
ORDER BY occurred_at DESC
LIMIT 20;
```

## Qdrant Checks

Collection status:

```bash
curl http://localhost:6333/collections/face_embeddings
```

Scroll points:

```bash
curl -X POST http://localhost:6333/collections/face_embeddings/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit":5,"with_payload":true,"with_vector":false}'
```

## Triton Checks

Readiness:

```bash
curl http://localhost:8000/v2/health/ready
```

Model metadata:

```bash
curl http://localhost:8000/v2/models/ultralight
curl http://localhost:8000/v2/models/arcface
curl http://localhost:8000/v2/models/lightqnet
curl http://localhost:8000/v2/models/fasnet_v1se
curl http://localhost:8000/v2/models/fasnet_v2
```

## Camera Debug

Test RTSP inside Docker:

```bash
docker compose --profile pipeline run --rm \
  -e RTSP_URL='rtsp://user:password@192.168.1.10:8080/h264.sdp' \
  worker python -u -c 'import os,cv2;c=cv2.VideoCapture(os.environ["RTSP_URL"]);print("opened",c.isOpened());ok,f=c.read();print("read",ok,None if f is None else f.shape);c.release()'
```

Preview detector and save snapshots:

```bash
python -m scripts.preview_detector \
  --source "rtsp://user:password@192.168.1.10:8080/h264.sdp" \
  --triton-url localhost:8000 \
  --snapshot-dir debug_detector \
  --crop-margin-x 0.1 \
  --crop-margin-y 0.15 \
  --save-crops
```

## Quality Debug

Save face crops used by LightQNet:

```bash
SAVE_QUALITY_DEBUG=1 docker compose --profile pipeline up -d --build worker
```

Inspect `debug_quality/`.

## Stage Smoke Test

```bash
python -m pipeline.stages.test_stages_io \
  --url localhost:8000 \
  --image FacenetDataset/alice/001.jpg
```

Expected output includes:

- detection count
- quality score
- embedding shape
- liveness score
- tracking count
- vector search result
