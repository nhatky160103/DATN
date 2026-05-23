# Usage Guide

## Admin Dashboard

Open:

```text
http://localhost:5000
```

The dashboard is organized into four tabs.

| Tab | Purpose |
| --- | --- |
| `Overview` | System metrics and recent recognition events |
| `Cameras` | Add, enable, disable, and delete camera sources |
| `Identities` | Enroll, activate, deactivate, and delete identities |
| `Settings` | Edit recognition thresholds in `config.yaml` |

## Camera Management

Camera fields:

| Field | Description |
| --- | --- |
| `id` | Stable camera ID, for example `camera-01` |
| `name` | Display name |
| `source` | RTSP URL, video path, or camera index |
| `enabled` | Whether `frame-reader` should read the camera |
| `rotate` | Rotation angle: `0`, `90`, `180`, `270` |

Example RTSP source:

```text
rtsp://user:password@192.168.1.10:8080/h264.sdp
```

Example laptop/webcam source inside a suitable runtime:

```text
0
```

## Identity Enrollment

Each subdirectory under the dataset root is treated as one `employee_id`.

```text
FacenetDataset/
  alice/
    001.jpg
    002.jpg
  bob/
    001.jpg
    002.jpg
```

Dashboard enrollment starts a background job. CLI enrollment:

```bash
docker compose --profile pipeline run --rm worker \
  python -m pipeline.enroll_qdrant_identity_store \
  --config config.yaml \
  --dataset-root FacenetDataset
```

Use `--min-quality` to override the configured quality threshold:

```bash
docker compose --profile pipeline run --rm worker \
  python -m pipeline.enroll_qdrant_identity_store \
  --config config.yaml \
  --dataset-root FacenetDataset \
  --min-quality 0.3
```

## Settings

Important runtime settings:

| Setting | Meaning |
| --- | --- |
| `qdrant.match_threshold` | Minimum cosine score for accepting an identity |
| `infer_video.qscore_threshold` | Minimum LightQNet quality score |
| `infer_video.bbox_threshold` | Face detector confidence threshold |
| `infer_video.required_images` | Minimum valid frames before a final decision |
| `infer_video.validation_threshold` | Required track-level agreement ratio |
| `infer_video.max_track_buffer` | Maximum frames before returning `unknown` |
| `detection.crop_margin_x` | Horizontal crop expansion |
| `detection.crop_margin_y` | Vertical crop expansion |

Restart the worker after changing recognition settings:

```bash
docker compose --profile pipeline restart worker
```

## REST API

Health:

```bash
curl http://localhost:5000/health
```

System status:

```bash
curl http://localhost:5000/system/status
```

Recent results:

```bash
curl "http://localhost:5000/results/recent?count=20"
```

Latest result:

```bash
curl http://localhost:5000/results/latest
```

List cameras:

```bash
curl http://localhost:5000/cameras
```

List identities:

```bash
curl http://localhost:5000/identities
```

Single image recognition:

```bash
curl -X POST http://localhost:5000/recognize \
  -F camera_id=manual-test \
  -F image=@FacenetDataset/alice/001.jpg
```
