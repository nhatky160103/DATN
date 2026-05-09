# Deploy on GCP CPU VM with Triton

This deployment matches the production flow without the old Flask dashboard and without Cloudinary:

`RTSP Camera -> frame-reader -> Redis -> worker AI stages -> Triton -> FAISS -> Firebase -> response API`

## 1. Create VM

Use an Ubuntu 22.04 CPU VM. Start with at least `e2-standard-4` and 30GB disk; use a larger machine if ArcFace/FASNet CPU latency is too high.

Open firewall ports only as needed:

- `5000/tcp` for the response API.
- `8000/tcp` only if Triton HTTP must be reachable outside the VM. Keep it private when possible.

## 2. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

## 3. Configure secrets and camera

Create `.env` in the repo root:

```bash
CAMERA_SOURCE=rtsp://user:password@camera-ip:554/stream1
BUCKET_NAME=Hust
```

Place the Firebase service account at `database/ServiceAccountKey.json`. Do not commit it.

Employee identity rule:

```text
Firebase key:      {BUCKET_NAME}/Employees/{employee_id}
Event log:         {BUCKET_NAME}/RecognitionEvents/{event_id}
FAISS vector row:  local_embeddings/{BUCKET_NAME}/*_employee_ids.pkl[row] == employee_id
Dataset folder:    dataset-root/{employee_id}/*.jpg
```

Build the local identity store:

```bash
python -m attendance_pipeline.enroll_identity_store \
  --bucket "$BUCKET_NAME" \
  --dataset-root data/employees
```

## 4. Prepare Triton model repository

Export/copy model files into:

```text
triton_model_repository/
  arcface/1/model.onnx
  lightqnet/1/model.graphdef
  fasnet_v1se/1/model.onnx
  fasnet_v2/1/model.onnx
```

ArcFace export:

```bash
python -m models.Arcface.export_onnx \
  --weights models/Arcface/weights/backbone.pth \
  --network r18 \
  --output triton_model_repository/arcface/1/model.onnx
```

LightQNet is already a frozen TensorFlow graph in `models/lightqnet/lightqnet-dm050.pb`; copy it to `triton_model_repository/lightqnet/1/model.graphdef`.

FASNet exports:

```bash
python -m models.Anti_spoof.export_onnx \
  --model-type v1se \
  --weights models/Anti_spoof/weights/4_0_0_80x80_MiniFASNetV1SE.pth \
  --output triton_model_repository/fasnet_v1se/1/model.onnx

python -m models.Anti_spoof.export_onnx \
  --model-type v2 \
  --weights models/Anti_spoof/weights/2.7_80x80_MiniFASNetV2.pth \
  --output triton_model_repository/fasnet_v2/1/model.onnx
```

If model files are missing, set `TRITON_ENABLED=false` or `infer_video.is_anti_spoof=false` and the worker will use legacy Python fallback where possible.

Smoke test artifacts before starting the full pipeline:

```bash
python -m models.test_model_inference --skip-mtcnn
```

Smoke test Triton after `triton` is running:

```bash
python -m models.test_model_inference --triton-url localhost:8000 --skip-mtcnn
```

## 5. Start services

```bash
docker compose -f deploy/docker-compose.cpu.yml up -d --build
docker compose -f deploy/docker-compose.cpu.yml ps
```

Check logs:

```bash
docker compose -f deploy/docker-compose.cpu.yml logs -f triton
docker compose -f deploy/docker-compose.cpu.yml logs -f worker
docker compose -f deploy/docker-compose.cpu.yml logs -f frame-reader
```

Headless result API:

```text
http://VM_EXTERNAL_IP:5000
```

API health:

```bash
curl http://localhost:5000/results/latest
curl http://localhost:8000/v2/health/ready
```
