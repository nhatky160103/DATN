# Headless Face Attendance Pipeline

Production pipeline:

```text
RTSP Camera -> Frame Reader/Sampler -> Redis Queue -> MTCNN Detection
-> Face Tracking -> LightQNet Quality -> FASNet Liveness
-> ArcFace Embedding on Triton -> FAISS Vector Search -> Firebase + Response API
```

The old Flask dashboard and Cloudinary flow have been removed from the runtime path. Employee identity is deterministic: each FAISS vector row stores the exact Firebase `employee_id`.

## Identity Contract

```text
Firebase employee: {bucket}/Employees/{employee_id}
Firebase events:   {bucket}/RecognitionEvents/{event_id}
FAISS metadata:    local_embeddings/{bucket}/ms1mv3_arcface_employee_ids.pkl
Vector mapping:    employee_ids[row] == employee_id for embeddings[row]
Dataset layout:    data/employees/{employee_id}/*.jpg
```

Build/update the local identity store:

```bash
python -m pipeline.enroll_identity_store \
  --bucket Hust \
  --dataset-root data/employees
```

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py frame-reader
python main.py worker
python main.py api
```

For local ONNX smoke tests:

```bash
pip install -r requirements.txt
python -m models.test_model_inference
```

For MTCNN ONNX and PyTorch comparison:

```bash
pip install -r requirements.txt
python -m models.Detection.export_onnx --output-dir models/Detection/onnx
python -m models.Detection.test_onnx --onnx-dir models/Detection/onnx --compare-detector
```

For model export:

```bash
pip install -r requirements.txt
```

## Deploy On GCP CPU VM

```bash
docker compose -f deploy/docker-compose.cpu.yml up -d --build
curl http://localhost:5000/health
curl http://localhost:5000/results/latest
```

Prepare Triton models under `triton_model_repository/*/1/` before starting the stack. See [deploy/gcp-cpu-vm.md](deploy/gcp-cpu-vm.md).

Smoke test model artifacts before running the pipeline:

```bash
python -m models.test_model_inference
python -m models.test_model_inference --triton-url localhost:8000
```

See [docs/model-runtime-status.md](docs/model-runtime-status.md) for ONNX/Triton status per model.

If MTCNN must run in C++, use it as a detector microservice outside Triton. See [docs/cpp-mtcnn-service.md](docs/cpp-mtcnn-service.md).

## Main Structure

```text
attendance_pipeline/       Runtime services and AI stages
database/                  Firebase/timekeeping only
infer/                     Shared preprocessing/model loading helpers
models/                    Model definitions, export scripts, and model smoke tests
triton_model_repository/   Triton config files
deploy/                    CPU VM deployment files
```
