# C++ MTCNN Service Design

Use this when MTCNN must run as a separate low-latency CPU/edge service.

## Runtime Flow

```text
frame-reader
  -> Redis list attendance:frames
mtcnn-cpp-service
  -> reads attendance:frames
  -> runs C++ MTCNN
  -> writes attendance:detections
worker
  -> reads attendance:detections when detection.provider=external_cpp
  -> tracking + quality + liveness + ArcFace + FAISS + Firebase
```

MTCNN stays outside Triton. Triton continues serving models that are clean inference graphs: ArcFace ONNX, optional LightQNet, optional FASNet.

## Config

```yaml
detection:
  provider: 'external_cpp'

redis:
  frame_queue: 'attendance:frames'
  detection_queue: 'attendance:detections'
```

Use `provider: 'python_mtcnn'` to keep the current optimized PyTorch MTCNN inside the Python worker.

## Input Message

The C++ service pops JSON strings from `attendance:frames`.

```json
{
  "camera_id": "camera-01",
  "frame_id": "uuid",
  "timestamp": 1710000000.0,
  "image_jpeg_b64": "/9j/..."
}
```

## Output Message

The C++ service pushes JSON strings to `attendance:detections`.

```json
{
  "camera_id": "camera-01",
  "frame_id": "uuid",
  "timestamp": 1710000000.0,
  "image_jpeg_b64": "/9j/...",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "score": 0.98,
      "crop_jpeg_b64": "/9j/..."
    }
  ]
}
```

Coordinates must be in original frame pixels, not resized detector pixels.

## C++ Implementation Options

Preferred production shape:

```text
OpenCV image decode
-> C++ MTCNN detector
-> bbox scale back to original frame
-> crop face JPEG
-> Redis push detection JSON
```

Practical dependencies:

- OpenCV: JPEG decode/encode, resize.
- hiredis or redis-plus-plus: Redis list pop/push.
- nlohmann/json: JSON parse/serialize.
- C++ MTCNN implementation: happynear Caffe MTCNN, OpenCV DNN Caffe port, or a custom PNet/RNet/ONet implementation.

## Why Not Triton For Caffe MTCNN

Full MTCNN is not just one model call. It includes scale pyramid generation, PNet/RNet/ONet cascade, bounding-box regression, NMS, crop generation, and coordinate transforms. Caffe PNet/RNet/ONet can run in C++, but the orchestration is application code. Keep that code in a detector service.

## Worker Behavior

When `detection.provider=external_cpp`, the Python worker no longer runs MTCNN. It consumes `attendance:detections` and continues from tracking onward.
