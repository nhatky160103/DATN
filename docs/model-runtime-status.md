# Model Runtime Status

## ArcFace r18

- Runtime target: ONNXRuntime through Triton.
- Status: supported.
- Required artifact: `triton_model_repository/arcface/1/model.onnx`.
- Export:

```bash
python -m models.Arcface.export_onnx \
  --weights models/Arcface/weights/backbone.pth \
  --network r18 \
  --output triton_model_repository/arcface/1/model.onnx
```

## MTCNN

- Runtime target: optimized CPU PyTorch in the worker, or ONNX Runtime through the `ONNXMTCNN` wrapper.
- Status: supported outside Triton.
- Reason: full MTCNN includes scale pyramid generation, proposal filtering, bbox regression and NMS around three subnetworks. PNet/RNet/ONet are exported separately; the Python wrapper keeps the existing pre/post-processing identical.
- Export:

```bash
python -m models.Detection.export_onnx --output-dir models/Detection/onnx
```

- Test:

```bash
python -m models.Detection.test_onnx --onnx-dir models/Detection/onnx --compare-detector
```

- Current optimization:
  - Resize frame before detection with `detection.mtcnn_max_side`.
  - Increase `detection.mtcnn_min_face_size`.
  - Use `landmarks=False`.
  - Set `post_process=False`.
  - Limit PyTorch CPU threads with `detection.torch_num_threads`.

## LightQNet

- Runtime target: TensorFlow GraphDef on Triton, or ONNX if converted later.
- Status: supported by test script for both GraphDef and ONNX.
- Required artifact for current Triton config: `triton_model_repository/lightqnet/1/model.graphdef`.
- Current repo note: no LightQNet model artifact is present unless you copy/export it.

## FASNet

- Runtime target: ONNXRuntime through Triton.
- Status: supported after exporting both backbones.
- Required artifacts:
  - `triton_model_repository/fasnet_v1se/1/model.onnx`
  - `triton_model_repository/fasnet_v2/1/model.onnx`
- Current repo note: FASNet weights are not present unless you copy them back.

## Smoke Test

Run local model load + inference checks:

```bash
python -m models.test_model_inference
```

Run Triton checks after starting Triton:

```bash
python -m models.test_model_inference --triton-url localhost:8000
```

The script prints one JSON line per model. `ok: false` means either the artifact is missing, dependency is missing, or output shape is not what the pipeline expects.
