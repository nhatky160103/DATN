# ArcFace ONNX Deploy Assets

This folder is intentionally trimmed for deployment.

Kept files:

- `backbones/`: PyTorch backbone definitions required to load `backbone.pth`.
- `weights/backbone.pth`: source checkpoint for ONNX export.
- `torch2onnx.py`: local exporter for Triton-compatible ONNX.

Export to Triton model repository:

```bash
python -m models.Arcface.torch2onnx \
  models/Arcface/weights/backbone.pth \
  --network r18 \
  --output triton_model_repository/arcface/1/model.onnx \
  --simplify
```

The bundled `weights/backbone.pth` is ResNet-18, so use `--network r18`.
