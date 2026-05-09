# Triton Model Repository

Expected CPU deployment layout:

```text
triton_model_repository/
  arcface/
    config.pbtxt
    1/model.onnx
  lightqnet/
    config.pbtxt
    1/model.graphdef
  fasnet_v1se/
    config.pbtxt
    1/model.onnx
  fasnet_v2/
    config.pbtxt
    1/model.onnx
```

The repository intentionally does not commit model weights. Export or copy model files before starting Triton.

