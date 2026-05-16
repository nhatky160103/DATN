# Triton Model Repository

Expected CPU deployment layout:

```text
triton_model_repository/
  arcface/
    config.pbtxt
    1/backbone_r18.onnx
  lightqnet/
    config.pbtxt
    1/lightqnet-dm100.onnx
  fasnet_v1se/
    config.pbtxt
    1/4_0_0_80x80_MiniFASNetV1SE.onnx
  fasnet_v2/
    config.pbtxt
    1/2.7_80x80_MiniFASNetV2.onnx
  ultralight/
    config.pbtxt
    1/version-slim-320-dynamic.onnx
```

The repository intentionally does not commit model weights. Export or copy model files before starting Triton.
