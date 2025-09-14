# import torch
# import os
# from collections import OrderedDict

# from .FasNetBackbone import MiniFASNetV2, MiniFASNetV1SE  # backbone trong repo

# first_model_weight_file = 'models/weights/2.7_80x80_MiniFASNetV2.pth'
# second_model_weight_file = 'models/weights/4_0_0_80x80_MiniFASNetV1SE.pth'

# onnx_dir = "models/weights"
# os.makedirs(onnx_dir, exist_ok=True)

# device = torch.device("cpu")  # có thể đổi sang "cuda" nếu muốn export trên GPU


# def load_model(backbone_cls, weight_path, device):
#     """Load backbone và state_dict (handle DataParallel)."""
#     model = backbone_cls(conv6_kernel=(5, 5)).to(device)
#     state_dict = torch.load(weight_path, map_location=device)

#     if list(state_dict.keys())[0].startswith("module."):
#         new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
#         state_dict = new_state_dict

#     model.load_state_dict(state_dict)
#     model.eval()
#     return model


# def export_onnx(model, save_path, device):
#     """Export model sang ONNX (input=1x3x80x80, output=logits)."""
#     dummy_input = torch.randn(1, 3, 80, 80, device=device)
#     torch.onnx.export(
#         model,
#         dummy_input,
#         save_path,
#         input_names=["input"],
#         output_names=["logits"],  # logits, softmax xử lý ngoài
#         opset_version=11,
#         dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
#     )
#     print(f"✅ Saved ONNX model: {save_path}")


# if __name__ == "__main__":
#     # --- Convert MiniFASNetV2 ---
#     model1 = load_model(MiniFASNetV2, first_model_weight_file, device)
#     export_onnx(model1, os.path.join(onnx_dir, "MiniFASNetV2.onnx"), device)

#     # --- Convert MiniFASNetV1SE ---
#     model2 = load_model(MiniFASNetV1SE, second_model_weight_file, device)
#     export_onnx(model2, os.path.join(onnx_dir, "MiniFASNetV1SE.onnx"), device)
