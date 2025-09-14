# import torch
# import os

# # import iresnet18 từ file bạn gửi
# from .backbones.iresnet import iresnet18   # sửa 'your_file' thành tên file .py bạn lưu code

# # Cấu hình
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# weights_path = "models/weights/backbone.pth"
# onnx_path = "models/weights/backbone.onnx"

# # 1. Khởi tạo model
# model = iresnet18()
# state_dict = torch.load(weights_path, map_location=device)

# # nếu state_dict có "module." thì bỏ prefix
# if list(state_dict.keys())[0].startswith("module."):
#     from collections import OrderedDict
#     new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
#     state_dict = new_state_dict

# model.load_state_dict(state_dict, strict=False)
# model.eval().to(device)

# # 2. Dummy input: [batch, 3, 112, 112]
# dummy_input = torch.randn(1, 3, 112, 112).to(device)

# # 3. Export sang ONNX
# torch.onnx.export(
#     model,
#     dummy_input,
#     onnx_path,
#     input_names=["input"],
#     output_names=["embedding"],
#     opset_version=11,
#     dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}}
# )

# print(f"✅ Saved ONNX model at: {onnx_path}")
