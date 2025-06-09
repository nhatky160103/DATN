import torch
import torch.nn as nn
import onnxruntime as ort
from models.Recognition.Arcface_torch.backbones import get_model
from models.Detection.mtcnn import MTCNN

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mtcnn = MTCNN(
    image_size=112, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    select_largest=True,
    selection_method='largest',
    device=device,
    keep_all=False,
)

def get_recogn_model(model_name: str = 'ms1mv3_arcface', backbone_name: str = 'r100'):
    model = None
    state_dict = None

    try:
        if model_name == 'glint360k_cosface':
            pretrained_path = f"models/Recognition/Arcface_torch/weights/{model_name}_{backbone_name}_fp16_0.1/backbone.pth"
        else:
            pretrained_path = f"models/Recognition/Arcface_torch/weights/{model_name}_{backbone_name}_fp16/backbone.pth"
      
        state_dict = torch.load(pretrained_path, map_location=device)
        model = get_model(backbone_name, fp16=True)

        if state_dict:
            model.load_state_dict(state_dict)

        model.eval()
        model.to(device)
        return model

    except Exception as e:
        print(f"⚠️ Error loading model {model_name} with backbone {backbone_name}: {e}")
        print("🔁 Falling back to default model ms1mv3_arcface with r100")
        # fallback
        try:
            fallback_path = "models/Recognition/Arcface_torch/weights/ms1mv3_arcface_r100_fp16_0.1/backbone.pth"
            fallback_model = get_model("r100", fp16=True)
            fallback_state_dict = torch.load(fallback_path, map_location=device)
            fallback_model.load_state_dict(fallback_state_dict)
            fallback_model.eval()
            return fallback_model
        except Exception as fe:
            print(f"❌ Fallback model failed too: {fe}")
            return None



if __name__ == "__main__":
    # Test PyTorch model
    model = get_recogn_model('casia_webface_cmd', 'r50')
    print("PyTorch model:", model)
    
