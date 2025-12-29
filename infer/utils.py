import onnxruntime as ort
import os


device  = 'cpu'

def load_onnx_model():
    session = None
    try:
        onnx_path = f"models/weights/backbone.onnx"
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        # load ONNX model
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        print(f"✅ Loaded ONNX model: {onnx_path}")
        return session
    except Exception as e:
        print(f"❌ Error loading ONNX model : {e}")
        return None
    


    
