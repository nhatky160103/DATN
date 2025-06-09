import numpy as np
import onnx
import torch
import os
from models.diffiqa_r.utils import construct_full_model

def convert_onnx(model_path, output_path, opset=11, simplify=False):
    """Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to the PyTorch model (.pth file)
        output_path: Path to save the ONNX model
        opset: ONNX operator set version
        simplify: Whether to simplify the ONNX model
    """
    # Load model and transform
    model, transform = construct_full_model("models/diffiqa_r/configs/model_config.yaml")
    
    # Load weights
    weight = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weight, strict=True)
    model.eval()
    
    # Create dummy input
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float32)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        img, 
        output_path,
        input_names=["data"],
        output_names=["output"],
        keep_initializers_as_inputs=False,
        verbose=False,
        opset_version=opset
    )
    
    # Load and modify ONNX model
    model = onnx.load(output_path)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    
    # Simplify if requested
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    
    # Save final model
    onnx.save(model, output_path)
    print(f"Model converted and saved to: {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert DiffIQA PyTorch model to ONNX')
    parser.add_argument('--input', type=str, required=True, help='Path to input .pth file')
    parser.add_argument('--output', type=str, help='Path to output .onnx file')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    
    # Convert model
    convert_onnx(args.input, args.output, simplify=args.simplify) 