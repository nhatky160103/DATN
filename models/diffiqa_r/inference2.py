import argparse
import torch
import numpy as np
from PIL import Image
import os
from typing import Union, List
import cv2

from .utils import seed_all, construct_full_model

# Default paths
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "model_weights", "diffiqa_r.pth")
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "model_config.yaml")

# Set device once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and transform once at module level
model = None
transform = None

def load_model_and_transform():
    """Load model and transform from config and weights - only load once"""
    global model, transform
    
    if model is None:
        # Check if files exist
        if not os.path.exists(DEFAULT_WEIGHTS):
            raise FileNotFoundError(f"Model weights not found: {DEFAULT_WEIGHTS}")
        if not os.path.exists(DEFAULT_CONFIG):
            raise FileNotFoundError(f"Model config not found: {DEFAULT_CONFIG}")

        print(f"Using device: {device}")

        # Load model & transform
        model, transform = construct_full_model(DEFAULT_CONFIG)
        model.load_state_dict(torch.load(DEFAULT_WEIGHTS, map_location=device))
        model.to(device).eval()
        
        # Enable cudnn benchmarking for faster inference
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
    
    return model, transform

def preprocess_image(image: Union[Image.Image, np.ndarray, torch.Tensor, str]) -> torch.Tensor:
    """Preprocess image to tensor format
    
    Args:
        image: Input image (PIL Image, numpy array, torch tensor or path to image)
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Convert to PIL Image if needed
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.permute(1, 2, 0).numpy()
            image = Image.fromarray(image)
        else:
            raise ValueError("Tensor must be 3D (H,W,C)")
    
    # Apply transform and move to device
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

@torch.no_grad()
def evaluate_image_quality(image: Union[Image.Image, np.ndarray, torch.Tensor, str, List[Union[Image.Image, np.ndarray, torch.Tensor, str]]]) -> Union[float, List[float]]:
    """Evaluate image quality score using DifFIQA(R) model
    
    Args:
        image: Input image(s) - can be:
            - PIL Image
            - numpy array
            - torch tensor
            - path to image file
            - list of any of the above
        
    Returns:
        Union[float, List[float]]: Quality score(s) between 0 and 1
    """
    global model, transform
    
    # Load model and transform if not loaded
    if model is None:
        model, transform = load_model_and_transform()
        
    # Handle single image or list of images
    is_list = isinstance(image, list)
    if not is_list:
        image = [image]
        
    # Process each image
    scores = []
    for img in image:
        # Preprocess image
        image_tensor = preprocess_image(img)
        
        # Predict quality score
        quality_score = model(image_tensor).squeeze().item()
        
        # Ensure score is between 0 and 1
        quality_score = max(0.0, min(1.0, quality_score))
        scores.append(quality_score)
        
        # Print score for single image
        if not is_list:
            print(f"Quality score: {quality_score:.4f}")
        else:
            print(f"Quality score for image {len(scores)}: {quality_score:.4f}")
            
    return scores[0] if not is_list else scores

def run_camera_inference( window_name: str = "Quality Score"):
    """Run inference on camera feed
    
    Args:
        camera_id (int): Camera device ID
        window_name (str): Name of the window to display
    """
    # Load model
    global model, transform
    if model is None:
        model, transform = load_model_and_transform()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera")
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get quality score
            score = evaluate_image_quality(frame_rgb)
            
            # Display score on frame
            cv2.putText(frame, f"Quality: {score:.4f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow(window_name, frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image quality using DifFIQA(R) model")
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument("--camera", action="store_true", help="Run inference on camera feed")
    args = parser.parse_args()
    
    if args.camera:
        run_camera_inference()
    elif args.image:
        evaluate_image_quality(image=args.image)
    else:
        print("Please provide either --image or --camera argument")


