import logging
import os
import argparse
import torch

from backbones import get_model
from eval import verification


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_dataset(bin_path: str, model: torch.nn.Module, image_size=(112, 112)):
    """
    Evaluate the dataset and log the results.
    """

    dataset_name = os.path.basename(bin_path).split('.')[0]

    # Load dataset
    dataset = verification.load_bin(bin_path, image_size)
    

    model.to(device)
    model.eval()
    # Perform evaluation
    acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(dataset, model, 10, 10)

    # Log results
    logging.info("[%s] XNorm: %f", dataset_name, xnorm)
    logging.info("[%s] Accuracy (Flip): %1.5f ± %1.5f", dataset_name, acc2, std2)
    logging.info("[%s] Highest Accuracy: %1.5f", dataset_name, acc1)

    return acc1, std1, acc2, std2, xnorm, embeddings_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a face recognition dataset.")
    
    # Command-line arguments
    parser.add_argument("bin_path", type=str, help="Path to the dataset .bin file.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(112, 112), help="Image size (width height).")
    parser.add_argument("--model_name", type=str, default = 'r100', help="backbone model name.")
    parser.add_argument("--model_path", type=str, default = 'weights/glint360k_cosface_r100_fp16_0.1/backbone.pth', help="backbone model path.")
    args = parser.parse_args()

    model = get_model(args.model_name, fp16=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))


    evaluate_dataset(args.bin_path, model, tuple(args.image_size))





