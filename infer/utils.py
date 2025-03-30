import torch
import torch.nn as nn
from models.Recognition.Arcface_torch.backbones import get_model

# use device
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_recogn_model(model_name: str = 'glint360k_cosface', backbone_name: str = 'r100'):
    """
    Loads a pre-trained recognition model based on the specified model and backbone names.
    Args:
        model_name (str): Name of the model to load. Default is 'glint360k_cosface' (glint360k_cosface or ms1mv3_arcface).
        backbone_name (str): Name of the backbone architecture. Default is 'r100' (r100 or r50).
    Returns:
        torch.nn.Module: The loaded model.
    """
  
    model = None
    state_dict = None
    
    if model_name == 'glint360k_cosface':
        pretrained_path = f"models/Recognition/Arcface_torch/weights/{model_name}_{backbone_name}_fp16_0.1/backbone.pth"
    else:
        pretrained_path = f"models/Recognition/Arcface_torch/weights/{model_name}_{backbone_name}_fp16/backbone.pth"

    try:
        state_dict = torch.load(pretrained_path, map_location=device)
    except FileNotFoundError:
        print(f"Pretrained model not found at {pretrained_path}.")
        return None
    try:
        model = get_model(backbone_name, fp16=True)
    except Exception as e:
        print(f"Error loading model {model_name} with backbone {backbone_name}: {e}")
        return None


    if state_dict:
        model.load_state_dict(state_dict)

    model.eval()
    return model

def set_model_gpu_mode(model):
    '''
    Configures the model to run on GPU if available, supporting multi-GPU setups if applicable.

    Parameters:
        model (torch.nn.Module): The PyTorch model to configure for GPU usage.

    Returns:
        tuple: 
            - torch.nn.Module: The model configured for single or multi-GPU usage.
            - bool: A flag indicating if multi-GPU mode is active.
    '''
    
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu
