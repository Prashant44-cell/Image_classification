# deep_image_analyzer/core/postprocessing.py

# Import modules for saving tensors as images
import torch
import numpy as np
from pathlib import Path
from torchvision.utils import save_image

def decode_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Clamp tensor values to [0,1] for visualization or saving.
    """
    return tensor.clamp(0,1)

def save_output(image_tensor: torch.Tensor, out_path: Path):
    """
    Save a tensor as an image file to the specified path.
    Creates parent directories if they do not exist.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(decode_tensor(image_tensor), str(out_path))
