# deep_image_analyzer/effects/halftone.py

# Import torch and config constants
import torch
from core.config import Config

def generate_bayer(n: int = Config.BAYER_ORDER):
    """
    Recursively generate an n-level Bayer dither matrix.
    """
    m = torch.tensor([[0.0]])
    for _ in range(n):
        size = m.size(0)
        new = torch.zeros(size*2, size*2)
        new[:size, :size] = 4 * m
        new[:size, size:] = 4 * m + 2
        new[size:, :size] = 4 * m + 3
        new[size:, size:] = 4 * m + 1
        m = new
    return m / (4**n - 1)

class HalftoneOperator(torch.nn.Module):
    """
    Applies ordered dithering halftone effect.
    """
    def __init__(self):
        super().__init__()
        self.matrix = generate_bayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compare image tensor against tiled Bayer matrix to produce binary halftone.
        """
        b, c, h, w = x.shape
        # Tile matrix to image dimensions
        tile = self.matrix.repeat((h // self.matrix.size(0) + 1,
                                   w // self.matrix.size(1) + 1))
        tile = tile[:h, :w].unsqueeze(0).unsqueeze(0)
        return (x > tile).float()
