# deep_image_analyzer/core/models.py

# Import torch and torchvision model definitions
import torch
import torchvision.models as models

# Registry mapping model names to factory functions
MODEL_REGISTRY = {
    "resnet50": lambda: models.resnet50(pretrained=True),
    "efficientnet_b0": lambda: models.efficientnet_b0(pretrained=True),
    "vit_b_16": lambda: models.vit_b_16(pretrained=True),
}

class ModelFactory:
    @staticmethod
    def get(name: str) -> torch.nn.Module:
        """
        Retrieve a pretrained model by name.
        Raises ValueError if the model name is unregistered.
        """
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'")
        model = MODEL_REGISTRY[name]()   # Instantiate the model
        model.eval()                     # Set to evaluation mode
        return model
