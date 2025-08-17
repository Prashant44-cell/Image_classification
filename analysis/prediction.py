# deep_image_analyzer/analysis/prediction.py

# Import torch and model factory
import torch
from core.models import ModelFactory
from core.config import Config
from utils.exceptions import InferenceError

class InferenceService:
    """
    Service for running model inference with optional AMP.
    """
    def __init__(self, model_name: str, device: str = Config.DEVICE, amp: bool = Config.AMP_ENABLED):
        self.device = device
        self.amp = amp
        # Load model on specified device
        self.model = ModelFactory.get(model_name).to(self.device)

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Accepts a batch tensor, performs forward pass, and returns softmax probabilities.
        """
        batch = batch.to(self.device, non_blocking=True)
        try:
            # Use AMP if enabled
            if self.amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch)
            else:
                logits = self.model(batch)
            return torch.softmax(logits, dim=1)
        except Exception as e:
            # Wrap exceptions in a custom error
            raise InferenceError(str(e))
