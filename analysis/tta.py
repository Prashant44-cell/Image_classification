# deep_image_analyzer/analysis/tta.py

# Import torch for tensor operations
import torch
from core.preprocessing import get_tta_transforms

class TTAPredictor:
    """
    Test-time augmentation predictor: runs multiple augmented inferences and aggregates.
    """
    def __init__(self, service: InferenceService, iterations: int):
        self.service = service
        self.iterations = iterations
        self.transform = get_tta_transforms()

    @torch.no_grad()
    def predict(self, image_pil):
        """
        Apply TTA transforms repeatedly, collect predictions, and return mean & std.
        """
        preds = []
        for _ in range(self.iterations):
            # Transform PIL image, add batch dimension, and run inference
            inp = self.transform(image_pil).unsqueeze(0)
            preds.append(self.service.predict(inp))
        stacked = torch.stack(preds)         # Shape: (iterations, batch, classes)
        return stacked.mean(dim=0), stacked.std(dim=0)
