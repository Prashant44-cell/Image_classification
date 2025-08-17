# deep_image_analyzer/analysis/uncertainty.py

# Import torch and config constants
import torch
from core.config import Config

class UncertaintyEstimator:
    """
    Monte Carlo dropout uncertainty estimation.
    """
    def __init__(self, service: InferenceService, samples: int = Config.MC_SAMPLES):
        self.service = service
        self.samples = samples

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor):
        """
        Enables dropout, runs multiple stochastic forward passes,
        and returns mean prediction & aggregated variance.
        """
        # Turn dropout layers to train mode
        for m in self.service.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
        preds = []
        for _ in range(self.samples):
            preds.append(self.service.predict(x))
        stacked = torch.stack(preds)        # Shape: (samples, batch, classes)
        mean = stacked.mean(dim=0)         # Mean over samples
        var = stacked.var(dim=0).sum(dim=1)  # Sum variances per example
        return mean, var
