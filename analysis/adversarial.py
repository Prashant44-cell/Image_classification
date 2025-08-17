# deep_image_analyzer/analysis/adversarial.py

# Import torch functions and config values
import torch
import torch.nn.functional as F
from core.config import Config

class AdversarialGenerator:
    """
    Implements FGSM and PGD adversarial attacks.
    """
    @staticmethod
    def fgsm(service, x: torch.Tensor, epsilon: float = Config.FGSM_EPSILON):
        """
        Fast Gradient Sign Method attack.
        """
        x_adv = x.clone().detach().requires_grad_(True)
        logits = service.model(x_adv)
        loss = F.cross_entropy(logits, logits.argmax(dim=1))
        loss.backward()
        # Apply sign of gradient
        x_adv = torch.clamp(x_adv + epsilon * x_adv.grad.sign(), 0, 1)
        return x_adv

    @staticmethod
    def pgd(service, x: torch.Tensor, epsilon=Config.FGSM_EPSILON,
            alpha=Config.PGD_ALPHA, steps=Config.PGD_STEPS):
        """
        Projected Gradient Descent attack.
        """
        x_adv = x.clone().detach()
        orig = x.detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = service.model(x_adv)
            loss = F.cross_entropy(logits, logits.argmax(dim=1))
            loss.backward()
            # Step in gradient direction and project
            x_adv = torch.clamp(x_adv + alpha * x_adv.grad.sign(), orig - epsilon, orig + epsilon).detach()
        return torch.clamp(x_adv, 0, 1)
