# deep_image_analyzer/analysis/gradcam.py

# Import image processing libraries
import cv2
import numpy as np
import torch

class GradCAMService:
    """
    Grad-CAM visualization service.
    """
    def __init__(self, service, target_layer: str):
        self.model = service.model
        self.gradients = None
        self.activations = None
        # Fetch layer by name and attach hooks
        target = dict(self.model.named_modules())[target_layer]
        target.register_forward_hook(self._forward_hook)
        target.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, outp):
        # Store activations from forward pass
        self.activations = outp

    def _backward_hook(self, module, grad_in, grad_out):
        # Store gradients from backward pass
        self.gradients = grad_out[0]

    def generate(self, x: torch.Tensor, class_idx=None):
        """
        Generates a heatmap for the predicted (or specified) class index.
        """
        self.model.zero_grad()
        out = self.model(x)
        idx = class_idx or out.argmax(dim=1).item()
        out[0, idx].backward()
        grads = self.gradients[0].cpu().numpy()
        acts = self.activations.cpu().numpy()
        # Compute weights as global-average-pooled gradients
        weights = grads.mean(axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)      # ReLU
        cam = cv2.resize(cam / cam.max(), (x.size(2), x.size(3)))
        return cam
