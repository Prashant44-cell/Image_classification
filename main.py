import sys
import os
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision import models
import requests

# ---- 1. Prompt user for image file path ----
# User specifies the path to the image file to be processed
image_path = input("Enter the path to your image: ").strip()
if not os.path.exists(image_path):
    print(f"File '{image_path}' not found.")
    sys.exit(1)

# ---- 2. Initialize and prepare classification model ----
# Load pretrained ResNet-50 for object recognition; set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Prepare image preprocessing for classifier (resize, normalize, etc.)
preprocess_cls = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download ImageNet label names for output
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.strip().split("\n")

# ---- 3. Run image classification ----
# Open and preprocess image, then predict the top class using ResNet-50
with Image.open(image_path).convert("RGB") as img:
    input_tensor = preprocess_cls(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
        predicted_idx = predicted.item()
        predicted_label = labels[predicted_idx]
    # Output: Print recognized object class in bold
    print(f"\nPredicted object: \033[1m{predicted_label}\033[0m\n")

# ---- 4. Define halftoning utilities ----
# Provide helpers for ordered dithering (Bayer halftoning)
class Normer:
    @staticmethod
    def decodes(x: torch.Tensor) -> torch.Tensor:  # [-1,1] → [0,1]
        return x.clamp(-1, 1) * 0.5 + 0.5
    @staticmethod
    def encodes(x: torch.Tensor) -> torch.Tensor:  # [0,1] → [-1,1]
        return x.clamp(0, 1) * 2 - 1

def generate_bayer_matrix(n: int = 3):
    # Recursively build a 2**n x 2**n Bayer matrix for dithering
    matrix = torch.tensor([[0.0]])
    for _ in range(n):
        size = matrix.shape[0]
        new_size = size * 2
        new_matrix = torch.zeros((new_size, new_size), dtype=matrix.dtype)
        new_matrix[:size, :size] = 4 * matrix
        new_matrix[:size, size:] = 4 * matrix + 2
        new_matrix[size:, :size] = 4 * matrix + 3
        new_matrix[size:, size:] = 4 * matrix + 1
        matrix = new_matrix
    max_val = 4 ** n - 1
    return matrix / max_val

class DifferentiableDither(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, dither_matrix):
        # Apply thresholding for halftoning
        binary_output = (image > dither_matrix).float()
        ctx.save_for_backward(image, dither_matrix)
        return binary_output
    @staticmethod
    def backward(ctx, grad_output):
        # Differentiable backward pass (for advanced use)
        image, dither_matrix = ctx.saved_tensors
        temperature = 10.0
        diff = image - dither_matrix
        sigmoid_approx = torch.sigmoid(diff * temperature)
        grad_approx = temperature * sigmoid_approx * (1 - sigmoid_approx)
        return grad_output * grad_approx, None

def ordered_dithering(image: torch.Tensor, n: int = 3, norm: bool = True) -> torch.Tensor:
    # Dither image using Bayer matrix; ensures correct normalization
    normer = Normer()
    if norm:
        image = normer.decodes(image)  # [-1,1] → [0,1]
    b, c, h, w = image.shape
    dither_matrix = generate_bayer_matrix(n).to(image.device).type_as(image)
    m = dither_matrix.shape[0]
    repeat_h = (h + m - 1) // m
    repeat_w = (w + m - 1) // m
    tiled_dither = dither_matrix.repeat(repeat_h, repeat_w)[:h, :w]
    tiled_dither = tiled_dither.view(1, 1, h, w)
    halftone = DifferentiableDither.apply(image, tiled_dither)
    if norm:
        halftone = normer.encodes(halftone)
    return halftone

class HalftoneOperator:
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return ordered_dithering(data)
    def transpose(self, data: torch.Tensor) -> torch.Tensor:
        return data

# ---- 5. Apply halftoning and save result ----
# Process image for halftoning, run ordered dithering, and save output image
preprocess_ht = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])
with Image.open(image_path).convert("RGB") as img:
    x_01 = preprocess_ht(img).unsqueeze(0).to(device)  # [B, C, H, W]
normer = Normer()
x = normer.encodes(x_01)  # [-1,1]
operator = HalftoneOperator()
y = operator.forward(x)   # [-1,1]

# Save halftoned image to outputs/halftone.png
output_folder = "./outputs"
os.makedirs(output_folder, exist_ok=True)
halftone_img = normer.decodes(y).clamp(0, 1)
halftone_filename = os.path.join(output_folder, "halftone.png")
save_image(halftone_img, halftone_filename)
print(f"Halftoned image saved to: {halftone_filename}")
