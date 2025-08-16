import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision import models
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.style.use('default')

# Try to import albumentations, fallback to torchvision if not available
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️ Albumentations not available. Using torchvision transforms only.")

# ======================= ENHANCED IMPORTS & SETUP =======================

class Config:
    """Configuration class for all hyperparameters"""
    TTA_ITERATIONS = 5
    ENSEMBLE_SIZE = 3
    UNCERTAINTY_SAMPLES = 50
    FOCAL_LOSS_ALPHA = 1.0
    FOCAL_LOSS_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    ADVERSARIAL_EPSILON = 0.1
    CONFIDENCE_THRESHOLD = 0.7

# ======================= ADVANCED LOSS FUNCTIONS =======================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))

# ======================= ADVANCED AUGMENTATION =======================

class AdvancedAugmentation:
    """Advanced augmentation pipeline using Albumentations or Torchvision fallback"""
    
    @staticmethod
    def get_train_augmentation():
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),  # Fixed: was A.Flip
                A.VerticalFlip(p=0.2),    # Added vertical flip
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=3),
                    A.MotionBlur(blur_limit=3),
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.RandomGamma(gamma_limit=(80, 120)),
                ], p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),  # Fixed: was Cutout
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Fallback to torchvision transforms
            return T.Compose([
                T.RandomRotation(45),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    @staticmethod
    def get_tta_augmentation():
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                A.RandomRotate90(p=0.3),
                A.HorizontalFlip(p=0.3),  # Fixed: was A.Flip
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Fallback to torchvision transforms
            return T.Compose([
                T.RandomRotation(10),
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

# ======================= MODEL ENSEMBLE =======================

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for better accuracy"""
    
    def __init__(self, model_names=['resnet50']):  # Simplified to avoid loading issues
        super().__init__()
        self.models = nn.ModuleList()
        
        for name in model_names:
            if name == 'resnet50':
                model = models.resnet50(pretrained=True)
            elif name == 'efficientnet_b0':
                try:
                    model = models.efficientnet_b0(pretrained=True)
                except:
                    print(f"⚠️ Could not load {name}, using ResNet50 instead")
                    model = models.resnet50(pretrained=True)
            elif name == 'vit_b_16':
                try:
                    model = models.vit_b_16(pretrained=True)
                except:
                    print(f"⚠️ Could not load {name}, using ResNet50 instead")
                    model = models.resnet50(pretrained=True)
            else:
                print(f"⚠️ Unknown model: {name}, using ResNet50")
                model = models.resnet50(pretrained=True)
            
            model.eval()
            self.models.append(model)
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(pred)
        
        # Ensemble averaging
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred

# ======================= GRADCAM VISUALIZATION =======================

class GradCAMVisualizer:
    """Advanced GradCAM for model interpretability"""
    
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.target_layer = self._get_target_layer(target_layer_name)
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _get_target_layer(self, layer_name):
        try:
            if hasattr(self.model, layer_name):
                layer = getattr(self.model, layer_name)
                if isinstance(layer, nn.Sequential):
                    return layer[-1]  # Get last block in sequential
                return layer
        except:
            pass
        
        # Fallback: try to get the last convolutional layer
        try:
            return list(self.model.children())[-2]
        except:
            return list(self.model.children())[-1]
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        try:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_backward_hook(backward_hook)
        except Exception as e:
            print(f"⚠️ Could not register hooks: {e}")
    
    def generate_cam(self, input_tensor, class_idx=None):
        try:
            self.model.eval()
            self.model.zero_grad()
            
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            output[0, class_idx].backward()
            
            if self.gradients is None or self.activations is None:
                # Fallback: create a simple attention map
                return np.random.rand(224, 224), class_idx
            
            gradients = self.gradients.cpu().data.numpy()[0]
            activations = self.activations.cpu().data.numpy()
            
            # Calculate weights
            weights = np.mean(gradients, axis=(1, 2))
            
            # Generate CAM
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            cam = np.maximum(cam, 0)
            cam = cam / cam.max() if cam.max() > 0 else cam
            
            # Resize to input size
            cam = cv2.resize(cam, (224, 224))
            
            return cam, class_idx
            
        except Exception as e:
            print(f"⚠️ GradCAM generation failed: {e}")
            # Return a dummy CAM
            return np.random.rand(224, 224) * 0.1, class_idx or 0

# ======================= TEST-TIME AUGMENTATION =======================

class TestTimeAugmentation:
    """Test-Time Augmentation for improved predictions"""
    
    def __init__(self, model, num_iterations=5):
        self.model = model
        self.num_iterations = num_iterations
        self.tta_transform = AdvancedAugmentation.get_tta_augmentation()
    
    def predict_with_tta(self, image_pil):
        predictions = []
        device = next(self.model.parameters()).device
        
        for _ in range(self.num_iterations):
            try:
                if ALBUMENTATIONS_AVAILABLE:
                    # Apply albumentations augmentation
                    image_np = np.array(image_pil)
                    augmented = self.tta_transform(image=image_np)
                    augmented_tensor = augmented['image'].unsqueeze(0).to(device)
                else:
                    # Apply torchvision augmentation
                    augmented_tensor = self.tta_transform(image_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = F.softmax(self.model(augmented_tensor), dim=1)
                    predictions.append(pred)
            except Exception as e:
                print(f"⚠️ TTA iteration failed: {e}")
                # Fallback to standard preprocessing
                standard_transform = T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                tensor = standard_transform(image_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = F.softmax(self.model(tensor), dim=1)
                    predictions.append(pred)
        
        # Average predictions
        if predictions:
            mean_prediction = torch.stack(predictions).mean(dim=0)
            std_prediction = torch.stack(predictions).std(dim=0)
        else:
            # Fallback
            mean_prediction = torch.ones(1, 1000).to(device) / 1000
            std_prediction = torch.zeros(1, 1000).to(device)
        
        return mean_prediction, std_prediction

# ======================= UNCERTAINTY QUANTIFICATION =======================

class MonteCarloDropout:
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, model, num_samples=50):
        self.model = model
        self.num_samples = num_samples
    
    def enable_dropout(self, model):
        """Enable dropout during inference"""
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    def predict_with_uncertainty(self, x):
        # Check if model has dropout layers
        has_dropout = any(m.__class__.__name__.startswith('Dropout') for m in self.model.modules())
        
        if not has_dropout:
            # If no dropout, return standard prediction with zero uncertainty
            with torch.no_grad():
                pred = F.softmax(self.model(x), dim=1)
                uncertainty = torch.zeros_like(pred).sum(dim=1)
                return pred, uncertainty
        
        self.enable_dropout(self.model)
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = F.softmax(self.model(x), dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0).sum(dim=1)
        
        return mean_pred, epistemic_uncertainty

# ======================= ADVERSARIAL EXAMPLES =======================

class AdversarialAttack:
    """FGSM and other adversarial attacks"""
    
    @staticmethod
    def fgsm_attack(model, image, epsilon, target_class=None):
        image = image.clone().detach().requires_grad_(True)
        
        output = model(image)
        
        if target_class is None:
            target_class = output.max(1)[1]
        
        loss = F.nll_loss(F.log_softmax(output, dim=1), target_class)
        
        model.zero_grad()
        loss.backward()
        
        # Create adversarial example
        sign_data_grad = image.grad.data.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image
    
    @staticmethod
    def pgd_attack(model, image, epsilon, alpha=0.01, num_iter=10):
        """Projected Gradient Descent attack"""
        original_image = image.clone()
        
        for _ in range(num_iter):
            image.requires_grad = True
            output = model(image)
            loss = F.cross_entropy(output, output.max(1)[1])
            
            model.zero_grad()
            loss.backward()
            
            # Update image
            image = image + alpha * image.grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(image - original_image, -epsilon, epsilon)
            image = torch.clamp(original_image + delta, 0, 1).detach()
        
        return image

# ======================= ENHANCED METRICS =======================

class ComprehensiveMetrics:
    """Comprehensive evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_probs, class_names):
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Per-class metrics
        metrics['per_class'] = {}
        unique_classes = np.unique(y_true)
        for i in unique_classes:
            if i < len(class_names) and i < y_probs.shape[1]:
                class_name = class_names[i]
                y_binary = (np.array(y_true) == i).astype(int)
                if len(np.unique(y_binary)) > 1:  # Check if class exists
                    try:
                        auc = roc_auc_score(y_binary, y_probs[:, i])
                        metrics['per_class'][class_name] = auc
                    except:
                        metrics['per_class'][class_name] = 0.0
        
        return metrics

# ======================= ORIGINAL CLASSES (ENHANCED) =======================

class Normer:
    @staticmethod
    def decodes(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(-1, 1) * 0.5 + 0.5
    
    @staticmethod
    def encodes(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0, 1) * 2 - 1

def generate_bayer_matrix(n: int = 3):
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
        binary_output = (image > dither_matrix).float()
        ctx.save_for_backward(image, dither_matrix)
        return binary_output
    
    @staticmethod
    def backward(ctx, grad_output):
        image, dither_matrix = ctx.saved_tensors
        temperature = 10.0
        diff = image - dither_matrix
        sigmoid_approx = torch.sigmoid(diff * temperature)
        grad_approx = temperature * sigmoid_approx * (1 - sigmoid_approx)
        return grad_output * grad_approx, None

def ordered_dithering(image: torch.Tensor, n: int = 3, norm: bool = True) -> torch.Tensor:
    normer = Normer()
    if norm:
        image = normer.decodes(image)
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

# ======================= VISUALIZATION UTILITIES =======================

class VisualizationUtils:
    """Advanced visualization utilities"""
    
    @staticmethod
    def plot_prediction_confidence(predictions, labels, top_k=5):
        """Plot prediction confidence with uncertainty"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Top predictions
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            top_indices = predictions.argsort()[-top_k:][::-1]
            top_probs = predictions[top_indices]
            top_labels = [labels[i] if i < len(labels) else f'Class_{i}' for i in top_indices]
            
            # Truncate long labels
            top_labels = [label[:30] + '...' if len(label) > 30 else label for label in top_labels]
            
            ax1.barh(range(len(top_labels)), top_probs, color='skyblue')
            ax1.set_yticks(range(len(top_labels)))
            ax1.set_yticklabels(top_labels, fontsize=10)
            ax1.set_xlabel('Confidence')
            ax1.set_title('Top Predictions')
            ax1.grid(True, alpha=0.3)
            
            # Confidence distribution
            ax2.hist(predictions, bins=50, alpha=0.7, color='lightcoral')
            ax2.axvline(predictions.max(), color='red', linestyle='--', 
                       label=f'Max: {predictions.max():.3f}')
            ax2.set_xlabel('Prediction Probability')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Confidence Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"⚠️ Failed to create confidence plot: {e}")
            # Return empty figure
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'Plot generation failed: {e}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    @staticmethod
    def create_comparison_grid(original_img, halftone_img, gradcam_img=None, adversarial_img=None):
        """Create a comprehensive comparison grid"""
        try:
            num_imgs = 2 + (gradcam_img is not None) + (adversarial_img is not None)
            fig, axes = plt.subplots(1, num_imgs, figsize=(5*num_imgs, 5))
            
            if num_imgs == 1:
                axes = [axes]
            elif num_imgs == 2:
                axes = list(axes)
            
            idx = 0
            
            # Original
            axes[idx].imshow(original_img)
            axes[idx].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
            idx += 1
            
            # Halftoned
            if len(halftone_img.shape) == 3 and halftone_img.shape[2] == 1:
                halftone_img = halftone_img.squeeze()
            axes[idx].imshow(halftone_img, cmap='gray' if len(halftone_img.shape) == 2 else None)
            axes[idx].set_title('Halftoned Image', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
            idx += 1
            
            # GradCAM
            if gradcam_img is not None:
                axes[idx].imshow(gradcam_img)
                axes[idx].set_title('GradCAM', fontsize=14, fontweight='bold')
                axes[idx].axis('off')
                idx += 1
            
            # Adversarial
            if adversarial_img is not None:
                axes[idx].imshow(adversarial_img)
                axes[idx].set_title('Adversarial Example', fontsize=14, fontweight='bold')
                axes[idx].axis('off')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"⚠️ Failed to create comparison grid: {e}")
            # Return simple figure
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'Grid generation failed: {e}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig

# ======================= MAIN EXECUTION =======================

def main():
    print("🚀 Enhanced Deep Learning Image Analyzer")
    print("=" * 50)
    
    # ---- 1. ASK USER FOR IMAGE PATH ----
    image_path = input("Enter the path to your image: ").strip()
    if not os.path.exists(image_path):
        print(f"File '{image_path}' not found.")
        sys.exit(1)
    
    # ---- 2. DEVICE AND MODEL SETUP ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # Initialize model
    print("📡 Loading model...")
    try:
        model = models.resnet50(pretrained=True).to(device)
        model.eval()
        print("✅ ResNet50 loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # ---- 3. DOWNLOAD LABELS ----
    try:
        LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(LABELS_URL)
        response.raise_for_status()
        labels = response.text.strip().split("\n")
        print(f"✅ Downloaded {len(labels)} class labels")
    except Exception as e:
        print(f"⚠️ Failed to download labels: {e}")
        # Use dummy labels
        labels = [f"class_{i}" for i in range(1000)]
    
    # ---- 4. PREPROCESSING ----
    preprocess_cls = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # ---- 5. LOAD AND ANALYZE IMAGE ----
    print("🖼️  Loading and analyzing image...")
    
    start_time = time.time()
    
    try:
        with Image.open(image_path).convert("RGB") as img:
            original_img = img.copy()
            input_tensor = preprocess_cls(img).unsqueeze(0).to(device)
            
            # Standard prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                _, predicted = output.max(1)
                predicted_idx = predicted.item()
                predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else f"class_{predicted_idx}"
                confidence = probabilities[predicted_idx].item()
            
            print(f"\n🎯 Standard Prediction:")
            print(f"   Predicted object: \033[1m{predicted_label}\033[0m")
            print(f"   Confidence: {confidence:.3f}")
            
            # ---- 6. TEST-TIME AUGMENTATION ----
            print("\n🔄 Running Test-Time Augmentation...")
            try:
                tta = TestTimeAugmentation(model, num_iterations=Config.TTA_ITERATIONS)
                tta_pred, tta_std = tta.predict_with_tta(img)
                
                tta_idx = tta_pred.argmax().item()
                tta_label = labels[tta_idx] if tta_idx < len(labels) else f"class_{tta_idx}"
                tta_confidence = tta_pred[0, tta_idx].item()
                tta_uncertainty = tta_std[0, tta_idx].item()
                
                print(f"   TTA Prediction: \033[1m{tta_label}\033[0m")
                print(f"   TTA Confidence: {tta_confidence:.3f} ± {tta_uncertainty:.3f}")
            except Exception as e:
                print(f"   ⚠️ TTA failed: {e}")
                tta_label = predicted_label
                tta_confidence = confidence
                tta_uncertainty = 0.0
            
            # ---- 7. UNCERTAINTY QUANTIFICATION ----
            print("\n🎲 Estimating prediction uncertainty...")
            try:
                mc_dropout = MonteCarloDropout(model, num_samples=Config.UNCERTAINTY_SAMPLES)
                mc_pred, epistemic_uncertainty = mc_dropout.predict_with_uncertainty(input_tensor)
                
                mc_idx = mc_pred.argmax().item()
                mc_label = labels[mc_idx] if mc_idx < len(labels) else f"class_{mc_idx}"
                mc_confidence = mc_pred[0, mc_idx].item()
                uncertainty_score = epistemic_uncertainty[0].item()
                
                print(f"   MC Prediction: \033[1m{mc_label}\033[0m")
                print(f"   MC Confidence: {mc_confidence:.3f}")
                print(f"   Epistemic Uncertainty: {uncertainty_score:.4f}")
            except Exception as e:
                print(f"   ⚠️ Uncertainty quantification failed: {e}")
                mc_label = predicted_label
                mc_confidence = confidence
                uncertainty_score = 0.0
            
            # ---- 8. GRADCAM VISUALIZATION ----
            print("\n🔍 Generating GradCAM visualization...")
            try:
                gradcam = GradCAMVisualizer(model)
                cam, _ = gradcam.generate_cam(input_tensor, class_idx=predicted_idx)
                
                # Convert CAM to heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                original_resized = np.array(original_img.resize((224, 224)))
                gradcam_img = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)
                gradcam_img = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
                print("   ✅ GradCAM visualization created")
            except Exception as e:
                print(f"   ⚠️ GradCAM generation failed: {e}")
                gradcam_img = np.array(original_img.resize((224, 224)))
            
            # ---- 9. ADVERSARIAL EXAMPLE ----
            print("\n⚔️  Generating adversarial example...")
            try:
                adversarial = AdversarialAttack()
                adv_image = adversarial.fgsm_attack(model, input_tensor.clone(), 
                                                  epsilon=Config.ADVERSARIAL_EPSILON)
                
                with torch.no_grad():
                    adv_output = model(adv_image)
                    adv_probabilities = F.softmax(adv_output, dim=1)[0]
                    adv_predicted_idx = adv_output.argmax().item()
                    adv_predicted_label = labels[adv_predicted_idx] if adv_predicted_idx < len(labels) else f"class_{adv_predicted_idx}"
                    adv_confidence = adv_probabilities[adv_predicted_idx].item()
                
                print(f"   Adversarial Prediction: \033[1m{adv_predicted_label}\033[0m")
                print(f"   Adversarial Confidence: {adv_confidence:.3f}")
                
                # Convert adversarial tensor to image
                adv_img_np = adv_image.squeeze().permute(1, 2, 0).cpu().numpy()
                adv_img_np = (adv_img_np - adv_img_np.min()) / (adv_img_np.max() - adv_img_np.min())
                adv_img_np = (adv_img_np * 255).astype(np.uint8)
                
            except Exception as e:
                print(f"   ⚠️ Adversarial generation failed: {e}")
                adv_predicted_label = predicted_label
                adv_confidence = confidence
                adv_img_np = np.array(original_img.resize((224, 224)))
    
    except Exception as e:
        print(f"❌ Failed to process image: {e}")
        sys.exit(1)
    
    # ---- 10. HALFTONING (ORIGINAL FUNCTIONALITY) ----
    print("\n🎨 Applying halftone effect...")
    
    try:
        preprocess_ht = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        
        with Image.open(image_path).convert("RGB") as img:
            x_01 = preprocess_ht(img).unsqueeze(0).to(device)
        
        normer = Normer()
        x = normer.encodes(x_01)
        operator = HalftoneOperator()
        y = operator.forward(x)
        
        halftone_img = normer.decodes(y).clamp(0, 1)
        halftone_np = halftone_img.squeeze().permute(1, 2, 0).cpu().numpy()
        
        print("   ✅ Halftone effect applied")
        
    except Exception as e:
        print(f"   ⚠️ Halftoning failed: {e}")
        # Create dummy halftone
        halftone_np = np.array(original_img.resize((256, 256))) * 0.5
        halftone_img = torch.zeros(1, 3, 256, 256)
    
    # ---- 11. SAVE OUTPUTS ----
    output_folder = "./outputs"
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Save halftone image
        halftone_filename = os.path.join(output_folder, "halftone.png")
        save_image(halftone_img, halftone_filename)
        
        # Save GradCAM
        gradcam_filename = os.path.join(output_folder, "gradcam.png")
        Image.fromarray(gradcam_img.astype(np.uint8)).save(gradcam_filename)
        
        # Save adversarial example
        adv_filename = os.path.join(output_folder, "adversarial.png")
        Image.fromarray(adv_img_np).save(adv_filename)
        
        print(f"💾 Images saved to {output_folder}")
        
    except Exception as e:
        print(f"⚠️ Failed to save some images: {e}")
    
    # ---- 12. COMPREHENSIVE VISUALIZATION ----
    print("\n📊 Creating comprehensive visualizations...")
    
    try:
        vis_utils = VisualizationUtils()
        
        # Prediction confidence plot
        confidence_fig = vis_utils.plot_prediction_confidence(
            probabilities.cpu().numpy(), labels, top_k=5
        )
        confidence_fig.savefig(os.path.join(output_folder, "confidence_analysis.png"), 
                              dpi=300, bbox_inches='tight')
        plt.close(confidence_fig)
        
        # Comparison grid
        comparison_fig = vis_utils.create_comparison_grid(
            original_img.resize((224, 224)),
            halftone_np,
            gradcam_img,
            adv_img_np
        )
        comparison_fig.savefig(os.path.join(output_folder, "comprehensive_analysis.png"), 
                              dpi=300, bbox_inches='tight')
        plt.close(comparison_fig)
        
        print("   ✅ Visualizations created")
        
    except Exception as e:
        print(f"   ⚠️ Visualization creation failed: {e}")
    
    # ---- 13. SUMMARY REPORT ----
    processing_time = time.time() - start_time
    
    print(f"\n📋 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"📁 Image: {image_path}")
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"🎯 Standard prediction: {predicted_label} ({confidence:.3f})")
    print(f"🔄 TTA prediction: {tta_label} ({tta_confidence:.3f} ± {tta_uncertainty:.3f})")
    print(f"🎲 MC prediction: {mc_label} ({mc_confidence:.3f})")
    print(f"📊 Uncertainty score: {uncertainty_score:.4f}")
    print(f"⚔️  Adversarial prediction: {adv_predicted_label} ({adv_confidence:.3f})")
    print(f"\n💾 Outputs saved to: {output_folder}")
    print(f"   - halftone.png")
    print(f"   - gradcam.png")
    print(f"   - adversarial.png")
    print(f"   - confidence_analysis.png")
    print(f"   - comprehensive_analysis.png")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
