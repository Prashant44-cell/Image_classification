# deep_image_analyzer/core/config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Paths
    LABELS_URL: str = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    OUTPUT_DIR: str = "./outputs"

    # Model & inference
    DEVICE: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    AMP_ENABLED: bool = True

    # Ensemble & TTA
    ENSEMBLE_SIZE: int = 3
    TTA_ITERATIONS: int = 5

    # Adversarial
    FGSM_EPSILON: float = 0.1
    PGD_ALPHA: float = 0.01
    PGD_STEPS: int = 10

    # Uncertainty
    MC_SAMPLES: int = 50

    # Halftone
    BAYER_ORDER: int = 3
