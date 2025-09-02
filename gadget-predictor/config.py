"""
Configuration file for the MLOps pipeline
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic.functional_validators import field_validator


class Config(BaseSettings):
    """Configuration settings for the MLOps pipeline"""

    # Project settings
    PROJECT_NAME: str = "gadget-predictor"
    VERSION: str = "1.0.0"

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "datasets"
    MODEL_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Data settings
    CLASSES: List[str] = [
        "camera",
        "headphones",
        "laptop",
        "smartphone",
        "smartwatch",
        "tablet",
    ]
    TRAIN_PCT: float = 0.7
    VALID_PCT: float = 0.2
    TEST_PCT: float = 0.1
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 64

    # Training settings
    LEARNING_RATE: float = 0.02
    EPOCHS: int = 3
    ARCHITECTURE: str = "resnet18"
    PRETRAINED: bool = True

    # MLOps settings
    WANDB_PROJECT: str = "gadgets-predictor"
    WANDB_ENTITY: str = "prinzz-personal"
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "gadget-classification"

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # Model deployment
    MODEL_NAME: str = "gadget_classifier_model"
    MODEL_VERSION: str = "latest"
    CONFIDENCE_THRESHOLD: float = 0.5

    # Data validation
    MIN_IMAGES_PER_CLASS: int = 10
    MAX_IMAGE_SIZE_MB: float = 10.0
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]

    # Data splitting strategy
    USE_FASTAI_SPLITTING: bool = True  # Use FastAI's built-in splitting by default
    MANUAL_SPLIT_SEED: int = 42  # For reproducible splits when using manual splitting
    CREATE_TEST_SET: bool = True  # Whether to create a separate test set

    # Data augmentation settings
    USE_AUGMENTATION: bool = True  # Enable data augmentation
    AUGMENTATION_STRATEGY: str = "fastai"  # "fastai", "offline", or "hybrid"

    # FastAI augmentation settings (recommended)
    FASTAI_AUG_TRANSFORMS: List[str] = [
        "flip_lr",  # Horizontal flip
        "rotate",  # Rotation (-30 to +30 degrees)
        "zoom",  # Random zoom (1.0 to 1.1)
        "lighting",  # Lighting changes
        "contrast",  # Contrast adjustments
    ]

    # Offline augmentation settings
    OFFLINE_AUG_MULTIPLIER: int = 3  # How many augmented versions per original image
    OFFLINE_AUG_TRANSFORMS: Dict[str, Dict] = {
        "rotation": {"range": (-30, 30)},
        "brightness": {"range": (0.8, 1.2)},
        "contrast": {"range": (0.8, 1.2)},
        "saturation": {"range": (0.8, 1.2)},
        "horizontal_flip": {"probability": 0.5},
        "vertical_flip": {"probability": 0.1},
        "gaussian_blur": {"probability": 0.3, "sigma": (0.1, 2.0)},
        "gaussian_noise": {"probability": 0.3, "std": (0.01, 0.05)},
    }

    # Monitoring
    PROMETHEUS_PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @field_validator("MODEL_DIR", "LOGS_DIR", pre=True, always=True)
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v


# Global config instance
config = Config()
