import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODEL = MODELS_DIR / "pretrained/yolov8n.pt"
TRAINED_MODEL = MODELS_DIR / "trained/best.pt"

# Data paths
DATA_DIR = PROJECT_ROOT / "data/processed"