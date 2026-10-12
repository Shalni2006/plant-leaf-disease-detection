"""
Central Configuration for Plant Leaf Disease Detection
Defines all dataset paths, model hyperparameters, and reproducible seeds.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "model"
STATIC_DIR = PROJECT_ROOT / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"

# Artifact file paths
MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
CLASS_INDICES_PATH = MODEL_DIR / "class_indices.json"
DISEASE_INFO_PATH = PROJECT_ROOT / "disease_info.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"
EVALUATION_METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

# Image and Pipeline Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
NUM_CHANNELS = 3
BATCH_SIZE = 32
SEED = 42

# Reproducible Stratified Split Ratios (Sum must equal 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Hyperparameters
NUM_CLASSES = 15
LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2
DROPOUT_RATE = 0.4
