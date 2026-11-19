import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Classes (Canonically mapped)
CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma"
]

DISPLAY_NAMES = {
    "adenocarcinoma": "Adenocarcinoma",
    "large_cell_carcinoma": "Large Cell Carcinoma",
    "normal": "Normal (Healthy)",
    "squamous_cell_carcinoma": "Squamous Cell Carcinoma"
}

NUM_CLASSES = len(CLASS_NAMES)

# Model Architecture Parameters
IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (128, 128, 3)
PROJECTION_FILTERS = 128

# Training Hyperparameters
BATCH_SIZE = 32
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 30
TOTAL_EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2

# Checkpoint paths
BEST_MODEL_PATH = MODELS_DIR / "best_ms_gf_vgg16.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_ms_gf_vgg16.keras"
METRICS_PATH = RESULTS_DIR / "metrics.json"
HISTORY_PATH = RESULTS_DIR / "history.json"
ABLATION_PATH = RESULTS_DIR / "ablation.json"

