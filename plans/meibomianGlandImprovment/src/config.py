"""
Configuration settings for MGD-1k DeepLabv3+ training pipeline
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
LOGS_ROOT = PROJECT_ROOT / "logs"

# MGD-1k dataset paths
MGD1K_OFFICIAL = WORKSPACE_ROOT / "mgd1k-official" / "Expore MGD1k Dataset"
MGD1K_IMAGES = MGD1K_OFFICIAL / "Original Images"
MGD1K_EYELID_MASKS = MGD1K_OFFICIAL / "Eyelid Lebels"
MGD1K_GLAND_MASKS = MGD1K_OFFICIAL / "Meibomian Gland Labels"
MGD1K_MEIBOSCORE = MGD1K_OFFICIAL / "Graded Meiboscore"

# Image preprocessing
IMAGE_SIZE = 320  # Target image size (320x320)
CLAHE_CLIP_LIMIT = 2.0  # CLAHE contrast limit
CLAHE_TILE_SIZE = 8  # CLAHE tile size

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 100
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 20
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# Model architecture
BACKBONE = "resnet50"  # DeepLabv3+ backbone: resnet50, resnet101, etc.
OUTPUT_STRIDE = 16  # DeepLabv3+ output stride
ATROUS_RATES = [6, 12, 18]  # Atrous spatial pyramid pooling rates

# Device
DEVICE = "cuda"  # "cuda" or "cpu"

# Loss weights (for multi-task or weighted loss)
SEGMENTATION_WEIGHT = 1.0
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 1.0
FOREGROUND_LOSS_WEIGHT = 10.0
USE_EYELID_ROI = True
ROI_MARGIN = 0.05
USE_AUGMENTATION = True
PREDICTION_SAMPLES = 12

# Checkpoint and saving
SAVE_INTERVAL = 5  # Save checkpoint every N epochs
CHECKPOINT_DIR = MODELS_ROOT / "checkpoints"
CONFIG_DIR = MODELS_ROOT / "configs"

# Logging
LOG_DIR = LOGS_ROOT
TENSORBOARD_DIR = LOGS_ROOT / "tensorboard"

# Create directories if they don't exist
for directory in [CHECKPOINT_DIR, CONFIG_DIR, LOG_DIR, TENSORBOARD_DIR, RESULTS_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)
