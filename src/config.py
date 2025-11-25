"""
Configuration for Qwen3-VL Training
Following official Qwen3-VL recommendations
"""

# Training hyperparameters (Qwen3-VL recommended)
LR = 1e-6  # Learning rate: 1e-6 to 2e-7 recommended
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 8

# Data paths
DATASET_ROOT = "./dataset_pkl"
RAW_DATASET_ROOT = "../dataset/fingertip-20k"

# Model settings
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 1280 * 28 * 28  # Image resolution

# Special tokens and constants (from Qwen3-VL)
IGNORE_INDEX = -100  # For label masking (non-trainable tokens)

# User splits
USER_TEST = list(range(1, 11))      # 1-10: Test users
USER_TRAIN = list(range(11, 74))    # 11-73: Training users
USER_VAL = list(range(74, 84))      # 74-83: Validation users

# Training settings
NUM_WORKERS = 4
SAVE_STEPS = 1000
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
