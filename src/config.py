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

# Checkpoint paths
CHECKPOINT = "/workspace/PIGG_checkpoints"
CHECKPOINT_GLOBAL_LORA = "/workspace/PIGG_checkpoints/global_agent_lora"
CHECKPOINT_GLOBAL_FULL = "/workspace/PIGG_checkpoints/global_agent_full"
CHECKPOINT_PERSON_LORA = "/workspace/PIGG_checkpoints/personalized_lora"
CHECKPOINT_PERSON_FULL = "/workspace/PIGG_checkpoints/personalized_full"

# Model settings
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 1280 * 28 * 28  # Image resolution

# Special tokens and constants (from Qwen3-VL)
IGNORE_INDEX = -100  # For label masking (non-trainable tokens)

# User splits
USER_TEST = list(range(1, 11))      # 1-10: Test users
USER_TRAIN = list(range(11, 84))    # 11-83: Training users
USER_VAL = []                       # Validation disabled (OOM issues)

# Training settings
NUM_WORKERS = 4
SAVE_STEPS = 500
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01

# Personalized training settings (conservative for small datasets)
K_FOLDS = 3
PERSON_EPOCHS = 3
PERSON_BATCH_SIZE = 1
PERSON_GRADIENT_ACCUMULATION_STEPS = 4  # Increased for stable gradients
PERSON_LR = 2e-5  # Higher than global but still conservative
PERSON_WARMUP_STEPS = 5  # Fewer warmup steps for small datasets
PERSON_WEIGHT_DECAY = 0.01
PERSON_SAVE_STEPS = 20  # Save more frequently for small datasets
PERSON_EVAL_STEPS = 10  # Evaluate frequently to catch overfitting
PERSON_LORA_RANK = 8  # Conservative rank for small data
PERSON_LORA_ALPHA = 32  # Lower alpha
