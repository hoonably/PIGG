"""
Qwen3-VL Training Script
Following official Qwen3-VL fine-tuning approach
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Trainer
from peft import LoraConfig, get_peft_model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_NAME, DATASET_ROOT, MAX_PIXELS
from dataset import Qwen3VLTrainDataset
from utils.train_utils import (
    Qwen3VLDataCollator,
    get_training_arguments,
    prepare_model_for_training
)


def train_lora(checkpoint_dir: str, user_train: list, user_val: list = None):
    """
    Train Qwen3-VL with LoRA.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        user_train: List of training user IDs
        user_val: Optional list of validation user IDs
    """
    print("=" * 80)
    print("Training Qwen3-VL with LoRA")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Training users: {len(user_train)} users")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print()
    
    # 1. Load processor
    print("Loading processor...")
    processor = Qwen3VLProcessor.from_pretrained(
        MODEL_NAME,
        max_pixels=MAX_PIXELS
    )
    
    # 2. Load model in bfloat16 for memory efficiency
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across available GPUs
    )
    
    # 3. Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model = prepare_model_for_training(model)
    
    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    print()
    
    # 4. Load dataset
    print("Loading dataset...")
    train_dataset = Qwen3VLTrainDataset(user_train, DATASET_ROOT)
    print(f"Training samples: {len(train_dataset)}")
    
    # Optional: Load validation dataset
    eval_dataset = None
    if user_val:
        eval_dataset = Qwen3VLTrainDataset(user_val, DATASET_ROOT)
        print(f"Validation samples: {len(eval_dataset)}")
    
    # 5. Create data collator
    data_collator = Qwen3VLDataCollator(processor=processor)
    
    # 6. Setup training arguments
    # Disable validation during training to avoid CUDA OOM during eval
    training_args = get_training_arguments(
        output_dir=checkpoint_dir,
        eval_strategy="no",  # Disable validation to prevent OOM
        eval_steps=None,
    )
    
    # 7. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 8. Train!
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()
    
    # 9. Save final model
    print("\nSaving final model...")
    trainer.save_model(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    print(f"\n✓ Training completed! Model saved to: {checkpoint_dir}")


def train_full(checkpoint_dir: str, user_train: list, user_val: list = None):
    """
    Full fine-tuning of Qwen3-VL (without LoRA).
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        user_train: List of training user IDs
        user_val: Optional list of validation user IDs
    """
    print("=" * 80)
    print("Full Fine-tuning Qwen3-VL")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Training users: {len(user_train)} users")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print()
    
    # 1. Load processor
    print("Loading processor...")
    processor = Qwen3VLProcessor.from_pretrained(
        MODEL_NAME,
        max_pixels=MAX_PIXELS
    )
    
    # 2. Load model
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # DO NOT freeze vision encoder for full fine-tuning
    # Vision features are critical for GUI grounding task
    
    model = prepare_model_for_training(model)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print()
    
    # 3. Load dataset
    print("Loading dataset...")
    train_dataset = Qwen3VLTrainDataset(user_train, DATASET_ROOT)
    print(f"Training samples: {len(train_dataset)}")
    
    eval_dataset = None
    if user_val:
        eval_dataset = Qwen3VLTrainDataset(user_val, DATASET_ROOT)
        print(f"Validation samples: {len(eval_dataset)}")
    
    # 4. Create data collator
    data_collator = Qwen3VLDataCollator(processor=processor)
    
    # 5. Setup training arguments
    # Disable validation during training to avoid CUDA OOM during eval
    # Validation can be done separately after training completes
    training_args = get_training_arguments(
        output_dir=checkpoint_dir,
        eval_strategy="no",  # Disable validation to prevent OOM
        eval_steps=None,
    )
    
    # 6. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 7. Train!
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()
    
    # 8. Save final model
    print("\nSaving final model...")
    trainer.save_model(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    print(f"\n✓ Training completed! Model saved to: {checkpoint_dir}")
