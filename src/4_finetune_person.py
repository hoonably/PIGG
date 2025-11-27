"""
Personalized GUI Agent Fine-tuning Script
K-fold personalized fine-tuning on top of global checkpoint.

Starting from global checkpoint, fine-tune on individual user's data using K-fold.
For each user (1-10), split their data into K=5 folds and train on 4 folds.

Usage:
    python 4_finetune_person.py 0 1 --lora --global_lora  # User 1, LoRA, from global LoRA checkpoint
    python 4_finetune_person.py 0 1 --global_full          # User 1, Full FT, from global full checkpoint
    python 4_finetune_person.py 0 2 --lora --global_lora   # User 2, LoRA, from global LoRA checkpoint
"""

import os
import sys
import argparse
import random
from datetime import datetime
import pickle
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset
from typing import List, Dict, Any

from utils.train_utils import (
    Qwen3VLDataCollator,
    get_training_arguments,
    prepare_model_for_training
)
from config import (
    MAX_PIXELS, 
    DATASET_ROOT,
    CHECKPOINT_GLOBAL_LORA,
    CHECKPOINT_GLOBAL_FULL,
    CHECKPOINT_PERSON_LORA,
    CHECKPOINT_PERSON_FULL,
    K_FOLDS,
    PERSON_BATCH_SIZE,
    PERSON_GRADIENT_ACCUMULATION_STEPS,
    PERSON_EPOCHS,
    PERSON_LR
)


class TeeLogger:
    """Write to both stdout and log file simultaneously"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', buffering=1)  # Line buffering
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class PersonalizedDataset(Dataset):
    """Dataset for a single fold of personalized training"""
    
    def __init__(self, samples: List[List[Dict[str, Any]]]):
        self.samples = samples
    
    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        """Returns messages in Qwen3-VL format with 'user' role for vision content"""
        messages = self.samples[idx]
        
        # Convert 'system' role to 'user' for Qwen3-VL compatibility
        converted_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                converted_messages.append({
                    'role': 'user',
                    'content': msg['content']
                })
            else:
                converted_messages.append(msg)
        
        return converted_messages
    
    def __len__(self) -> int:
        return len(self.samples)


def load_user_data(user_id: int, data_path: str) -> List:
    """Load all samples for a specific user"""
    pkl_path = os.path.join(data_path, f"user_{user_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"User data not found: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        user_data = pickle.load(f)
    
    return user_data


def create_k_folds(samples: List, k: int = 5, seed: int = 42) -> List[List]:
    """
    Split samples into K folds randomly.
    
    Returns:
        List of K folds, where each fold is a list of samples
    """
    random.seed(seed)
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    fold_size = len(shuffled_samples) // k
    folds = []
    
    for i in range(k):
        if i == k - 1:
            # Last fold gets remaining samples
            fold = shuffled_samples[i * fold_size:]
        else:
            fold = shuffled_samples[i * fold_size:(i + 1) * fold_size]
        folds.append(fold)
    
    return folds


def get_training_folds(folds: List[List], test_fold_idx: int) -> List:
    """
    Get training samples by combining all folds except the test fold.
    
    Args:
        folds: List of K folds
        test_fold_idx: Index of fold to hold out for testing (0 to K-1)
    
    Returns:
        Combined list of training samples
    """
    training_samples = []
    for i, fold in enumerate(folds):
        if i != test_fold_idx:
            training_samples.extend(fold)
    return training_samples


def finetune_personalized(
    user_id: int,
    fold_idx: int,
    use_lora: bool = True,
    from_global_lora: bool = True,
    log_file: str = None
):
    """
    Fine-tune personalized model for a specific user and fold.
    
    Args:
        user_id: Target user ID (1-10)
        fold_idx: Fold index to hold out (0 to K-1)
        use_lora: Whether to use LoRA for personalized fine-tuning
        from_global_lora: Whether to start from global LoRA checkpoint (vs full checkpoint)
        log_file: Optional log file path
    """
    print("=" * 80)
    print("Personalized GUI Agent Fine-tuning")
    print("=" * 80)
    print(f"User ID: {user_id}")
    print(f"Fold: {fold_idx + 1}/{K_FOLDS} (holding out fold {fold_idx})")
    print(f"Method: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    print(f"Base checkpoint: {'Global LoRA' if from_global_lora else 'Global Full'}")
    if log_file:
        print(f"Log file: {log_file}")
    print()
    
    # 1. Load user data and create K-folds
    print(f"Loading data for user {user_id}...")
    user_samples = load_user_data(user_id, DATASET_ROOT)
    print(f"Total samples: {len(user_samples)}")
    
    print(f"\nCreating {K_FOLDS}-fold split...")
    folds = create_k_folds(user_samples, k=K_FOLDS)
    for i, fold in enumerate(folds):
        status = "[TEST]" if i == fold_idx else "[TRAIN]"
        print(f"  Fold {i}: {len(fold)} samples {status}")
    
    # Get training samples (all folds except test fold)
    training_samples = get_training_folds(folds, fold_idx)
    print(f"\nTraining samples: {len(training_samples)}")
    print(f"Test samples: {len(folds[fold_idx])} (fold {fold_idx})")
    print()
    
    # 2. Determine base checkpoint and output directory
    base_checkpoint = CHECKPOINT_GLOBAL_LORA if from_global_lora else CHECKPOINT_GLOBAL_FULL
    
    if use_lora:
        output_dir = os.path.join(CHECKPOINT_PERSON_LORA, f"user_{user_id}", f"fold_{fold_idx}")
    else:
        output_dir = os.path.join(CHECKPOINT_PERSON_FULL, f"user_{user_id}", f"fold_{fold_idx}")
    
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 3. Load processor from base checkpoint
    print("Loading processor...")
    processor = Qwen3VLProcessor.from_pretrained(
        base_checkpoint,
        max_pixels=MAX_PIXELS
    )
    
    # 4. Load model from base checkpoint
    print("Loading model from base checkpoint...")
    
    if from_global_lora:
        # Load base model first, then apply LoRA weights
        print("  Loading base Qwen3-VL model...")
        from config import MODEL_NAME
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        print(f"  Loading global LoRA weights from {base_checkpoint}...")
        model = PeftModel.from_pretrained(base_model, base_checkpoint)
        
        if use_lora:
            # Add new LoRA adapters on top of global LoRA
            print("  Adding new LoRA adapters for personalization...")
            lora_config = LoraConfig(
                r=4,
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
        else:
            # Merge LoRA weights and do full fine-tuning
            print("  Merging LoRA weights for full fine-tuning...")
            model = model.merge_and_unload()
    else:
        # Load from global full checkpoint
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        if use_lora:
            # Add LoRA on top of full checkpoint
            print("  Adding LoRA adapters for personalization...")
            lora_config = LoraConfig(
                r=4,
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
    
    # Print trainable parameters
    if use_lora and hasattr(model, 'print_trainable_parameters'):
        print("\nTrainable parameters:")
        model.print_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print()
    
    # 5. Create dataset
    print("Creating training dataset...")
    train_dataset = PersonalizedDataset(training_samples)
    print(f"Dataset size: {len(train_dataset)}")
    
    # 6. Create data collator
    data_collator = Qwen3VLDataCollator(processor=processor)
    
    # 7. Setup training arguments (personalized settings)
    training_args = get_training_arguments(
        output_dir=output_dir,
        eval_strategy="no",
        eval_steps=None,
        num_train_epochs=PERSON_EPOCHS,
        per_device_train_batch_size=PERSON_BATCH_SIZE,
        gradient_accumulation_steps=PERSON_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=PERSON_LR,
    )
    
    # 8. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 9. Train!
    print("\n" + "=" * 80)
    print("Starting personalized training...")
    print("=" * 80)
    trainer.train()
    
    # 10. Save final model
    print("\nSaving personalized model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"\n✓ Personalized training completed!")
    print(f"  Model saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Personalized fine-tuning on top of global checkpoint'
    )
    parser.add_argument(
        'gpu', type=str,
        help='GPU device(s) to use (e.g., "0" or "0,1")'
    )
    parser.add_argument(
        'user_id', type=int,
        help='User ID to train on (1-10)'
    )
    parser.add_argument(
        '--fold', type=int, default=None,
        help='Specific fold index to train (0-4). If not specified, trains all folds.'
    )
    parser.add_argument(
        '--lora', action='store_true',
        help='Use LoRA for personalized fine-tuning (default: full fine-tuning)'
    )
    parser.add_argument(
        '--global_lora', action='store_true',
        help='Start from global LoRA checkpoint (default: global full checkpoint)'
    )
    
    args = parser.parse_args()
    
    # Validate user ID
    if args.user_id not in range(1, 11):
        print(f"Error: user_id must be between 1 and 10, got {args.user_id}")
        sys.exit(1)
    
    # Validate fold index if specified
    if args.fold is not None and args.fold not in range(K_FOLDS):
        print(f"Error: fold must be between 0 and {K_FOLDS-1}, got {args.fold}")
        sys.exit(1)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU(s): {args.gpu}\n")
    
    # Determine which folds to train
    if args.fold is not None:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(K_FOLDS))
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = "lora" if args.lora else "full"
    base = "global_lora" if args.global_lora else "global_full"
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Train each fold
    for fold_idx in folds_to_train:
        log_file = f"{log_dir}/finetune_person_{method}_{base}_user{args.user_id}_fold{fold_idx}_{timestamp}.log"
        
        # Redirect stdout to both terminal and log file
        tee = TeeLogger(log_file)
        sys.stdout = tee
        sys.stderr = tee
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training started")
        print(f"Log file: {log_file}\n")
        
        try:
            finetune_personalized(
                user_id=args.user_id,
                fold_idx=fold_idx,
                use_lora=args.lora,
                from_global_lora=args.global_lora,
                log_file=log_file
            )
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training completed successfully!")
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            tee.close()
            sys.stdout = tee.terminal
            sys.stderr = tee.terminal
            print(f"\nLog saved to: {log_file}")
        
        print()  # Blank line between folds
