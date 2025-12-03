"""
Personalized GUI Agent Fine-tuning + Evaluation Script
K-fold personalized fine-tuning on top of global checkpoint, with immediate evaluation.

Starting from global checkpoint, fine-tune on individual user's data using K-fold.
For each user (1-10), split their data into K=5 folds, train on 4 folds, and evaluate on 1 fold.

Usage:
    python 4_finetune_person.py 0,1,2 --lora                          # GPU 0,1 / All users / LoRA on global full
    python 4_finetune_person.py 0 --users 1                         # GPU 0 / User 1 / Full FT on global full
    python 4_finetune_person.py 0 --users 1 2 3 --lora              # GPU 0 / Users 1,2,3 / LoRA on global full
"""

import os
import sys
import argparse
import random
from datetime import datetime
import pickle
import torch
import json
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Trainer, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset
from typing import List, Dict, Any

from utils.train_utils import (
    Qwen3VLDataCollator,
    get_training_arguments,
    prepare_model_for_training
)
from utils.kfold import create_k_folds, get_training_folds, get_test_fold
from utils.eval import save_results, parse_inference
from scripts.eval_qwen3 import eval_user
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
    PERSON_LR,
    PERSON_WARMUP_STEPS,
    PERSON_WEIGHT_DECAY,
    PERSON_SAVE_STEPS,
    PERSON_EVAL_STEPS,
    PERSON_LORA_RANK,
    PERSON_LORA_ALPHA,
    USER_TEST
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


class FoldEvalDataset:
    """Evaluation dataset for a specific fold's test set"""
    
    def __init__(self, samples):
        self.samples = samples
        self.gt_contents = []
        self.gt_coords = []
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            trust_remote_code=True,
            max_pixels=MAX_PIXELS
        )
        self._extract_ground_truth()
    
    def _extract_ground_truth(self):
        """Extract ground truth from samples"""
        for sample in self.samples:
            for msg in sample:
                if msg['role'] == 'assistant':
                    gt_text = msg['content'][0]['text'] if isinstance(msg['content'], list) else msg['content']
                    gt_content, gt_x, gt_y = parse_inference(gt_text)
                    self.gt_contents.append(gt_content)
                    self.gt_coords.append((gt_x, gt_y))
                    break
    
    def __getitem__(self, idx: int):
        """Returns (processed_inputs, gt_content, gt_coord) for evaluation"""
        messages = self.samples[idx]
        
        # Convert first message to user role
        user_message = messages[0].copy()
        user_message['role'] = 'user'
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Extract image path
        image_path = None
        for content_item in user_message['content']:
            if content_item['type'] == 'image':
                image_path = content_item['image']
                break
        
        # Process
        processed = self.processor(
            text=[text],
            images=[image_path] if image_path else None,
            return_tensors="pt",
            padding=True
        )
        
        return processed, self.gt_contents[idx], self.gt_coords[idx]
    
    def __len__(self):
        return len(self.samples)


def get_user_folds(user_id: int, k_folds: int = K_FOLDS):
    """Get consistent K-folds for a user across training and evaluation"""
    user_samples = load_user_data(user_id, DATASET_ROOT)
    # Use fixed seed for reproducible fold splits
    folds = create_k_folds(user_samples, k=k_folds, seed=42)
    return folds, user_samples


def eval_fold(user_id: int, fold_idx: int, checkpoint_path: str, folds: list, is_lora: bool = False):
    """Evaluate a single fold's model on its test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating Fold {fold_idx}")
    print(f"{'='*60}")
    
    # Use pre-created folds to ensure consistency
    fold_samples = get_test_fold(folds, fold_idx)
    print(f"Test samples: {len(fold_samples)}")
    
    # Create evaluation dataset
    eval_dataset = FoldEvalDataset(fold_samples)
    
    # Evaluate
    save_root = f"./eval_results/personalized/user_{user_id}/fold_{fold_idx}"
    result = eval_user(
        checkpoint_path,
        [user_id],
        save_root,
        is_lora=is_lora,
        eval_dataset=eval_dataset
    )
    
    print(f"\nFold {fold_idx} Results:")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Mean L2: {result['mean_l2']:.2f}")
    print(f"  Median L2: {result['median_l2']:.2f}")
    
    # Save results
    save_results(result, save_root)
    
    # Clean up evaluation model from memory - AGGRESSIVE
    print("\nCleaning up evaluation objects...")
    import gc
    
    # Delete local variables
    if 'eval_dataset' in locals():
        del eval_dataset
    if 'result' in locals():
        pass  # Keep result to return
    
    # Multiple cleanup rounds
    for i in range(3):
        torch.cuda.empty_cache()
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print("Evaluation cleanup completed.")
    
    return result


def load_user_data(user_id: int, data_path: str) -> List:
    """Load all samples for a specific user"""
    pkl_path = os.path.join(data_path, f"user_{user_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"User data not found: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        user_data = pickle.load(f)
    
    return user_data


def finetune_personalized(
    user_id: int,
    fold_idx: int,
    use_lora: bool = True,
    log_file: str = None
):
    """
    Fine-tune personalized model for a specific user and fold.
    Always uses global full checkpoint as base.
    
    Args:
        user_id: Target user ID (1-10)
        fold_idx: Fold index to hold out (0 to K-1)
        use_lora: Whether to use LoRA for personalized fine-tuning
        log_file: Optional log file path
    """
    print("=" * 80)
    print("Personalized GUI Agent Fine-tuning")
    print("=" * 80)
    print(f"User ID: {user_id}")
    print(f"Fold: {fold_idx + 1}/{K_FOLDS} (holding out fold {fold_idx})")
    print(f"Method: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    print(f"Base checkpoint: Global Full")
    if log_file:
        print(f"Log file: {log_file}")
    print()
    
    # 1. Load user data and create K-folds (consistent with evaluation)
    print(f"Loading data for user {user_id}...")
    folds, user_samples = get_user_folds(user_id, K_FOLDS)
    print(f"Total samples: {len(user_samples)}")
    
    print(f"\nCreating {K_FOLDS}-fold split...")
    for i, fold in enumerate(folds):
        status = "[TEST]" if i == fold_idx else "[TRAIN]"
        print(f"  Fold {i}: {len(fold)} samples {status}")
    
    # Get training samples (all folds except test fold)
    training_samples = get_training_folds(folds, fold_idx)
    print(f"\nTraining samples: {len(training_samples)}")
    print(f"Test samples: {len(folds[fold_idx])} (fold {fold_idx})")
    
    # Dynamically adjust epochs based on training data size
    # if len(training_samples) >= 600:
    #     person_epochs = 1
    #     print(f"Large dataset ({len(training_samples)} samples) → Using {person_epochs} epoch")
    # elif len(training_samples) >= 300:
    #     person_epochs = 2
    #     print(f"Medium dataset ({len(training_samples)} samples) → Using {person_epochs} epochs")
    # else:
    #     person_epochs = 3
    #     print(f"Small dataset ({len(training_samples)} samples) → Using {person_epochs} epochs")
    person_epochs = PERSON_EPOCHS
    print()
    
    # 2. Determine base checkpoint and output directory
    base_checkpoint = CHECKPOINT_GLOBAL_FULL
    
    if use_lora:
        output_dir = os.path.join(CHECKPOINT_PERSON_LORA, f"user_{user_id}", f"fold_{fold_idx}")
    else:
        output_dir = os.path.join(CHECKPOINT_PERSON_FULL, f"user_{user_id}", f"fold_{fold_idx}")
    
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 3. Load processor from original model (processor unchanged during fine-tuning)
    print("Loading processor from original model...")
    from config import MODEL_NAME
    processor = Qwen3VLProcessor.from_pretrained(
        MODEL_NAME,
        max_pixels=MAX_PIXELS
    )
    
    # 4. Load model from base checkpoint (always from global full checkpoint)
    print("Loading global full fine-tuned model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if use_lora:
        # Add LoRA on top of global full checkpoint
        print("  Adding LoRA adapters for personalization...")
        lora_config = LoraConfig(
            r=PERSON_LORA_RANK,
            lora_alpha=PERSON_LORA_ALPHA,
            target_modules="all-linear",
            # target_modules=[
            #     "q_proj", "k_proj", "v_proj", "o_proj",  # Attention only
            #     "up_proj", "down_proj", "gate_proj"  # MLP
            # ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    model = prepare_model_for_training(model)
    
    # Freeze vision encoder and aligner according to official best practices
    print("\nFreezing vision encoder and aligner...")
    if hasattr(model, 'vision_tower') or hasattr(model, 'visual'):
        # Freeze vision encoder
        vision_tower = getattr(model, 'vision_tower', None) or getattr(model, 'visual', None)
        if vision_tower is not None:
            for param in vision_tower.parameters():
                param.requires_grad = False
            print("  Vision encoder frozen")
    
    # Freeze aligner/projector
    if hasattr(model, 'multi_modal_projector'):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False
        print("  Multi-modal projector (aligner) frozen")
    elif hasattr(model, 'mm_projector'):
        for param in model.mm_projector.parameters():
            param.requires_grad = False
        print("  MM projector (aligner) frozen")
    
    # For PEFT models, also check base model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model
        
        # Freeze vision components in base model
        if hasattr(base_model, 'vision_tower') or hasattr(base_model, 'visual'):
            vision_tower = getattr(base_model, 'vision_tower', None) or getattr(base_model, 'visual', None)
            if vision_tower is not None:
                for param in vision_tower.parameters():
                    param.requires_grad = False
                print("  Base model vision encoder frozen")
        
        if hasattr(base_model, 'multi_modal_projector'):
            for param in base_model.multi_modal_projector.parameters():
                param.requires_grad = False
            print("  Base model multi-modal projector frozen")
        elif hasattr(base_model, 'mm_projector'):
            for param in base_model.mm_projector.parameters():
                param.requires_grad = False
            print("  Base model mm projector frozen")
    
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
        eval_strategy="no",  # Disable evaluation during training
        eval_steps=None,
        save_strategy="no",  # Disable intermediate checkpointing
        save_steps=None,
        num_train_epochs=person_epochs,  # Use dynamic epochs
        per_device_train_batch_size=PERSON_BATCH_SIZE,
        gradient_accumulation_steps=PERSON_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=PERSON_LR,
        warmup_steps=PERSON_WARMUP_STEPS,
        weight_decay=PERSON_WEIGHT_DECAY,
        save_total_limit=1,  # Only keep final model
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
    
    # 11. Clean up training objects to free memory
    print("\nCleaning up training memory...")
    del trainer, model
    if 'base_model' in locals():
        del base_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # 12. Immediately evaluate this fold
    print("\n" + "=" * 80)
    print("EVALUATING TRAINED MODEL")
    print("=" * 80)
    
    fold_result = eval_fold(user_id, fold_idx, output_dir, folds, is_lora=use_lora)
    
    # 13. Clean up evaluation memory
    print("\nCleaning up evaluation memory...")
    
    # Force delete all model-related objects
    import gc
    if 'eval_dataset' in locals():
        del eval_dataset
    
    # Multiple rounds of cleanup
    for i in range(3):
        torch.cuda.empty_cache()
        gc.collect()
        if i < 2:
            import time
            time.sleep(2)
    
    # Verify memory cleanup
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory still allocated: {memory_allocated:.2f} GB")
        
        # If still too much memory allocated, force more aggressive cleanup
        if memory_allocated > 5.0:
            print("High memory usage detected - forcing aggressive cleanup...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            import time
            time.sleep(5)
            torch.cuda.empty_cache()
    
    print("Memory cleanup completed.")
    
    return output_dir, fold_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Personalized fine-tuning on top of global checkpoint'
    )
    parser.add_argument(
        'gpu', type=str,
        help='GPU device(s) to use (e.g., "0" or "0,1")'
    )
    parser.add_argument(
        '--users', '-u', type=int, nargs='+', default=None,
        help='User IDs to train (e.g., --users 1 2 3). If not specified, trains all test users (1-10).'
    )
    parser.add_argument(
        'user_id', type=int, nargs='?', default=None,
        help='Single user ID to train on (1-10). Alternative to --users. If neither specified, trains all users.'
    )
    parser.add_argument(
        '--fold', type=int, default=None,
        help='Specific fold index to train (0-4). If not specified, trains all folds.'
    )
    parser.add_argument(
        '--lora', action='store_true',
        help='Use LoRA for personalized fine-tuning (default: full fine-tuning)'
    )
    
    args = parser.parse_args()
    
    # Determine which users to train
    if args.users:
        user_list = args.users
    elif args.user_id:
        user_list = [args.user_id]
    else:
        # Default: all test users (1-10)
        user_list = USER_TEST
        print(f"No users specified, training all test users: {user_list}\n")
    
    # Validate user IDs
    for uid in user_list:
        if uid not in range(1, 11):
            print(f"Error: user_id must be between 1 and 10, got {uid}")
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
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/finetune_person_{method}_{timestamp}", exist_ok=True)
    
    # Track all results for summary
    all_user_results = []
    
    # Train each user
    for user_id in user_list:
        print("\n" + "=" * 80)
        print(f"PROCESSING USER {user_id}")
        print("=" * 80 + "\n")
        
        user_fold_results = []
        
        # Train each fold
        for fold_idx in folds_to_train:
            # Pre-cleanup before starting fold - CRITICAL for preventing OOM
            print(f"\n{'='*80}")
            print(f"[Pre-fold cleanup] Preparing for User {user_id}, Fold {fold_idx}...")
            print(f"{'='*80}")
            
            import gc
            import time
            
            # Aggressive memory cleanup
            print("Step 1: Initial cleanup...")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3)
            
            print("Step 2: Synchronize and cleanup...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3)
            
            print("Step 3: Final cleanup round...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Check and verify available memory
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                memory_free = memory_total - memory_reserved
                
                print(f"\nGPU Memory Status:")
                print(f"  Allocated: {memory_allocated:.2f} GB")
                print(f"  Reserved:  {memory_reserved:.2f} GB")
                print(f"  Free:      {memory_free:.2f} GB")
                print(f"  Total:     {memory_total:.2f} GB")
                
                # If not enough free memory, force more aggressive cleanup
                if memory_free < 30 or memory_allocated > 5:
                    print(f"\n⚠️  WARNING: Insufficient free memory ({memory_free:.1f} GB free, {memory_allocated:.2f} GB allocated)")
                    print("Forcing extended cleanup sequence...")
                    
                    for round_num in range(5):
                        print(f"  Cleanup round {round_num + 1}/5...")
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(4)
                    
                    # Final check
                    memory_allocated_after = torch.cuda.memory_allocated() / (1024**3)
                    memory_free_after = memory_total - torch.cuda.memory_reserved() / (1024**3)
                    print(f"  After cleanup: {memory_free_after:.2f} GB free, {memory_allocated_after:.2f} GB allocated")
                    
                    if memory_free_after < 25:
                        print(f"\n❌ ERROR: Still insufficient memory after cleanup!")
                        print(f"   This may cause OOM. Consider:")
                        print(f"   1. Reducing batch size or gradient accumulation")
                        print(f"   2. Running fewer users simultaneously")
                        print(f"   3. Restarting the script")
                        print(f"\n   Waiting 20 seconds before attempting to continue...\n")
                        time.sleep(20)
                else:
                    print(f"\n✓ Memory check passed. Sufficient free memory available.")
            
            print(f"{'='*80}\n")
            time.sleep(2)  # Final brief wait
            
            log_file = f"{log_dir}/finetune_person_{method}_{timestamp}/user{user_id}_fold{fold_idx}.log"
            
            # Redirect stdout to both terminal and log file
            tee = TeeLogger(log_file)
            sys.stdout = tee
            sys.stderr = tee
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training started")
            print(f"Log file: {log_file}\n")
            
            try:
                checkpoint_path, fold_result = finetune_personalized(
                    user_id=user_id,
                    fold_idx=fold_idx,
                    use_lora=args.lora,
                    log_file=log_file
                )
                user_fold_results.append(fold_result)
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training & Evaluation completed successfully!")
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
        
        # Calculate averaged metrics for this user
        if user_fold_results:
            avg_accuracy = np.mean([r['accuracy'] for r in user_fold_results])
            avg_mean_l2 = np.mean([r['mean_l2'] for r in user_fold_results])
            avg_median_l2 = np.mean([r['median_l2'] for r in user_fold_results])
            
            std_accuracy = np.std([r['accuracy'] for r in user_fold_results])
            std_mean_l2 = np.std([r['mean_l2'] for r in user_fold_results])
            std_median_l2 = np.std([r['median_l2'] for r in user_fold_results])
            
            user_summary = {
                "user_id": user_id,
                "n_folds": len(user_fold_results),
                "avg_accuracy": float(avg_accuracy),
                "avg_mean_l2": float(avg_mean_l2),
                "avg_median_l2": float(avg_median_l2),
                "std_accuracy": float(std_accuracy),
                "std_mean_l2": float(std_mean_l2),
                "std_median_l2": float(std_median_l2),
            }
            all_user_results.append(user_summary)
            
            print("\n" + "=" * 80)
            print(f"USER {user_id} K-FOLD SUMMARY")
            print("=" * 80)
            print(f"Evaluated {len(user_fold_results)}/{K_FOLDS} folds:")
            print(f"  Accuracy:   {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"  Mean L2:    {avg_mean_l2:.2f} ± {std_mean_l2:.2f}")
            print(f"  Median L2:  {avg_median_l2:.2f} ± {std_median_l2:.2f}")
            print()
            
            # Save user summary
            save_root = f"./eval_results/personalized/"
            os.makedirs(save_root, exist_ok=True)
            with open(os.path.join(save_root, f"user_{user_id}_summary.json"), "w") as f:
                json.dump(user_summary, f, indent=4)
            print(f"User summary saved to: {save_root}/user_{user_id}_summary.json\n")
    
    # Print overall summary if multiple users
    if len(all_user_results) > 1:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY ACROSS ALL USERS")
        print("=" * 80)
        
        overall_acc = np.mean([r['avg_accuracy'] for r in all_user_results])
        overall_mean_l2 = np.mean([r['avg_mean_l2'] for r in all_user_results])
        overall_median_l2 = np.mean([r['avg_median_l2'] for r in all_user_results])
        
        print(f"Evaluated {len(all_user_results)} users:")
        print(f"  Overall Accuracy:   {overall_acc:.4f}")
        print(f"  Overall Mean L2:    {overall_mean_l2:.2f}")
        print(f"  Overall Median L2:  {overall_median_l2:.2f}")
        print()
        
        print("Per-user breakdown:")
        for result in all_user_results:
            print(f"  User {result['user_id']}: "
                  f"Acc={result['avg_accuracy']:.4f}, "
                  f"Mean L2={result['avg_mean_l2']:.2f}, "
                  f"Median L2={result['avg_median_l2']:.2f}")
        
        # Save overall summary
        save_root = f"./eval_results/personalized_summary/{method}"
        overall_summary = {
            "n_users": len(all_user_results),
            "overall_accuracy": float(overall_acc),
            "overall_mean_l2": float(overall_mean_l2),
            "overall_median_l2": float(overall_median_l2),
            "user_results": all_user_results
        }
        with open(os.path.join(save_root, "overall_summary.json"), "w") as f:
            json.dump(overall_summary, f, indent=4)
        print(f"\nOverall summary saved to: {save_root}/overall_summary.json")
