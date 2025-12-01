"""
Personalized GUI Agent Fine-tuning + Evaluation Script
K-fold personalized fine-tuning on top of global checkpoint, with immediate evaluation.

Starting from global checkpoint, fine-tune on individual user's data using K-fold.
For each user (1-10), split their data into K=5 folds, train on 4 folds, and evaluate on 1 fold.

Usage:
    python 4_finetune_person.py 1,2 --lora                          # GPU 1,2 / All users / from global full checkpoint
    python 4_finetune_person.py 0 --users 1 --lora --global_lora    # User 1, LoRA, from global LoRA checkpoint
    python 4_finetune_person.py 0 --users 1 2 3 --lora              # Multiple users
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


def eval_fold(user_id: int, fold_idx: int, checkpoint_path: str, is_lora: bool = False):
    """Evaluate a single fold's model on its test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating Fold {fold_idx}")
    print(f"{'='*60}")
    
    # Load test data for this fold
    print(f"Loading test data for fold {fold_idx}...")
    user_samples = load_user_data(user_id, DATASET_ROOT)
    folds = create_k_folds(user_samples, k=K_FOLDS, seed=42)
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
    
    # Clean up evaluation model from memory
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
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
    
    # 3. Load processor from original model (processor unchanged during fine-tuning)
    print("Loading processor from original model...")
    from config import MODEL_NAME
    processor = Qwen3VLProcessor.from_pretrained(
        MODEL_NAME,
        max_pixels=MAX_PIXELS
    )
    
    # 4. Load model from base checkpoint
    print("Loading fine-tuned model from base checkpoint...")
    
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
                r=4,  #! 랭크
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
    
    fold_result = eval_fold(user_id, fold_idx, output_dir, is_lora=use_lora)
    
    # 13. Clean up evaluation memory
    print("\nCleaning up evaluation memory...")
    torch.cuda.empty_cache()
    gc.collect()
    
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
    parser.add_argument(
        '--global_lora', action='store_true',
        help='Start from global LoRA checkpoint (default: global full checkpoint)'
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
    base = "global_lora" if args.global_lora else "global_full"
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
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
            log_file = f"{log_dir}/finetune_person_{method}_{base}_user{user_id}_fold{fold_idx}_{timestamp}.log"
            
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
                    from_global_lora=args.global_lora,
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
            save_root = f"./eval_results/personalized_summary/{method}_{base}"
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
        save_root = f"./eval_results/personalized_summary/{method}_{base}"
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
