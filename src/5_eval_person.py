"""
Evaluate personalized models trained with K-fold
Evaluates each fold's model on its held-out test set and reports averaged metrics.

Usage:
    python 5_eval_person.py 0 1 --lora --global_lora  # User 1, LoRA from global LoRA
    python 5_eval_person.py 0 2 --global_full          # User 2, Full FT from global full    # User 2, Full FT 개인화, Global Full checkpoint 기반, fold 0만 학습
    python 5_eval_person.py 0 --users 1 2 3 --lora     # Multiple users                      # User 3, LoRA 개인화, Global Full checkpoint 기반
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
from scripts.eval_qwen3 import eval_user
from utils.eval import save_results, evaluate_coordinates
from config import (
    USER_TEST,
    CHECKPOINT_PERSON_LORA,
    CHECKPOINT_PERSON_FULL,
    K_FOLDS,
    DATASET_ROOT
)
from dataset import Qwen3VLEvalDataset
import json
import numpy as np


def load_fold_data(user_id: int, fold_idx: int, data_path: str):
    """
    Load test data for a specific fold.
    Returns samples from the held-out fold.
    """
    import random
    
    # Load all user data
    pkl_path = os.path.join(data_path, f"user_{user_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"User data not found: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        user_data = pickle.load(f)
    
    # Recreate the same folds (must use same seed as training)
    random.seed(42)
    shuffled_samples = user_data.copy()
    random.shuffle(shuffled_samples)
    
    fold_size = len(shuffled_samples) // K_FOLDS
    folds = []
    
    for i in range(K_FOLDS):
        if i == K_FOLDS - 1:
            fold = shuffled_samples[i * fold_size:]
        else:
            fold = shuffled_samples[i * fold_size:(i + 1) * fold_size]
        folds.append(fold)
    
    # Return the test fold
    return folds[fold_idx]


class FoldEvalDataset:
    """
    Evaluation dataset for a specific fold's test set.
    Similar to Qwen3VLEvalDataset but for a specific fold.
    """
    def __init__(self, samples):
        from transformers import AutoProcessor
        from config import MAX_PIXELS
        from utils.eval import parse_inference
        
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
        from utils.eval import parse_inference
        
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
    """
    Evaluate a single fold's model on its test set.
    
    Args:
        user_id: User ID
        fold_idx: Fold index (0 to K-1)
        checkpoint_path: Path to the fold's checkpoint
        is_lora: Whether the checkpoint is LoRA
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Fold {fold_idx}")
    print(f"{'='*60}")
    
    # Load test data for this fold
    print(f"Loading test data for fold {fold_idx}...")
    fold_samples = load_fold_data(user_id, fold_idx, DATASET_ROOT)
    print(f"Test samples: {len(fold_samples)}")
    
    # Create evaluation dataset
    eval_dataset = FoldEvalDataset(fold_samples)
    
    # Evaluate
    save_root = f"./eval_results/personalized/user_{user_id}/fold_{fold_idx}"
    result = eval_user(
        checkpoint_path,
        [user_id],  # Not actually used since we pass eval_dataset
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
    
    return result


def eval_personalized_user(
    user_id: int,
    use_lora: bool = True,
    from_global_lora: bool = True,
    gpu: str = "0"
):
    """
    Evaluate all K folds for a user and report averaged metrics.
    
    Args:
        user_id: User ID (1-10)
        use_lora: Whether personalized model used LoRA
        from_global_lora: Whether it was based on global LoRA checkpoint
        gpu: GPU device to use
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    print("=" * 80)
    print("Personalized Model Evaluation")
    print("=" * 80)
    print(f"User ID: {user_id}")
    print(f"Method: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    print(f"Base: {'Global LoRA' if from_global_lora else 'Global Full'}")
    print(f"K-Folds: {K_FOLDS}")
    print(f"GPU: {gpu}")
    print()
    
    # Determine checkpoint base directory
    if use_lora:
        checkpoint_base = os.path.join(CHECKPOINT_PERSON_LORA, f"user_{user_id}")
    else:
        checkpoint_base = os.path.join(CHECKPOINT_PERSON_FULL, f"user_{user_id}")
    
    # Evaluate each fold
    fold_results = []
    for fold_idx in range(K_FOLDS):
        checkpoint_path = os.path.join(checkpoint_base, f"fold_{fold_idx}")
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for fold {fold_idx}: {checkpoint_path}")
            print(f"Skipping fold {fold_idx}")
            continue
        
        print(f"\nCheckpoint: {checkpoint_path}")
        result = eval_fold(user_id, fold_idx, checkpoint_path, is_lora=use_lora)
        fold_results.append(result)
    
    # Calculate averaged metrics
    if not fold_results:
        print("\nError: No fold results available!")
        return None
    
    print("\n" + "=" * 80)
    print("K-Fold Averaged Results")
    print("=" * 80)
    
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_mean_l2 = np.mean([r['mean_l2'] for r in fold_results])
    avg_median_l2 = np.mean([r['median_l2'] for r in fold_results])
    
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    std_mean_l2 = np.std([r['mean_l2'] for r in fold_results])
    std_median_l2 = np.std([r['median_l2'] for r in fold_results])
    
    print(f"User {user_id} - {len(fold_results)}/{K_FOLDS} folds evaluated:")
    print(f"  Accuracy:   {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Mean L2:    {avg_mean_l2:.2f} ± {std_mean_l2:.2f}")
    print(f"  Median L2:  {avg_median_l2:.2f} ± {std_median_l2:.2f}")
    print()
    
    # Individual fold breakdown
    print("Individual Fold Results:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i}: Acc={result['accuracy']:.4f}, "
              f"Mean L2={result['mean_l2']:.2f}, "
              f"Median L2={result['median_l2']:.2f}")
    
    # Save averaged results
    averaged_result = {
        "user_id": user_id,
        "n_folds_evaluated": len(fold_results),
        "avg_accuracy": float(avg_accuracy),
        "avg_mean_l2": float(avg_mean_l2),
        "avg_median_l2": float(avg_median_l2),
        "std_accuracy": float(std_accuracy),
        "std_mean_l2": float(std_mean_l2),
        "std_median_l2": float(std_median_l2),
        "fold_results": fold_results
    }
    
    # Save to file
    method = "lora" if use_lora else "full"
    base = "global_lora" if from_global_lora else "global_full"
    save_root = f"./eval_results/personalized_summary/{method}_{base}"
    os.makedirs(save_root, exist_ok=True)
    
    with open(os.path.join(save_root, f"user_{user_id}_summary.json"), "w") as f:
        json.dump(averaged_result, f, indent=4)
    
    print(f"\nSummary saved to: {save_root}/user_{user_id}_summary.json")
    
    return averaged_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate personalized K-fold models'
    )
    parser.add_argument(
        'gpu', type=str,
        help='GPU device to use (e.g., "0")'
    )
    parser.add_argument(
        '--users', '-u', type=int, nargs='+', default=None,
        help='User IDs to evaluate (default: all test users 1-10)'
    )
    parser.add_argument(
        '--lora', action='store_true',
        help='Evaluate LoRA personalized models (default: full fine-tuned)'
    )
    parser.add_argument(
        '--global_lora', action='store_true',
        help='Models were based on global LoRA checkpoint (default: global full)'
    )
    
    args = parser.parse_args()
    
    # Determine users to evaluate
    if args.users:
        user_list = args.users
    else:
        user_list = USER_TEST
    
    print(f"Using GPU: {args.gpu}\n")
    
    # Evaluate each user
    all_results = []
    for user_id in user_list:
        if user_id not in range(1, 11):
            print(f"Warning: user_id {user_id} not in test set (1-10), skipping")
            continue
        
        result = eval_personalized_user(
            user_id=user_id,
            use_lora=args.lora,
            from_global_lora=args.global_lora,
            gpu=args.gpu
        )
        
        if result:
            all_results.append(result)
        
        print("\n" + "=" * 80 + "\n")
    
    # Print overall summary across all users
    if len(all_results) > 1:
        print("=" * 80)
        print("OVERALL SUMMARY ACROSS ALL USERS")
        print("=" * 80)
        
        overall_acc = np.mean([r['avg_accuracy'] for r in all_results])
        overall_mean_l2 = np.mean([r['avg_mean_l2'] for r in all_results])
        overall_median_l2 = np.mean([r['avg_median_l2'] for r in all_results])
        
        print(f"Evaluated {len(all_results)} users:")
        print(f"  Overall Accuracy:   {overall_acc:.4f}")
        print(f"  Overall Mean L2:    {overall_mean_l2:.2f}")
        print(f"  Overall Median L2:  {overall_median_l2:.2f}")
        print()
        
        print("Per-user breakdown:")
        for result in all_results:
            print(f"  User {result['user_id']}: "
                  f"Acc={result['avg_accuracy']:.4f}, "
                  f"Mean L2={result['avg_mean_l2']:.2f}, "
                  f"Median L2={result['avg_median_l2']:.2f}")
        
        # Save overall summary
        method = "lora" if args.lora else "full"
        base = "global_lora" if args.global_lora else "global_full"
        save_root = f"./eval_results/personalized_summary/{method}_{base}"
        
        overall_summary = {
            "n_users": len(all_results),
            "overall_accuracy": float(overall_acc),
            "overall_mean_l2": float(overall_mean_l2),
            "overall_median_l2": float(overall_median_l2),
            "user_results": all_results
        }
        
        with open(os.path.join(save_root, "overall_summary.json"), "w") as f:
            json.dump(overall_summary, f, indent=4)
        
        print(f"\nOverall summary saved to: {save_root}/overall_summary.json")
