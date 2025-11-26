"""
Evaluate a trained checkpoint on test users
Similar to 1_vanilla_eval.py but for fine-tuned models

Usage:
    python 3_eval_checkpoint.py 0 --cf 500         # Full fine-tuned checkpoint-500
    python 3_eval_checkpoint.py 0 --cl 500         # LoRA checkpoint-500
    python 3_eval_checkpoint.py 0 --cf 500 --users 1 2 3
    python 3_eval_checkpoint.py 0 --checkpoint /path/to/checkpoint  # Custom path
"""

import os
import sys
import argparse
from datetime import datetime
from scripts.eval_qwen3 import eval_user
from utils.eval import save_results
from config import USER_TEST, CHECKPOINT_GLOBAL_LORA, CHECKPOINT_GLOBAL_FULL
import json


def eval_checkpoint(checkpoint_path, user_list, gpu, is_lora=False):
    """
    Evaluate a checkpoint on specified users.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        user_list: List of user IDs to evaluate
        gpu: GPU device to use
        is_lora: Whether the checkpoint is LoRA
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    print("=" * 80)
    print("Checkpoint Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Users to evaluate: {user_list}")
    print(f"GPU: {gpu}")
    print(f"Model type: {'LoRA' if is_lora else 'Full'}")
    print()
    
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Evaluate each user (following 1_vanilla_eval.py pattern)
    for user_id in user_list:
        save_root = f"./eval_results/{checkpoint_name}/user_{user_id}"
        result = eval_user(checkpoint_path, [user_id], save_root, is_lora=is_lora)
        print(f'===== User {user_id} =====')
        print(result)
        save_results(result, save_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate a fine-tuned checkpoint on test users'
    )
    parser.add_argument(
        'gpu', type=str,
        help='GPU device to use (e.g., "0")'
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, default=None,
        help='Path to checkpoint directory (custom path)'
    )
    parser.add_argument(
        '--cf', type=int, default=None,
        help='Full fine-tuned checkpoint number (e.g., 1 for checkpoint-1)'
    )
    parser.add_argument(
        '--cl', type=int, default=None,
        help='LoRA checkpoint number (e.g., 500 for checkpoint-500)'
    )
    parser.add_argument(
        '--users', '-u', type=int, nargs='+', default=None,
        help='User IDs to evaluate (default: all test users 1-10)'
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path and type
    is_lora = False
    checkpoint_path = None
    
    if args.checkpoint:
        # Custom checkpoint path
        checkpoint_path = args.checkpoint
        print(f"Using custom checkpoint: {checkpoint_path}")
        # Try to infer if it's LoRA from path
        if 'lora' in checkpoint_path.lower():
            is_lora = True
    elif args.cf is not None:
        # Full fine-tuned checkpoint
        checkpoint_path = f"{CHECKPOINT_GLOBAL_FULL}/checkpoint-{args.cf}"
        is_lora = False
        print(f"Using full checkpoint: {checkpoint_path}")
    elif args.cl is not None:
        # LoRA checkpoint
        checkpoint_path = f"{CHECKPOINT_GLOBAL_LORA}/checkpoint-{args.cl}"
        is_lora = True
        print(f"Using LoRA checkpoint: {checkpoint_path}")
    else:
        print("Error: Must specify either --checkpoint, --cf, or --cl")
        sys.exit(1)
    
    # Determine users to evaluate
    if args.users:
        user_list = args.users
    else:
        user_list = USER_TEST
    
    print(f"Using GPU: {args.gpu}\n")
    
    eval_checkpoint(checkpoint_path, user_list, args.gpu, is_lora=is_lora)
