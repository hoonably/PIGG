"""
Global GUI Agent Fine-tuning Script
Multi-user supervised fine-tuning of Qwen3-VL model following official approach.

Train: users 11-73
Val: users 74-83
Test: users 1-10

Usage:
    python 2_finetune_global.py 0 --lora  # LoRA fine-tuning on GPU 0
    python 2_finetune_global.py 0,1       # Full fine-tuning on GPUs 0,1
"""

import os
import sys
import argparse
from datetime import datetime
from scripts.train_qwen3 import train_lora, train_full
from scripts.eval_qwen3 import eval_user
from utils.eval import save_results
from config import USER_TEST, USER_TRAIN, USER_VAL, CHECKPOINT_GLOBAL_LORA, CHECKPOINT_GLOBAL_FULL


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


def finetune_global_agent(use_lora=True, log_file=None):
    """
    Fine-tune a global GUI agent on multi-user data.
    
    Following official Qwen3-VL training approach:
    - apply_chat_template with tokenize=True
    - Assistant-only label masking
    - Gradient checkpointing for memory efficiency
    """
    print("=" * 80)
    print("Qwen3-VL Global GUI Agent Fine-tuning")
    print("=" * 80)
    print(f"Training users: {len(USER_TRAIN)} users")
    if USER_TRAIN:
        print(f"  User IDs: {USER_TRAIN[0]}-{USER_TRAIN[-1]}")
    print(f"Method: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    if log_file:
        print(f"Log file: {log_file}")
    print()
    
    # Training - save to local SSD on the root filesystem
    checkpoint_dir = CHECKPOINT_GLOBAL_LORA if use_lora else CHECKPOINT_GLOBAL_FULL
    
    print(f"Training will save to: {checkpoint_dir}")
    print(f"  (Using /tmp partition with more disk space)")
    print()
    
    if use_lora:
        train_lora(checkpoint_dir, USER_TRAIN, USER_VAL)
    else:
        train_full(checkpoint_dir, USER_TRAIN, USER_VAL)
    
    print(f"\n✓ Training completed! Model saved to: {checkpoint_dir}")
    
    return checkpoint_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fine-tune Qwen3-VL global agent on multi-user data'
    )
    parser.add_argument(
        'gpu', type=str, nargs='?', default='0',
        help='GPU device(s) to use (e.g., "0" or "0,1")'
    )
    parser.add_argument(
        '--lora', action='store_true',
        help='Use LoRA fine-tuning (default: full fine-tuning)'
    )
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU(s): {args.gpu}\n")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = "lora" if args.lora else "full"
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/finetune_global_{method}_{timestamp}.log"
    
    # Redirect stdout to both terminal and log file
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training started")
    print(f"Log file: {log_file}\n")
    
    try:
        finetune_global_agent(use_lora=args.lora, log_file=log_file)
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
