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
from config import USER_TEST, USER_TRAIN, USER_VAL


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
    print(f"Training users: {len(USER_TRAIN)} users (IDs {USER_TRAIN[0]}-{USER_TRAIN[-1]})")
    print(f"Validation users: {len(USER_VAL)} users (IDs {USER_VAL[0]}-{USER_VAL[-1]})")
    print(f"Test users: {len(USER_TEST)} users (IDs {USER_TEST[0]}-{USER_TEST[-1]})")
    print(f"Method: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    if log_file:
        print(f"Log file: {log_file}")
    print()
    
    # Training
    checkpoint_dir = (
        "./checkpoints/global_agent_lora" if use_lora 
        else "./checkpoints/global_agent_full"
    )
    
    print(f"Training will save to: {checkpoint_dir}")
    print(f"Validation will be performed every 1000 steps")
    print()
    
    if use_lora:
        train_lora(checkpoint_dir, USER_TRAIN, USER_VAL)
    else:
        train_full(checkpoint_dir, USER_TRAIN, USER_VAL)
    
    print(f"\n✓ Training completed! Best model saved to: {checkpoint_dir}")
    
    # Test on users 1-10
    print(f"\n{'='*80}")
    print("Testing on test users (1-10)")
    print('='*80)
    
    test_results = {}
    for user_id in USER_TEST:
        print(f"\nEvaluating User {user_id}...")
        save_root = f"./out/global_agent/test/user_{user_id}"
        result = eval_user(checkpoint_dir, [user_id], save_root, is_lora=use_lora)
        
        print(f"User {user_id} Results:")
        print(f"  Click Accuracy @ 14%: {result.get('accuracy', 'N/A')}")
        print(f"  Mean L2 Error: {result.get('mean_l2', 'N/A')}")
        print(f"  Median L2 Error: {result.get('median_l2', 'N/A')}")
        
        save_results(result, save_root)
        test_results[user_id] = result
    
    # Summary
    print("\n" + "=" * 80)
    print("GLOBAL AGENT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}\n")
    
    for user_id, result in test_results.items():
        print(f"User {user_id:2d}: "
              f"Acc={result.get('accuracy', 'N/A'):.3f}, "
              f"Mean L2={result.get('mean_l2', 'N/A'):.1f}, "
              f"Median L2={result.get('median_l2', 'N/A'):.1f}")
    
    # Calculate average
    avg_acc = sum(r.get('accuracy', 0) for r in test_results.values()) / len(test_results)
    avg_l2 = sum(r.get('mean_l2', 0) for r in test_results.values()) / len(test_results)
    print(f"\nAverage: Acc={avg_acc:.3f}, Mean L2={avg_l2:.1f}")
    
    return checkpoint_dir, test_results


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
