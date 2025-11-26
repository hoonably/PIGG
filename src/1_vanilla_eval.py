"""
Qwen3-VL Vanilla Evaluation Script
Evaluate zero-shot performance of the vanilla Qwen3-VL model on specified users.

Usage:
    python 1_vanilla_eval.py 0           # Evaluate users on GPU 0 (default 1-10)
    python 1_vanilla_eval.py 0 --u 1 10  # Evaluate users 1-10 on GPU 0
"""

import os
import argparse
from utils.eval import save_results
from scripts.eval_qwen3 import eval_user


def vanilla_eval(user_list):
    """Evaluate vanilla Qwen3-VL on specified users"""
    for i in user_list:
        save_root = f"./eval_results/qwen3_vanilla/user_{i}"
        result = eval_user("Qwen/Qwen3-VL-8B-Instruct", [i], save_root, is_lora=False)
        print(f'===== User {i} =====')
        print(result)
        save_results(result, save_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate vanilla Qwen3-VL model for specified users on given GPU(s)'
    )
    parser.add_argument(
        'gpu', type=str, nargs='?', default='0',
        help='GPU device(s) to use (e.g., "0" or "0,1")'
    )
    parser.add_argument(
        '--u', '--user_list', type=int, nargs='+',
        default=list(range(1, 11)),
        help='List of user IDs or an inclusive range (e.g. --u 1 3 -> [1,2,3])',
        dest='user_list'
    )
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Parse user list
    raw = args.user_list
    if isinstance(raw, list) and len(raw) == 2:
        start, end = raw
        if start <= end:
            user_list = list(range(start, end + 1))
        else:
            user_list = list(range(start, end - 1, -1))
    else:
        user_list = raw
    
    vanilla_eval(user_list)
