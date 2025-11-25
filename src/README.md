# Qwen3-VL Fine-tuning (Clean Implementation)

This folder contains a clean re-implementation of Qwen3-VL fine-tuning following **official Qwen3-VL methodology**.

## 🎯 Key Features

- **Official Approach**: Follows Qwen3-VL's official training methodology
- **Proper Label Masking**: Assistant-only training (user inputs masked with -100)
- **Memory Efficient**: Gradient checkpointing, bf16, multi-GPU support
- **Clean Architecture**: Separated dataset, collator, and training logic

## 📂 Structure

```
src/
├── config.py                   # Configuration (hyperparameters, paths)
├── dataset.py                  # Dataset classes (train/eval)
├── 1_vanilla_eval.py          # Zero-shot evaluation
├── 2_finetune_global.py       # Global agent fine-tuning
├── scripts/
│   ├── train_qwen3.py         # Training functions (LoRA/Full)
│   └── eval_qwen3.py          # Evaluation functions
└── utils/
    ├── train_utils.py         # Data collator & training utilities
    └── eval.py                # Evaluation utilities (reused from src)
```

## 🚀 Usage

### 1. Vanilla Evaluation (Zero-shot)

```bash
# Evaluate users 1-10 on GPU 0
python 1_vanilla_eval.py 0 --u 1 10

# Evaluate specific users
python 1_vanilla_eval.py 0 --u 1 3 5
```

### 2. Global Agent Fine-tuning

```bash
# LoRA fine-tuning on GPU 1
python 2_finetune_global.py 1 --lora

# Full fine-tuning on GPUs 0,1
python 2_finetune_global.py 0,1
```

## 🔧 Technical Details

### Data Collator (`Qwen3VLDataCollator`)

Following official Qwen3-VL implementation:

1. **Input Processing**: Uses `apply_chat_template(tokenize=True)` for proper preprocessing
2. **Label Creation**: Masks all tokens except assistant responses
   - User inputs: `label = -100` (ignored in loss)
   - Assistant responses: `label = input_ids` (trained)
3. **Vision Handling**: Automatically processes images via `pixel_values` and `image_grid_thw`

### Label Masking Strategy

```python
# Token structure:
<|im_start|>user\n...user content...<|im_end|>\n         # labels = -100
<|im_start|>assistant\n...RESPONSE...<|im_end|>\n       # labels = input_ids

# Only RESPONSE tokens are trained
```

### Training Configuration

- **Learning Rate**: 1e-6 (Qwen3-VL recommended)
- **Batch Size**: 1 per device (effective 8 with gradient accumulation)
- **Precision**: bfloat16
- **Memory**: Gradient checkpointing enabled
- **LoRA Config**: r=4, alpha=32, targeting q/k/v/o/gate/up/down projections

## 📊 Experiments

### Experiment Setup

- **Training**: Users 11-73 (63 users)
- **Validation**: Users 74-83 (10 users)
- **Test**: Users 1-10 (10 users)

### Running Full Pipeline

```bash
# 1. Vanilla baseline
python 1_vanilla_eval.py 0 --u 1 10

# 2. Train global agent
python 2_finetune_global.py 0 --lora

# Results saved to:
# - checkpoints/global_agent_lora/
# - out/global_agent/test/user_*/
```

## 🔬 Key Differences from src/

1. **Clean Collator**: Properly implements official Qwen3-VL label masking
2. **No Deprecated Code**: Removed old/unused preprocessing functions
3. **Documented**: Each component has clear docstrings
4. **Reuses Evaluation**: Imports working eval logic from `src/`

## 📖 References

- Official Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen2-VL Training: https://github.com/QwenLM/Qwen2-VL
