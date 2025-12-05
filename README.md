# PIGG
PIGG: Personalized Interactive GUI Grounding

## Installation

### 1. Create Conda Environment
```bash
conda create -n pigg python=3.10 -y
conda activate pigg 
```

### 2. Install PyTorch with CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Other Dependencies
```bash
pip install pillow transformers peft accelerate einops natsort qwen-vl-util scikit-learn
```

## Dataset
- `qinglongyang/fingertip-20k`
```bash
cd dataset
python download.py
python fixdir.py
```

## Model
This project uses **Qwen3-VL-8B-Instruct** model from Hugging Face.

---

## 📂 Project Structure

```
PIGG/
├── src/                          # Main source code
│   ├── config.py                 # Configuration (hyperparameters, paths)
│   ├── dataset.py                # Dataset classes (train/eval)
│   ├── 1_vanilla_eval.py         # Zero-shot evaluation
│   ├── 2_finetune_global.py      # Global agent fine-tuning
│   ├── 3_eval_checkpoint.py      # Checkpoint evaluation
│   ├── 4_finetune_person.py      # Personalized fine-tuning (K-fold)
│   ├── visualize_prediction.py   # Prediction visualization tool
│   ├── scripts/
│   │   ├── train_qwen3.py        # Training functions (LoRA/Full)
│   │   └── eval_qwen3.py         # Evaluation functions
│   └── utils/
│       ├── train_utils.py        # Data collator & training utilities
│       ├── eval.py               # Evaluation utilities
│       └── kfold.py              # K-fold split utilities
├── dataset/                      # fingertip-20k dataset
│   ├── download.py
│   └── fixdir.py
├── survey.py                     # Dataset analysis tool
└── README.md                     # This file
```

### Directory Descriptions

- **`src/`**: All training and evaluation code
- **`src/eval_results/`**: Evaluation outputs (JSON results, metrics)
  - `1_qwen3_vanilla/`: Vanilla model (zero-shot) results
  - `global_agent_full/`: Global full fine-tuned model results
  - `personalized/`: Personalized K-fold results per user
- **`src/logs/`**: Training logs (loss, learning rate, etc.)
- **`src/visualizations/`**: Generated prediction visualizations
- **`dataset/`**: Raw dataset files (fingertip-20k)
- **`dataset_pkl/`**: Preprocessed dataset (pickle format)

---

## 🚀 Usage

### 1. Vanilla Evaluation (Zero-shot)

Evaluate vanilla Qwen3-VL model without any fine-tuning.

```bash
cd src

# Evaluate users 1-10 on GPU 0
python 1_vanilla_eval.py 0 --u 1 10

# Evaluate specific users
python 1_vanilla_eval.py 0 --u 1 3 5
```

**Output**: `eval_results/1_qwen3_vanilla/user_{id}/`

---

### 2. Global Agent Fine-tuning

Train a global GUI agent on users 11-83.

```bash
cd src

# LoRA fine-tuning on GPU 0
python 2_finetune_global.py 0 --lora

# Full fine-tuning on multiple GPUs
python 2_finetune_global.py 0,1

# Custom users
python 2_finetune_global.py 0 --u 11 50 --lora
```

**Output**: 
- Checkpoint: `/workspace/PIGG_checkpoints/global_agent_lora/` or `global_agent_full/`
- Logs: `logs/finetune_global_*`

---

### 3. Checkpoint Evaluation

Evaluate any saved checkpoint on test users.

```bash
cd src

# Evaluate LoRA checkpoint
python 3_eval_checkpoint.py 0 \
    --checkpoint /workspace/PIGG_checkpoints/global_agent_lora \
    --u 1 10 \
    --lora

# Evaluate full checkpoint
python 3_eval_checkpoint.py 0 \
    --checkpoint /workspace/PIGG_checkpoints/global_agent_full \
    --u 1 10
```

**Output**: `eval_results/{checkpoint_name}/user_{id}/`

---

### 4. Personalized Fine-tuning (K-fold)

Fine-tune on individual user's data with K-fold cross-validation.
Always starts from the global full checkpoint.

```bash
cd src

# LoRA personalized training for all test users (1-10)
python 4_finetune_person.py 0 --lora

# Full fine-tuning for specific users
python 4_finetune_person.py 0 --users 1 2 3

# Multiple GPUs with LoRA
python 4_finetune_person.py 0,1 --users 1 5 --lora
```

**K-fold Process**:
- Splits user data into K=3 folds
- Trains on 2 folds, evaluates on 1 fold
- Repeats for each fold and averages metrics

**Output**: 
- Checkpoints: `/workspace/PIGG_checkpoints/personalized_lora/user_{id}/fold_{k}/`
- Results: `eval_results/personalized/user_{id}/fold_{k}/`
- Summary: `eval_results/personalized/user_{id}_summary.json`

---

### 5. Prediction Visualization

Visualize model predictions on GUI screenshots.

```bash
cd src

# Single sample
python visualize_prediction.py 0 --user 1 --sample 0

# Multiple samples
python visualize_prediction.py 0 --user 1 --samples 0 1 2 3 4

# All samples for a user
python visualize_prediction.py 0 --user 1 --all
```

**Visualization shows**:
- Blue circle: Ground Truth (GT)
- Red circle: Prediction
- Green/Red transparent circle: 14% accuracy range (green if accurate)

**Output**: `visualizations/user{id}_sample{idx}_vanilla.png`

---

## 📊 Evaluation Metrics

- **Click Accuracy@140px**: Success rate within 140px (≈14% of 1000px reference)
- **Mean L2 Distance**: Average pixel distance between prediction and GT
- **Median L2 Distance**: Median pixel distance

Results are saved in JSON format:
```json
{
  "click_acc_140": 0.45,
  "mean_l2": 234.5,
  "median_l2": 189.2
}
```

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

- **Learning rates**: `LR = 1e-6`, `PERSON_LR = 2e-5`
- **K-fold settings**: `K_FOLDS = 3`
- **Checkpoint paths**: `CHECKPOINT_GLOBAL_FULL`, `CHECKPOINT_PERSON_LORA`
- **User splits**: `USER_TEST = [1-10]`, `USER_TRAIN = [11-83]`

---

## 🔧 Technical Details

### Training Approach
- **Global Training**: Train on 11-83 users for general GUI understanding
- **Personalized Training**: Fine-tune on individual user (1-10) for personalization
- **Base Model**: Always use global full checkpoint as starting point for personalization

### Label Masking
Follows official Qwen3-VL methodology:
- User inputs: `label = -100` (ignored in loss)
- Assistant responses: `label = input_ids` (trained)

### Memory Optimization
- Gradient checkpointing enabled
- bfloat16 precision
- Conservative batch sizes (1 per GPU)
- Gradient accumulation for stability

---

## 📖 References

- Official Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Dataset: https://huggingface.co/datasets/qinglongyang/fingertip-20k

