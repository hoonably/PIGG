# Personalized Fine-tuning: Personalized GUI Agent Training and Evaluation

## 1. Overview

Personalized Fine-tuning is an experiment that **trains a GUI Agent optimized for individual users by leveraging each user's data, building upon the model trained through Global Fine-tuning**.

Key characteristics of this experiment:
- **K-Fold Cross Validation**: Split each user's data into K folds to ensure reliable training and evaluation
- **Transfer Learning**: Perform personalization based on knowledge from the Global model
- **LoRA or Full Fine-tuning**: Support for two training methods
- **Immediate Evaluation After Training**: Evaluate on the hold-out test set immediately after each fold's training

### 1.1 Experimental Objectives

1. **Verify Personalization Effectiveness**: Measure the degree of performance improvement of personalized models compared to the Global model
2. **Learn User-Specific Characteristics**: Learn individual user's app usage patterns, preferred UI elements, etc.
3. **Prevent Overfitting**: Achieve reliable performance evaluation even on small datasets through K-Fold

---

## 2. Experimental Structure

### 2.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERSONALIZED FINE-TUNING ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────────┐
                        │   Global Full Model     │
                        │   (Pre-trained on       │
                        │    73 users' data)      │
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  User 1   │   │  User 2   │   │  User 10  │
            │Personalized│   │Personalized│   │Personalized│
            │   Model   │   │   Model   │...│   Model   │
            └───────────┘   └───────────┘   └───────────┘
                 │               │               │
         ┌───────┴───────┐      ...             ...
         ▼       ▼       ▼
     ┌──────┐┌──────┐┌──────┐
     │Fold 0││Fold 1││Fold 2│
     └──────┘└──────┘└──────┘
```

### 2.2 K-Fold Cross Validation Structure

Data is split into K=3 folds for each user:

```
User Data (e.g., 268 samples)
     │
     ▼
┌────────────────────────────────────────────────────────┐
│                    K-Fold Split (K=3)                   │
├──────────────────┬─────────────────┬───────────────────┤
│     Fold 0       │     Fold 1      │     Fold 2        │
│   (90 samples)   │  (89 samples)   │  (89 samples)     │
└──────────────────┴─────────────────┴───────────────────┘

Iteration 1: Train on Fold 1+2 (178 samples), Test on Fold 0 (90 samples)
Iteration 2: Train on Fold 0+2 (179 samples), Test on Fold 1 (89 samples)
Iteration 3: Train on Fold 0+1 (179 samples), Test on Fold 2 (89 samples)
```

### 2.3 Data Split

| Category | User ID | Role |
|----------|---------|------|
| **Training (Global)** | 11-83 | Used for Global model training |
| **Test (Personalized)** | 1-10 | Target for personalized training and evaluation |

---

## 3. Experimental Data (Dataset)

### 3.1 Data Status by Test User

| User ID | Total Samples | Fold 0 | Fold 1 | Fold 2 | Train (per fold) |
|---------|---------------|--------|--------|--------|------------------|
| 1 | 268 | 90 | 89 | 89 | ~178 |
| 2 | 1,310 | 437 | 437 | 436 | ~873 |
| 3 | 561 | 187 | 187 | 187 | ~374 |
| 4 | 1,058 | 353 | 353 | 352 | ~705 |
| 5 | 457 | 153 | 152 | 152 | ~304 |
| 6 | 305 | 102 | 102 | 101 | ~203 |
| 7 | 3,700 | 1,234 | 1,233 | 1,233 | ~2,466 |
| 8 | 255 | 85 | 85 | 85 | ~170 |
| 9 | 530 | 177 | 177 | 176 | ~353 |
| 10 | 1,619 | 540 | 540 | 539 | ~1,079 |
| **Total** | **10,063** | - | - | - | - |

### 3.2 Fold Splitting Logic

```python
def create_k_folds(samples, k=3, seed=42):
    random.seed(seed)  # Fixed seed for reproducibility
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    fold_size = len(shuffled_samples) // k
    folds = []
    
    for i in range(k):
        if i == k - 1:
            fold = shuffled_samples[i * fold_size:]  # Last fold includes remainder
        else:
            fold = shuffled_samples[i * fold_size:(i + 1) * fold_size]
        folds.append(fold)
    
    return folds
```

**Key Design**:
- **Fixed Seed (seed=42)**: Ensures identical fold splits between training and evaluation
- **Equal Split**: Split into equal sizes as much as possible
- **Last Fold**: Includes all remaining samples to prevent data loss

---

## 4. Experimental Workflow

### 4.1 Overall Process Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERSONALIZED FINE-TUNING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

For each User (1-10):
  For each Fold (0, 1, 2):
    
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
    │  Load User   │     │   K-Fold         │     │    Load Global      │
    │  Data        │────▶│   Data Split     │────▶│    Full Model       │
    │              │     │                  │     │                      │
    └──────────────┘     └──────────────────┘     └──────────────────────┘
                                                            │
                                                            ▼
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
    │  Fold        │     │   Personalized   │     │    Add LoRA         │
    │  Evaluation  │◀────│   Training       │◀────│    Adapters         │
    │  (Hold-out)  │     │   (K-1 folds)    │     │    (Optional)       │
    └──────────────┘     └──────────────────┘     └──────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         Save and Aggregate Results                    │
    │  - Per-fold results: ./eval_results/personalized/user_{id}/fold_{idx}│
    │  - User average: user_{id}_summary.json                               │
    └──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Step-by-Step Detailed Description

#### **Step 1: Environment Setup and Argument Parsing**

```bash
python 4_finetune_person.py 0 --users 1 2 3 --lora  # GPU 0, Users 1,2,3, LoRA method
```

- **GPU Specification**: Set `CUDA_VISIBLE_DEVICES` environment variable
- **User Selection**: Specific users or all test users (1-10)
- **Fold Selection**: Train specific fold only or all K folds
- **Training Method**: LoRA or Full Fine-tuning

#### **Step 2: Load User Data and K-Fold Split**

1. **Data Loading**: Load all samples for the user from `user_{id}.pkl` file
2. **K-Fold Creation**: Generate consistent fold splits with fixed seed
3. **Train/Test Separation**: 
   - Training Data: (K-1) folds excluding current fold
   - Test Data: Current hold-out fold

#### **Step 3: Model Loading**

**Base Model**: Always use Global Full Fine-tuned model as base

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    CHECKPOINT_GLOBAL_FULL,  # Global Full checkpoint
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

#### **Step 4: Add LoRA Adapters (Optional)**

When using LoRA:
```python
lora_config = LoraConfig(
    r=8,                          # Rank (small value to prevent overfitting)
    lora_alpha=32,                # Alpha value
    target_modules="all-linear",  # Apply to all Linear layers
    lora_dropout=0.1,             # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

**Design Rationale**:
- **Low Rank (r=8)**: Prevent overfitting on small personalized datasets
- **Dropout (0.1)**: Additional regularization effect
- **all-linear**: Ensure sufficient expressiveness

#### **Step 5: Freeze Vision Encoder and Aligner**

Following Qwen3-VL official recommendations:

```python
# Freeze Vision Encoder
for param in vision_tower.parameters():
    param.requires_grad = False

# Freeze Multi-modal Projector (Aligner)
for param in multi_modal_projector.parameters():
    param.requires_grad = False
```

**Reason**: 
- Vision understanding capability is already sufficiently acquired through pre-training
- Fine-tune only the Language Model portion to prevent overfitting

#### **Step 6: Personalized Training**

**Training Hyperparameters**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PERSON_EPOCHS` | 3 | Number of training epochs |
| `PERSON_BATCH_SIZE` | 1 | Batch size |
| `PERSON_GRADIENT_ACCUMULATION_STEPS` | 4 | Gradient accumulation (effective batch=4) |
| `PERSON_LR` | 2e-5 | Learning rate (higher than Global) |
| `PERSON_WARMUP_STEPS` | 5 | Warmup steps |
| `PERSON_WEIGHT_DECAY` | 0.01 | Weight decay |

**Global vs Personalized Hyperparameter Comparison**:

| Parameter | Global | Personalized | Reason |
|-----------|--------|--------------|--------|
| Learning Rate | 1e-6 | 2e-5 | Need faster adaptation on small datasets |
| Epochs | 1 | 3 | More iterations needed due to smaller data size |
| Effective Batch | 8 | 4 | More frequent updates due to lower data diversity |
| Warmup Steps | 50 | 5 | Reduced to match small dataset |

#### **Step 7: Save Model**

Save checkpoint after training completion:

```
/workspace/PIGG_checkpoints/personalized_lora/user_{id}/fold_{idx}/
  ├── adapter_config.json    # LoRA configuration
  ├── adapter_model.safetensors  # LoRA weights
  └── ...
```

#### **Step 8: Fold Evaluation**

Evaluate on the hold-out fold's test set immediately after training:

1. **Create Test Dataset**: Use hold-out fold samples
2. **Perform Inference**: Predict coordinates for each sample
3. **Calculate Metrics**: Accuracy, Mean L2, Median L2

#### **Step 9: Memory Cleanup**

Aggressive memory management to prevent OOM:

```python
del trainer, model
torch.cuda.empty_cache()
gc.collect()
```

Multiple cleanup rounds between folds to ensure sufficient GPU memory

#### **Step 10: Result Aggregation**

**Per-Fold Result Saving**:
```
./eval_results/personalized/user_{id}/fold_{idx}/
  ├── out.json      # Detailed per-sample results
  └── results.txt   # Summary metrics
```

**User K-Fold Average Results**:
```json
{
    "user_id": 1,
    "n_folds": 3,
    "avg_accuracy": 0.7234,
    "avg_mean_l2": 89.45,
    "avg_median_l2": 67.23,
    "std_accuracy": 0.0156,
    "std_mean_l2": 5.67,
    "std_median_l2": 4.32
}
```

---

## 5. Key Design Decisions

### 5.1 Global Full → Personalized Transfer Learning

**Why use Global Full instead of Global LoRA as base?**

1. **Stronger Base Performance**: Full fine-tuning updates all parameters, acquiring more knowledge
2. **Reserve Room for Personalization**: Add new LoRA adapters for personalized training
3. **Simplified Model Structure**: Base model + single LoRA adapter structure for easier management

### 5.2 Reasons for Using K-Fold Cross Validation

1. **Small Datasets**: Individual user data ranges from 255 to 3,700 samples, relatively small
2. **Reliable Evaluation**: Prevent bias from single train/test split
3. **Full Data Utilization**: Every sample is used for testing exactly once

### 5.3 Reasons for Choosing K=3

- **Advantages over K=5**: Higher training data ratio per fold (67% vs 80%)
- **Data Efficiency**: Secure sufficient training data on small datasets
- **Computational Cost**: Complete within reasonable time with 3 training runs

### 5.4 LoRA vs Full Fine-tuning

| Aspect | LoRA | Full Fine-tuning |
|--------|------|------------------|
| **Trainable Parameters** | ~0.1% | ~100% |
| **Memory Usage** | Low | High |
| **Overfitting Risk** | Low | High |
| **Expressiveness** | Limited | High |
| **Recommended For** | Small datasets | Large datasets |

---

## 6. Evaluation Metrics

### 6.1 Individual Fold Metrics

Metrics calculated for each fold:

1. **Click Accuracy (δ=140)**
   - Proportion of samples with L2 distance ≤ 140 pixels
   
2. **Mean L2 Error**
   - Average L2 distance across all samples
   
3. **Median L2 Error**
   - Median of L2 distances

### 6.2 K-Fold Aggregated Metrics

Aggregating results from all folds:

- **Mean**: K-fold average of each metric
- **Standard Deviation (Std)**: Measure of performance variance

### 6.3 Overall User Aggregation

Calculate overall average when evaluating multiple users:

```python
overall_acc = np.mean([r['avg_accuracy'] for r in all_user_results])
overall_mean_l2 = np.mean([r['avg_mean_l2'] for r in all_user_results])
```

---

## 7. Example Experimental Results

### 7.1 Single Fold Result

```
Fold 0 Results:
  Accuracy: 0.7556
  Mean L2: 78.34
  Median L2: 52.18
```

### 7.2 Per-User K-Fold Summary

```
USER 1 K-FOLD SUMMARY
================================================================================
Evaluated 3/3 folds:
  Accuracy:   0.7234 ± 0.0156
  Mean L2:    89.45 ± 5.67
  Median L2:  67.23 ± 4.32
```

### 7.3 Overall User Summary

```
OVERALL SUMMARY ACROSS ALL USERS
================================================================================
Evaluated 10 users:
  Overall Accuracy:   0.6892
  Overall Mean L2:    102.34
  Overall Median L2:  78.56

Per-user breakdown:
  User 1: Acc=0.7234, Mean L2=89.45, Median L2=67.23
  User 2: Acc=0.6543, Mean L2=115.23, Median L2=89.12
  ...
```

---

## Appendix A: Detailed Function Descriptions

### A.1 Main Function

#### `finetune_personalized(user_id, fold_idx, use_lora, log_file)`
- **Purpose**: Perform personalized training and evaluation for a specific user's specific fold
- **Input**:
  - `user_id`: Target user ID (1-10)
  - `fold_idx`: Fold index to hold out (0 to K-1)
  - `use_lora`: Whether to use LoRA
  - `log_file`: Log file path
- **Behavior**:
  1. Load user data and create K-fold split
  2. Load Global Full model
  3. (Optional) Add LoRA adapters
  4. Freeze vision encoder/aligner
  5. Train with Trainer
  6. Save model
  7. Evaluate on hold-out fold
- **Output**: `(checkpoint_path, fold_result)` tuple

---

### A.2 Dataset Classes

#### `PersonalizedDataset`
- **Purpose**: PyTorch Dataset for personalized training
- **Initialization**: Receive and store sample list
- **`__getitem__`**: Return messages in Qwen3-VL format (system → user role conversion)

#### `FoldEvalDataset`
- **Purpose**: Evaluation dataset for a specific fold's test set
- **Initialization**: Load samples, extract ground truth, initialize processor
- **`__getitem__`**: Return preprocessed inputs + ground truth

---

### A.3 K-Fold Utilities (`utils/kfold.py`)

#### `create_k_folds(samples, k, seed)`
- **Purpose**: Split samples into K folds
- **Key Point**: Ensure reproducible splits with fixed seed

#### `get_training_folds(folds, test_fold_idx)`
- **Purpose**: Return training samples excluding the test fold

#### `get_test_fold(folds, test_fold_idx)`
- **Purpose**: Return test samples for a specific fold

---

### A.4 Evaluation Function

#### `eval_fold(user_id, fold_idx, checkpoint_path, folds, is_lora)`
- **Purpose**: Evaluate the trained model on the fold's test set
- **Behavior**:
  1. Create FoldEvalDataset with test fold samples
  2. Call `eval_user()` for evaluation
  3. Save results
  4. Memory cleanup
- **Output**: Evaluation result dictionary

---

### A.5 Configuration Constants (Personalized)

| Constant | Value | Description |
|----------|-------|-------------|
| `K_FOLDS` | 3 | K value for K-Fold |
| `PERSON_EPOCHS` | 3 | Personalized training epochs |
| `PERSON_BATCH_SIZE` | 1 | Batch size |
| `PERSON_GRADIENT_ACCUMULATION_STEPS` | 4 | Gradient accumulation |
| `PERSON_LR` | 2e-5 | Learning rate |
| `PERSON_WARMUP_STEPS` | 5 | Warmup steps |
| `PERSON_WEIGHT_DECAY` | 0.01 | Weight decay |
| `PERSON_LORA_RANK` | 8 | LoRA rank |
| `PERSON_LORA_ALPHA` | 32 | LoRA alpha |

---

## Appendix B: Example Execution Commands

```bash
# All test users (1-10), all folds, LoRA method
python 4_finetune_person.py 0 --lora

# Train specific users only
python 4_finetune_person.py 0 --users 1 2 3 --lora

# Train specific fold only
python 4_finetune_person.py 0 --users 1 --fold 0 --lora

# Full Fine-tuning method
python 4_finetune_person.py 0,1 --users 1

# Multi-GPU usage
python 4_finetune_person.py 0,1,2 --lora
```

---

## Appendix C: Directory Structure

```
/workspace/PIGG_checkpoints/
├── global_agent_full/           # Global Full checkpoint (Base)
├── personalized_lora/           # Personalized LoRA checkpoints
│   ├── user_1/
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   └── fold_2/
│   ├── user_2/
│   │   └── ...
│   └── ...
└── personalized_full/           # Personalized Full checkpoints
    └── ...

./eval_results/personalized/
├── user_1/
│   ├── fold_0/
│   │   ├── out.json
│   │   └── results.txt
│   ├── fold_1/
│   │   └── ...
│   └── fold_2/
│       └── ...
├── user_1_summary.json          # User 1's K-fold average results
├── user_2/
│   └── ...
└── ...
```

---

*Document Created: December 3, 2025*
*Project: PIGG (Personalized Interactive GUI Grounding)*
