# Global Fine-tuning: Multi-User GUI Agent Training

## 1. Overview

Global Fine-tuning is an experiment that **trains a single general-purpose GUI Agent model by integrating data from multiple users**. Based on the Qwen3-VL-8B-Instruct model, Supervised Fine-tuning (SFT) is performed using data from 73 users (User 11-83).

The objectives of this experiment are:
- **Building a General-Purpose GUI Agent**: Developing a model applicable to new users by learning diverse user patterns
- **Creating a Base Model for Personalized Fine-tuning**: Serving as the starting point (Base Checkpoint) for personalization training
- **Validating Performance Improvement over Zero-Shot**: Measuring the effectiveness of Fine-tuning compared to the Vanilla model

---

## 2. Training Methodology

### 2.1 Two Fine-tuning Approaches

This experiment supports two fine-tuning approaches:

| Approach | Description | Trainable Parameters | Memory Usage |
|----------|-------------|---------------------|--------------|
| **LoRA** | Low-Rank Adaptation | ~0.5% | Low |
| **Full Fine-tuning** | Training entire LLM parameters | ~99% (excluding Vision) | High |

### 2.2 Common Design Principles

Following Qwen3-VL official guidelines, the following principles are applied:

1. **Vision Encoder Freezing**: Fixing parameters of the pre-trained visual encoder
2. **Multi-modal Projector Freezing**: Fixing the Vision-Language alignment layer
3. **Assistant-only Label Masking**: Setting only assistant response tokens as training targets
4. **Gradient Checkpointing**: Applying memory-efficient training techniques

---

## 3. Data Configuration

### 3.1 User Split

```
┌───────────────────────────────────────────────────────────────────┐
│                        USER DATA SPLIT                            │
├───────────────────────────────────────────────────────────────────┤
│  Training Users    │  User 11 - 83  │  73 users │  90,727 samples │
│  Validation Users  │  (Disabled)    │  0 users  │  0 samples      │
│  Test Users        │  User 1 - 10   │  10 users │  10,063 samples │
└───────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Format

Each sample is structured in conversational format:

```json
[
  {
    "role": "user",
    "content": [
      {"type": "image", "image": "screenshot path"},
      {"type": "text", "text": "GUI command prompt"}
    ]
  },
  {
    "role": "assistant",
    "content": [
      {"type": "text", "text": "referring + grounding response"}
    ]
  }
]
```

---

## 4. Training Workflow

### 4.1 Overall Process Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GLOBAL FINE-TUNING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │   Environment│     │   Load Qwen3-VL  │     │   Apply LoRA/Full    │
  │   Setup      │────▶│   Model          │────▶│   FT Configuration   │
  │   (GPU, Log) │     │                  │     │                      │
  └──────────────┘     └──────────────────┘     └──────────────────────┘
                                                          │
                                                          ▼
  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │   Save       │     │   Execute        │     │   Load Dataset       │
  │   Checkpoint │◀────│   Trainer        │◀────│   (73 users)         │
  └──────────────┘     └──────────────────┘     └──────────────────────┘
```

### 4.2 Step-by-Step Detailed Description

#### **Step 1: Environment Setup and Logging Initialization**

Users specify the GPU and training method from the command line.

```bash
python 2_finetune_global.py 0 --lora    # GPU 0, LoRA training
python 2_finetune_global.py 0,1         # GPU 0,1, Full Fine-tuning
```

**Logging System (TeeLogger):**
- Simultaneous writing to stdout and log file
- Automatic creation of log file with timestamp
- Path: `./logs/finetune_global_{method}_{timestamp}.log`

#### **Step 2: Model and Processor Loading**

Load the Qwen3-VL-8B-Instruct model from Hugging Face.

**Processor Settings:**
- `Qwen3VLProcessor`: Handles image and text preprocessing
- `max_pixels = 1280 * 28 * 28`: Image resolution setting

**Model Settings:**
- `torch_dtype=torch.bfloat16`: Memory-efficient 16-bit floating point
- `device_map="auto"`: Automatic multi-GPU distribution

#### **Step 3: Apply Fine-tuning Method**

**For LoRA Training:**
```python
LoraConfig(
    r=8,                          # Rank: low-rank decomposition dimension
    lora_alpha=32,                # Scaling factor
    target_modules="all-linear",  # Apply to all Linear layers
    lora_dropout=0.1,             # Dropout rate
    bias="none",                  # No bias training
    task_type="CAUSAL_LM"         # Causal language modeling
)
```

**For Full Fine-tuning:**
- Train entire LLM parameters without LoRA
- Vision Encoder and Projector are frozen the same way

#### **Step 4: Vision Component Freezing**

Following official best practices, the following components are frozen:

1. **Vision Encoder (vision_tower/visual)**
   - Preserving pre-trained image feature extractor
   - Maintaining representational power learned from large-scale image data

2. **Multi-modal Projector**
   - Vision-Language alignment layer
   - Acts as the connection between Vision and LLM

**Effects of Freezing:**
- Reduced number of trainable parameters
- Prevention of overfitting
- Improved training stability

#### **Step 5: Dataset and Data Collator Preparation**

**Qwen3VLTrainDataset:**
- Load pickle files for specified users
- Integrate all samples into a single list
- Convert 'system' role to 'user' (Qwen3-VL compatibility)

**Qwen3VLDataCollator:**
- `apply_chat_template(tokenize=True)`: Official tokenization method
- **Assistant-only Label Masking**: Core training technique

#### **Step 6: Label Masking Detailed Explanation**

Label Masking determines which tokens to use as training targets.

**Qwen3-VL Chat Template Structure:**
```
<|im_start|>user
[image + prompt]<|im_end|>
<|im_start|>assistant
[response - ONLY THIS PART IS TRAINED!]<|im_end|>
```

**Masking Logic:**
1. Search for `<|im_start|>assistant` pattern
2. Set tokens up to `<|im_end|>` as training targets
3. Mask remaining tokens with `IGNORE_INDEX(-100)`

**Mathematical Expression:**
$$
\text{labels}_i = 
\begin{cases}
\text{input\_ids}_i & \text{if } i \in \text{assistant\_response} \\
-100 & \text{otherwise}
\end{cases}
$$

#### **Step 7: Training Arguments Configuration**

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `learning_rate` | 1e-6 | Official recommended range (1e-6 ~ 2e-7) |
| `num_train_epochs` | 1 | Number of epochs |
| `per_device_train_batch_size` | 1 | Batch size per GPU |
| `gradient_accumulation_steps` | 8 | Effective batch size = 8 |
| `warmup_steps` | 50 | Warmup steps |
| `weight_decay` | 0.01 | Weight decay |
| `bf16` | True | BFloat16 mixed precision |
| `gradient_checkpointing` | True | Memory optimization |
| `save_steps` | 500 | Checkpoint save interval |

#### **Step 8: Trainer Execution**

Training is performed using Hugging Face `Trainer`.

**Key Features:**
- Automatic checkpoint saving (every 500 steps)
- Keep only 3 most recent checkpoints (disk savings)
- Use Safetensors format (stability)
- Validation disabled (OOM prevention)

#### **Step 9: Model Saving**

Save the final model after training completion.

**Save Paths:**
- LoRA: `/workspace/PIGG_checkpoints/global_agent_lora`
- Full: `/workspace/PIGG_checkpoints/global_agent_full`

**Saved Items:**
- Model weights (only adapters for LoRA)
- Processor (including tokenizer)
- Training configuration

---

## 5. LoRA vs Full Fine-tuning Comparison

### 5.1 Trainable Parameters Comparison

**LoRA Fine-tuning:**
```
trainable params: 41,943,040 / 8,294,967,296 (0.51%)
```

**Full Fine-tuning:**
```
trainable params: 8,222,674,944 / 8,294,967,296 (99.13%)
(Excluding Vision Encoder + Projector)
```

### 5.2 Pros and Cons Comparison

| Feature | LoRA | Full Fine-tuning |
|---------|------|------------------|
| **Memory Usage** | Low (~16GB) | High (~40GB+) |
| **Training Speed** | Fast | Slow |
| **Performance** | Good | Best |
| **Storage Size** | ~100MB | ~16GB |
| **Flexibility** | Adapter swappable | Full model required |

---

## 6. Memory Optimization Techniques

### 6.1 Gradient Checkpointing

Save memory by not storing intermediate activations and recomputing during backpropagation.

**Application:**
```python
model.gradient_checkpointing_enable()
```

### 6.2 Mixed Precision Training

Mixed precision training using BFloat16.

**Advantages:**
- 50% reduction in memory usage
- Improved training speed
- Numerical stability maintained (bf16's wide dynamic range)

### 6.3 Device Map Auto

Automatic model distribution in multi-GPU environments.

```python
model = Model.from_pretrained(..., device_map="auto")
```

---

## 7. Execution Examples

### 7.1 LoRA Fine-tuning

```bash
# LoRA training on GPU 0
python 2_finetune_global.py 0 --lora

# Output example
================================================================================
Qwen3-VL Global GUI Agent Fine-tuning
================================================================================
Training users: 73 users
  User IDs: 11-83
Method: LoRA
...
Training samples: 90727
...
Starting training...
```

### 7.2 Full Fine-tuning

```bash
# Full Fine-tuning on GPU 0,1
python 2_finetune_global.py 0,1

# Automatic multi-GPU distribution
```

---

## Appendix A: Detailed Function Descriptions

### A.1 Main Script (`2_finetune_global.py`)

#### `finetune_global_agent(use_lora=True, log_file=None)`
- **Purpose**: Execute Global GUI Agent Fine-tuning
- **Input**:
  - `use_lora`: Whether to use LoRA
  - `log_file`: Log file path
- **Behavior**:
  1. Call `train_lora()` or `train_full()` based on training method
  2. Return checkpoint path
- **Output**: Saved checkpoint path

#### `TeeLogger`
- **Purpose**: Simultaneous logging to stdout and file
- **Methods**:
  - `write(message)`: Simultaneous output to terminal and file
  - `flush()`: Flush buffer
  - `close()`: Close file handle

---

### A.2 Training Script (`scripts/train_qwen3.py`)

#### `train_lora(checkpoint_dir, user_train, user_val=None)`
- **Purpose**: Train Qwen3-VL with LoRA
- **Input**:
  - `checkpoint_dir`: Checkpoint save path
  - `user_train`: Training user ID list
  - `user_val`: Validation user ID list (optional)
- **Behavior**:
  1. Load Processor and model
  2. Apply LoRA configuration (`get_peft_model`)
  3. Freeze Vision components
  4. Create dataset and Collator
  5. Execute training with Trainer
  6. Save final model

#### `train_full(checkpoint_dir, user_train, user_val=None)`
- **Purpose**: Full parameter Fine-tuning
- **Behavior**: Similar to `train_lora()` except for LoRA configuration

---

### A.3 Utilities (`utils/train_utils.py`)

#### `Qwen3VLDataCollator`
- **Purpose**: Batch data preprocessing and label creation
- **Key Methods**:
  - `__call__(instances)`: Execute batch collation
  - `_create_labels_with_assistant_masking(input_ids)`: Mask only assistant tokens as training targets

#### `get_training_arguments(output_dir, **kwargs)`
- **Purpose**: Create TrainingArguments
- **Defaults**: Use hyperparameters from config.py
- **kwargs**: Override defaults with additional arguments

#### `prepare_model_for_training(model)`
- **Purpose**: Prepare model for training
- **Behavior**:
  - Enable gradient checkpointing
  - Enable input gradients (PEFT compatibility)

---

### A.4 Dataset Class (`dataset.py`)

#### `Qwen3VLTrainDataset`
- **Purpose**: Training dataset
- **Initialization**: Load pickle files for specified users
- **`__getitem__(idx)`**: 
  - Return raw messages (preprocessing in Collator)
  - Convert 'system' → 'user' role

---

### A.5 Configuration Constants (`config.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `LR` | `1e-6` | Learning rate |
| `EPOCHS` | `1` | Number of epochs |
| `BATCH_SIZE` | `1` | Batch size |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Gradient accumulation |
| `SAVE_STEPS` | `500` | Save interval |
| `WARMUP_STEPS` | `50` | Warmup |
| `WEIGHT_DECAY` | `0.01` | Weight decay |
| `USER_TRAIN` | `[11, ..., 83]` | Training users (73) |
| `IGNORE_INDEX` | `-100` | Masking index |

---

## Appendix B: Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING DATA FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

user_11.pkl ─┐
user_12.pkl ─┤
    ...      ├──▶ Qwen3VLTrainDataset ──▶ DataLoader ───┐
    ...      │         │                                │
user_83.pkl ─┘         │                                │
                       ▼                                │
               ┌───────────────┐                        │
               │ Raw Messages  │                        │
               │ (image + text │                        │
               │  + response)  │                        │
               └───────────────┘                        │
                                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLATION PROCESS                             │
│                                                                            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ apply_chat_     │───▶│ Label Masking   │───▶│ Batch Tensors           │ │
│  │ template        │    │ (Assistant-only)│    │ - input_ids             │ │
│  │ (tokenize=True) │    │                 │    │ - labels                │ │
│  └─────────────────┘    └─────────────────┘    │ - attention_mask        │ │
│                                                │ - pixel_values          │ │
│                                                │ - image_grid_thw        │ │
│                                                └─────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING LOOP                                 │
│                                                                            │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Forward   │───▶│   Loss       │───▶│  Backward   │───▶│  Optimizer  │ │
│  │   Pass      │    │  Computation │    │   Pass      │    │   Step      │ │
│  │             │    │  (CE Loss)   │    │             │    │             │ │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘ │
│                                                                            │
│                     Loss = CrossEntropy(logits, labels)                    │
│                     (tokens where labels == -100 are ignored)              │
└────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   CHECKPOINT SAVING     │
                            │   - Every 500 steps     │
                            │   - Final model         │
                            │   - Processor           │
                            └─────────────────────────┘
```

---

## Appendix C: Execution Commands

```bash
# LoRA Fine-tuning (Recommended: Single GPU)
python 2_finetune_global.py 0 --lora

# Full Fine-tuning (Recommended: Multi-GPU)
python 2_finetune_global.py 0,1

# Specify specific GPU
python 2_finetune_global.py 2 --lora

# Monitor logs
tail -f ./logs/finetune_global_lora_*.log
```

---

## Appendix D: Troubleshooting

### D.1 CUDA Out of Memory

**Symptom:** GPU memory shortage error

**Solutions:**
1. Use LoRA (`--lora` flag)
2. Increase `gradient_accumulation_steps`
3. Use multi-GPU (`device_map="auto"`)
4. Keep `per_device_train_batch_size=1`

### D.2 Training Loss Not Decreasing

**Symptom:** Loss is fixed or increasing

**Checklist:**
1. Check label masking log: "Trainable tokens found" > 0
2. Verify learning rate is not too high (recommended: 1e-6)
3. Validate data format

### D.3 Checkpoint Save Failure

**Symptom:** Insufficient disk space

**Solutions:**
1. Limit checkpoints with `save_total_limit=3`
2. Secure sufficient disk space (Full: ~50GB, LoRA: ~1GB)

---

*Document Created: December 3, 2025*
*Project: PIGG (Personalized Interactive GUI Grounding)*
