# Vanilla Evaluation: Zero-Shot GUI Agent Performance Assessment

## 1. Overview

Vanilla Evaluation is an experiment that measures the **zero-shot performance of the pre-trained Qwen3-VL-8B-Instruct model without any fine-tuning**, predicting click coordinates corresponding to user's natural language commands on GUI screens.

The objectives of this experiment are:
- **Establishing Baseline Performance**: Measuring the model's baseline performance before fine-tuning to create a reference point for comparing the effectiveness of subsequent Global/Personalized fine-tuning
- **Validating Zero-Shot Capability**: Evaluating the fundamental ability of Vision-Language Models for GUI understanding tasks

---

## 2. Experimental Data (Dataset)

### 2.1 Data Structure

Each sample is structured in a **Conversational Format** with the following schema:

```json
[
  {
    "role": "system",
    "content": [
      {"type": "image", "image": "path/to/screenshot.jpg"},
      {"type": "text", "text": "prompt and instructions"}
    ]
  },
  {
    "role": "assistant", 
    "content": [
      {"type": "text", "text": "ground truth response (referring + grounding)"}
    ]
  }
]
```

### 2.2 Input Prompt Format

The prompt delivered to the model follows this template:

```
Given a GUI image, what are the relative (0-1000) pixel point coordinates 
for the element corresponding to the following instruction: [user command]

Think step-by-step, provide referring for the element first 
and then the grounded point coordinates.
```

### Output (Example)
```referring
"left arrow"
```
```grounding
(100, 200)
```


### 2.3 Ground Truth Format

The ground truth consists of two parts:
1. **Referring**: A description of the UI element to be clicked (e.g., "blue plus button")
2. **Grounding**: The coordinates of that element [x, y] (relative coordinates in range 0-1000)

Example:
```referring
blue plus button
```
```grounding
[311, 541]
```


### 2.4 Evaluation Target Users

| Category | User ID | Number of Samples |
|----------|---------|-------------------|
| Test User 1 | 1 | 268 |
| Test User 2 | 2 | 1,310 |
| Test User 3 | 3 | 561 |
| Test User 4 | 4 | 1,058 |
| Test User 5 | 5 | 457 |
| Test User 6 | 6 | 305 |
| Test User 7 | 7 | 3,700 |
| Test User 8 | 8 | 255 |
| Test User 9 | 9 | 530 |
| Test User 10 | 10 | 1,619 |
| **Total** | - | **10,063** |

---

## 3. Experimental Workflow

### 3.1 Overall Process Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VANILLA EVALUATION PIPELINE                     │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │   User       │     │   Dataset        │     │    Qwen3-VL-8B       │
  │   Selection  │────▶│   Loading        │────▶│    Model Loading     │
  │   (1-10)     │     │   (pkl files)    │     │    (HuggingFace)     │
  └──────────────┘     └──────────────────┘     └──────────────────────┘
                                                          │
                                                          ▼
  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │   Result     │     │   Coordinate     │     │    Model Inference   │
  │   Saving     │◀────│   Comparison &   │◀────│    (Generation)      │
  │   (JSON)     │     │   Evaluation     │     │                      │
  └──────────────┘     └──────────────────┘     └──────────────────────┘
```

### 3.2 Step-by-Step Detailed Description

#### **Step 1: Experiment Environment Setup**

Users specify the GPU and the list of users to evaluate from the command line.

```bash
python 1_vanilla_eval.py 0 --u 1 10  # Use GPU 0, evaluate Users 1-10
```

- Set the `CUDA_VISIBLE_DEVICES` environment variable to run the experiment on a specific GPU
- User list parsing: When two numbers are provided, they are interpreted as a range (e.g., `1 10` → `[1,2,3,...,10]`)

#### **Step 2: Evaluation Dataset Loading**

Data for specified users is loaded through the `Qwen3VLEvalDataset` class.

**Loading Process:**
1. Read pickle files for each user (`user_{id}.pkl`)
2. Merge all samples into a single list
3. Ground Truth extraction: Parse ground truth coordinates and referring content from the assistant message of each sample

**Ground Truth Extraction Logic:**
- Parse ````referring` and ````grounding` blocks from assistant messages using regular expressions
- Coordinates support various formats: `(x, y)`, `[x, y]`, `x, y`, etc.

#### **Step 3: Model Loading**

Load the `Qwen/Qwen3-VL-8B-Instruct` model from Hugging Face.

**Model Configuration:**
- `dtype="auto"`: Automatically select appropriate data type (typically bfloat16)
- `trust_remote_code=True`: Allow execution of Qwen3-VL's custom code
- Set model to evaluation mode (`model.eval()`)
- Move to CUDA device (`model.to("cuda")`)

#### **Step 4: Input Preprocessing**

The following preprocessing is performed for each sample:

1. **Chat Template Application**: 
   - Convert system messages to user role (Qwen3-VL compatibility)
   - Prepare for response generation with `add_generation_prompt=True`

2. **Image Processing**:
   - Extract image path
   - Convert image to model input format through `AutoProcessor`
   - Resize to `max_pixels = 1280 * 28 * 28` resolution

3. **Tensor Conversion**:
   - Convert text and image to PyTorch tensors
   - Handle batch dimensions (squeeze)

#### **Step 5: Model Inference**

Inference is performed within the `torch.inference_mode()` context.

**Generation Settings:**
- `max_new_tokens=512`: Generate up to 512 tokens
- Model generates responses including referring and grounding

**Output Decoding:**
- Convert token sequence to text with `tokenizer.decode()`
- Remove special tokens with `skip_special_tokens=True`

#### **Step 6: Prediction Coordinate Parsing**

Extract predicted coordinates from the generated text.

**Parsing Logic (`parse_inference` function):**
1. Extract only the assistant response portion (`text.split("assistant")[-1]`)
2. Extract referring content: Parse ````referring` block
3. Extract grounding coordinates: Apply various patterns sequentially
   - 4-value coordinates: `(x1, y1, x2, y2)`
   - "to" format: `(x1, y1) to (x2, y2)`
   - Keyword format: `Top-left: (x, y)`
   - Simple format: `[x, y]` or `(x, y)`

#### **Step 7: Evaluation Metrics Calculation**

Measure performance by comparing predicted and ground truth coordinates.

**Evaluation Metrics:**

1. **Click Accuracy (δ=140)**
   - Proportion of samples with L2 distance ≤ δ (140 pixels)
   - $\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\|pred_i - gt_i\|_2 \leq \delta]$

2. **Mean L2 Error**
   - Average L2 distance of all samples
   - $\text{Mean L2} = \frac{1}{N}\sum_{i=1}^{N} \|pred_i - gt_i\|_2$

3. **Median L2 Error**
   - Median of L2 distances (robust to outliers)

#### **Step 8: Result Saving**

Results are saved for each user.

**Save Path:** `./eval_results/qwen3_vanilla/user_{id}/`

**Saved Files:**
- `out.json`: Detailed results for each sample
  - `pred_raw`: Model's raw output
  - `pred_coord`: Parsed predicted coordinates
  - `gt_coord`: Ground truth coordinates
  - `pred_content`: Predicted referring content
  - `gt_content`: Ground truth referring content
  
- `results.txt`: Summary metrics (accuracy, mean_l2, median_l2)

---

## 4. Key Design Decisions

### 4.1 Zero-Shot Configuration

- **No Fine-tuning**: Evaluate GUI understanding ability using only the model's pre-trained knowledge
- **Prompt Engineering**: Provide clear output format examples to induce consistent responses

### 4.2 Coordinate Representation

- **Relative Coordinates (0-1000)**: Adaptable to screenshots of various resolutions
- **Single Point**: Represent click location as a single (x, y) coordinate

### 4.3 Evaluation Criteria

- **δ=140 pixels**: Tolerance considering typical UI element sizes
  - Most buttons and icons are included within this range
  - 14% tolerance range based on 1000-scale relative coordinates

---

## 5. Example Experimental Results

After execution, results are output in the following format:

```
===== User 1 =====
{
    'accuracy': 0.4328,
    'mean_l2': 187.23,
    'median_l2': 142.56
}
```

---

## Appendix A: Detailed Function Descriptions

### A.1 Main Script (`1_vanilla_eval.py`)

#### `vanilla_eval(user_list)`
- **Purpose**: Perform vanilla model evaluation for specified user list
- **Input**: `user_list` - List of user IDs to evaluate
- **Behavior**:
  1. Iterate over each user
  2. Call `eval_user()` to perform evaluation
  3. Output and save results
- **Output**: Print results for each user to console and save to file

---

### A.2 Evaluation Script (`scripts/eval_qwen3.py`)

#### `eval_user(checkpoint_path, user_eval, save_root, is_lora=False, eval_dataset=None)`
- **Purpose**: Evaluate specified users with a specific checkpoint
- **Input**:
  - `checkpoint_path`: Model path (for vanilla: `"Qwen/Qwen3-VL-8B-Instruct"`)
  - `user_eval`: List of user IDs to evaluate
  - `save_root`: Result save path
  - `is_lora`: Whether to use LoRA adapter
  - `eval_dataset`: Pre-created evaluation dataset (optional)
- **Behavior**:
  1. Verify checkpoint path and find latest checkpoint
  2. Load model (distinguish vanilla/fine-tuned/LoRA)
  3. Batch-wise inference through DataLoader
  4. Save prediction results for each sample to JSON in real-time
  5. Calculate overall evaluation metrics
- **Output**: Evaluation result dictionary (`accuracy`, `mean_l2`, `median_l2`)

---

### A.3 Dataset Class (`dataset.py`)

#### `Qwen3VLEvalDataset`
- **Purpose**: Evaluation dataset - returns preprocessed inputs and ground truth
- **Initialization (`__init__`)**:
  - Load pickle files for each user
  - Initialize AutoProcessor (for image preprocessing)
  - Pre-extract ground truth
- **`_extract_ground_truth()`**:
  - Parse ground truth from assistant messages of each sample
  - Extract coordinates using `parse_inference()` function
- **`__getitem__(idx)`**:
  - Convert input messages using chat template
  - Preprocess image and text with processor
  - Return: `(processed_inputs, gt_content, gt_coord)`

---

### A.4 Utility Functions (`utils/eval.py`)

#### `parse_inference(text, is_gt=True)`
- **Purpose**: Extract coordinates from model output or ground truth text
- **Input**:
  - `text`: Text to parse
  - `is_gt`: Whether it's ground truth text (if False, extract only assistant portion)
- **Behavior**:
  1. Extract referring content (```referring block)
  2. Extract grounding coordinates (apply various regex patterns sequentially)
- **Output**: `(content, x, y)` tuple

#### `evaluate_coordinates(pred_coords, gt_coords, delta=140)`
- **Purpose**: Calculate metrics by comparing predicted and ground truth coordinates
- **Input**:
  - `pred_coords`: Predicted coordinate list `[(x1,y1), (x2,y2), ...]`
  - `gt_coords`: Ground truth coordinate list
  - `delta`: Click Accuracy threshold (default 140)
- **Behavior**:
  1. Convert to NumPy arrays
  2. Calculate L2 distance: $\|pred - gt\|_2$
  3. Calculate each metric
- **Output**: `{"accuracy": float, "mean_l2": float, "median_l2": float}`

#### `save_results(result, save_path)`
- **Purpose**: Save evaluation results to file
- **Input**: `result` - Result dictionary, `save_path` - Save path
- **Behavior**: Record results in `results.txt` file

---

### A.5 Configuration Constants (`config.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_NAME` | `"Qwen/Qwen3-VL-8B-Instruct"` | Base model |
| `MAX_PIXELS` | `1280 * 28 * 28` | Maximum image resolution |
| `DATASET_ROOT` | `"./dataset_pkl"` | Dataset path |
| `NUM_WORKERS` | `4` | DataLoader worker count |
| `USER_TEST` | `[1, 2, ..., 10]` | Test user IDs |

---

## Appendix B: Data Flow Diagram

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        DATA FLOW DIAGRAM                                          │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘

user_1.pkl ─┐
user_2.pkl ─┼──▶ Qwen3VLEvalDataset ──▶ DataLoader ───┐
   ...      │         │                               │
user_10.pkl─┘         │                               │
                      ▼                               │
              ┌───────────────┐                       │
              │ Ground Truth  │                       │
              │ Extraction    │                       │
              │ - gt_coords   │                       │
              │ - gt_contents │                       │
              └───────────────┘                       │
                                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           INFERENCE LOOP                                           │ 
│                                                                                                    │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Batch     │───▶│  Processor   │───▶│   Model     │───▶│  Tokenizer  │───▶│ parse_inference │  │
│  │   (image +  │    │  (encode)    │    │  .generate()│    │  .decode()  │    │ - pred_coord    │  │
│  │   prompt)   │    │              │    │             │    │             │    │ - pred_content  │  │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │      
                                                         ▼
                                            ┌─────────────────────────┐
                                            │  evaluate_coordinates   │
                                            │  - Click Accuracy@140   │
                                            │  - Mean L2 Error        │
                                            │  - Median L2 Error      │
                                            └─────────────────────────┘
                                                         │
                                                         ▼
                                            ┌─────────────────────────┐
                                            │      RESULTS            │
                                            │  - out.json (per sample)│
                                            │  - results.txt (summary)│
                                            └─────────────────────────┘
```

---

## Appendix C: Example Execution Commands

```bash
# Default execution (GPU 0, User 1-10)
python 1_vanilla_eval.py 0

# Specify specific GPU
python 1_vanilla_eval.py 1

# Specify specific user range
python 1_vanilla_eval.py 0 --u 1 5    # Evaluate only Users 1-5

# Specify individual users
python 1_vanilla_eval.py 0 --u 1 3 7  # Evaluate only Users 1, 3, 7
```

---

*Document Created: December 3, 2025*
*Project: PIGG (Personalized Interactive GUI Grounding)*
