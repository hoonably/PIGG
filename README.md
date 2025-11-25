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
```
cd dataset
python download.py
python fixdir.py
```

## Model
This project uses **Qwen3-VL-8B-Instruct** model from Hugging Face.

