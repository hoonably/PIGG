"""
Visualize global model predictions on GUI screenshots
Shows predicted point, ground truth, and 14% accuracy circle for global fine-tuned models

Usage:
    # Visualize Full global model
    python visualize_global.py 0 --user 1 --sample 0
    python visualize_global.py 0 --user 1 --all
    
    # Custom checkpoint path
    python visualize_global.py 0 --user 1 --all --checkpoint /path/to/checkpoint
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, Qwen3VLProcessor
from peft import PeftModel

from config import (
    DATASET_ROOT, 
    MAX_PIXELS, 
    CHECKPOINT_GLOBAL_LORA, 
    CHECKPOINT_GLOBAL_FULL,
    MODEL_NAME
)
from dataset import Qwen3VLEvalDataset
from utils.eval import parse_inference


def draw_prediction_visualization(
    image_path,
    pred_coord,
    gt_coord,
    pred_content,
    gt_content,
    output_path,
    delta=140
):
    """
    Draw prediction and ground truth on GUI screenshot with accuracy circle.
    
    Args:
        image_path: Path to GUI screenshot
        pred_coord: Predicted coordinate (x, y) - can be relative [0-1000] or absolute
        gt_coord: Ground truth coordinate (x, y) - can be relative [0-1000] or absolute
        pred_content: Predicted content description
        gt_content: Ground truth content description
        output_path: Path to save visualization
        delta: Accuracy threshold in pixels (140px = ~14% of 1000px reference)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Create drawing context
    draw = ImageDraw.Draw(img)
    
    # Convert relative coordinates to absolute if needed
    pred_x, pred_y = pred_coord
    gt_x, gt_y = gt_coord
    
    # Check if prediction is relative (0-1000 range) or absolute
    pred_is_relative = (pred_x <= 1000 and pred_y <= 1000)
    gt_is_relative = (gt_x <= 1000 and gt_y <= 1000)
    
    # Convert to absolute coordinates
    if pred_is_relative:
        pred_x_abs = int(pred_x * width / 1000)
        pred_y_abs = int(pred_y * height / 1000)
        pred_coord_type = "relative"
    else:
        pred_x_abs = int(pred_x)
        pred_y_abs = int(pred_y)
        pred_coord_type = "absolute"
    
    if gt_is_relative:
        gt_x_abs = int(gt_x * width / 1000)
        gt_y_abs = int(gt_y * height / 1000)
        gt_coord_type = "relative"
    else:
        gt_x_abs = int(gt_x)
        gt_y_abs = int(gt_y)
        gt_coord_type = "absolute"
    
    # Scale delta to image size (delta is in 1000px reference scale)
    delta_scaled = int(delta * min(width, height) / 1000)
    
    # Calculate L2 distance
    distance = np.sqrt((pred_x_abs - gt_x_abs)**2 + (pred_y_abs - gt_y_abs)**2)
    is_accurate = distance <= delta_scaled
    
    # Draw accuracy circle around GT (14% range)
    circle_color = (0, 255, 0, 80) if is_accurate else (255, 0, 0, 80)  # Green if accurate, red if not
    
    # Create semi-transparent overlay for circle
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw circle around GT
    overlay_draw.ellipse(
        [
            gt_x_abs - delta_scaled,
            gt_y_abs - delta_scaled,
            gt_x_abs + delta_scaled,
            gt_y_abs + delta_scaled
        ],
        fill=circle_color,
        outline=(circle_color[0], circle_color[1], circle_color[2], 200),
        width=3
    )
    
    # Composite overlay with original image
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw GT point (blue circle)
    marker_radius = 20
    draw.ellipse(
        [gt_x_abs - marker_radius, gt_y_abs - marker_radius,
         gt_x_abs + marker_radius, gt_y_abs + marker_radius],
        fill=(0, 0, 255),
        outline=(255, 255, 255),
        width=4
    )
    
    # Draw prediction point (red circle)
    draw.ellipse(
        [pred_x_abs - marker_radius, pred_y_abs - marker_radius,
         pred_x_abs + marker_radius, pred_y_abs + marker_radius],
        fill=(255, 0, 0),
        outline=(255, 255, 255),
        width=4
    )
    
    # Draw connecting line
    draw.line(
        [(pred_x_abs, pred_y_abs), (gt_x_abs, gt_y_abs)],
        fill=(255, 255, 0),
        width=2
    )
    
    # Save result (no text overlay on image)
    img.save(output_path)
    print(f"Visualization saved to: {output_path}")
    print(f"  Image size: {width}x{height}")
    print(f"  GT: ({gt_x}, {gt_y}) [{gt_coord_type}] -> abs ({gt_x_abs}, {gt_y_abs})")
    print(f"  Pred: ({pred_x}, {pred_y}) [{pred_coord_type}] -> abs ({pred_x_abs}, {pred_y_abs})")
    print(f"  Distance: {distance:.1f}px (threshold: {delta_scaled}px)")
    print(f"  Accurate: {'YES' if is_accurate else 'NO'}")
    
    return is_accurate


def visualize_sample(checkpoint_path, user_id, sample_idx, is_lora=False, output_dir="./visualizations_global"):
    """
    Visualize a specific sample from global model.
    
    Args:
        checkpoint_path: Path to global checkpoint
        user_id: User ID
        sample_idx: Index of sample to visualize
        is_lora: Whether the model is LoRA or full fine-tuned
        output_dir: Directory to save visualizations
    """
    print("=" * 80)
    print("Global Model Prediction Visualization")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"User: {user_id}")
    print(f"Sample: {sample_idx}")
    print(f"Model type: {'LoRA' if is_lora else 'Full'}")
    print()
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please run training first: python 2_finetune_global.py 0 {'--lora' if is_lora else ''}")
        return None
    
    # Load dataset
    print("Loading dataset...")
    eval_dataset = Qwen3VLEvalDataset([user_id], DATASET_ROOT)
    
    if sample_idx >= len(eval_dataset):
        print(f"Error: Sample index {sample_idx} out of range (dataset has {len(eval_dataset)} samples)")
        return None
    
    # Get sample
    inputs, gt_content, gt_coord = eval_dataset[sample_idx]
    
    # Extract image path from dataset
    sample_messages = eval_dataset.samples[sample_idx]
    image_path = None
    for msg in sample_messages:
        if msg['role'] == 'system':
            for content_item in msg['content']:
                if content_item['type'] == 'image':
                    image_path = content_item['image']
                    break
            break
    
    if not image_path:
        print("Error: Could not find image path in sample")
        return None
    
    print(f"Image: {image_path}")
    print(f"Ground truth: {gt_content} at {gt_coord}")
    print()
    
    # Load model
    print("Loading global model...")
    # Load processor from original model (processor unchanged during fine-tuning)
    processor = Qwen3VLProcessor.from_pretrained(
        MODEL_NAME,
        max_pixels=MAX_PIXELS
    )
    
    if is_lora:
        # Load vanilla base + LoRA adapter
        print(f"  Loading vanilla base model: {MODEL_NAME}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype="auto",
            trust_remote_code=True
        )
        print(f"  Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        # Load full fine-tuned model
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            dtype="auto",
            trust_remote_code=True
        )
    
    model.eval()
    model.to("cuda")
    
    # Run inference
    print("Running inference...")
    inputs = inputs.to("cuda")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
        )
    
    # Decode output
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nModel output:\n{decoded}\n")
    
    # Parse prediction
    pred_content, pred_x, pred_y = parse_inference(decoded, is_gt=False)
    print(f"Parsed prediction: {pred_content} at ({pred_x}, {pred_y})")
    
    # Extract GT coordinates
    gt_x, gt_y = gt_coord
    if isinstance(gt_x, torch.Tensor):
        gt_x = gt_x.item()
    if isinstance(gt_y, torch.Tensor):
        gt_y = gt_y.item()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    checkpoint_name = os.path.basename(checkpoint_path)
    method = "lora" if is_lora else "full"
    output_filename = f"user{user_id}_sample{sample_idx}_global_{method}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Draw visualization
    print("\nGenerating visualization...")
    is_accurate = draw_prediction_visualization(
        image_path=image_path,
        pred_coord=(pred_x, pred_y),
        gt_coord=(gt_x, gt_y),
        pred_content=pred_content,
        gt_content=gt_content,
        output_path=output_path,
        delta=140
    )
    
    print(f"\n✓ Visualization complete!")
    print(f"  Output: {output_path}")
    
    # Cleanup
    del model, inputs, outputs
    torch.cuda.empty_cache()
    
    return is_accurate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize global model predictions on GUI screenshots'
    )
    parser.add_argument(
        'gpu', type=str,
        help='GPU device to use (e.g., "0")'
    )
    parser.add_argument(
        '--user', '-u', type=int, required=True,
        help='User ID to visualize (1-10)'
    )
    parser.add_argument(
        '--sample', '-s', type=int, default=None,
        help='Single sample index to visualize'
    )
    parser.add_argument(
        '--samples', type=int, nargs='+', default=None,
        help='Multiple sample indices to visualize (e.g., --samples 0 1 2 3)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Visualize all samples for the user'
    )
    parser.add_argument(
        '--lora', action='store_true',
        help='Use LoRA global model (default: full fine-tuned model)'
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, default=None,
        help='Custom checkpoint path (overrides --lora)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./visualizations_global',
        help='Output directory for visualizations (default: ./visualizations_global)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.user not in range(1, 11):
        print(f"Error: user must be between 1 and 10, got {args.user}")
        sys.exit(1)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        # Infer if LoRA from path
        is_lora = 'lora' in checkpoint_path.lower()
        print(f"Using custom checkpoint: {checkpoint_path}")
    elif args.lora:
        checkpoint_path = CHECKPOINT_GLOBAL_LORA
        is_lora = True
        print(f"Using global LoRA checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = CHECKPOINT_GLOBAL_FULL
        is_lora = False
        print(f"Using global full checkpoint: {checkpoint_path}")
    
    # Determine which samples to visualize
    if args.all:
        # Load dataset to get total number of samples
        from dataset import Qwen3VLEvalDataset
        from config import DATASET_ROOT
        eval_dataset = Qwen3VLEvalDataset([args.user], DATASET_ROOT)
        sample_indices = list(range(len(eval_dataset)))
        print(f"Visualizing all {len(sample_indices)} samples for user {args.user}")
    elif args.samples is not None:
        sample_indices = args.samples
        print(f"Visualizing {len(sample_indices)} samples: {sample_indices}")
    elif args.sample is not None:
        sample_indices = [args.sample]
        print(f"Visualizing single sample: {args.sample}")
    else:
        # Default to sample 0
        sample_indices = [0]
        print(f"Visualizing default sample: 0")
    
    # Visualize each sample
    success_count = 0
    total_count = 0
    for i, sample_idx in enumerate(sample_indices):
        print(f"\n{'='*80}")
        print(f"Processing sample {i+1}/{len(sample_indices)}: index {sample_idx}")
        print(f"{'='*80}")
        
        is_accurate = visualize_sample(
            checkpoint_path=checkpoint_path,
            user_id=args.user,
            sample_idx=sample_idx,
            is_lora=is_lora,
            output_dir=args.output
        )
        
        if is_accurate is not None:
            total_count += 1
            if is_accurate:
                success_count += 1
    
    if total_count > 0:
        print(f"\n{'='*80}")
        print(f"Completed! Generated {total_count} visualizations ({success_count} accurate, {total_count - success_count} inaccurate) in {args.output}")
        print(f"Accuracy: {success_count}/{total_count} = {100*success_count/total_count:.1f}%")
        print(f"{'='*80}")
