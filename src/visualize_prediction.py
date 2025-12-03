"""
Visualize prediction results on GUI screenshots (Vanilla Qwen3-VL)
Shows predicted point, ground truth, and 14% accuracy circle

Usage:
    python visualize_prediction.py 0 --user 1 --sample 0           # Single sample
    python visualize_prediction.py 0 --user 1 --samples 4 5 6 7 8  # Multiple samples
    python visualize_prediction.py 0 --user 1 --all                # All samples
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
from peft import PeftModel

from config import DATASET_ROOT, MAX_PIXELS
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
    # Detect coordinate type based on image size and coordinate range
    pred_x, pred_y = pred_coord
    gt_x, gt_y = gt_coord
    
    # Check if prediction is relative (0-1000 range) or absolute (matches image size)
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
    
    # Draw accuracy circle around prediction (14% range)
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
    
    # Try to load a font, fallback to default if not available
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
    
    # Save result (no text overlay on image)
    img.save(output_path)
    print(f"Visualization saved to: {output_path}")
    print(f"  Image size: {width}x{height}")
    print(f"  GT: ({gt_x}, {gt_y}) [{gt_coord_type}] -> abs ({gt_x_abs}, {gt_y_abs})")
    print(f"  Pred: ({pred_x}, {pred_y}) [{pred_coord_type}] -> abs ({pred_x_abs}, {pred_y_abs})")
    print(f"  Distance: {distance:.1f}px (threshold: {delta_scaled}px)")
    print(f"  Accurate: {'YES' if is_accurate else 'NO'}")
    
    return is_accurate


def visualize_sample(user_id, sample_idx, output_dir="./visualizations"):
    """
    Visualize a specific sample from evaluation dataset using vanilla Qwen3-VL.
    
    Args:
        user_id: User ID
        sample_idx: Index of sample to visualize
        output_dir: Directory to save visualizations
    """
    print("=" * 80)
    print("Prediction Visualization (Vanilla Qwen3-VL)")
    print("=" * 80)
    print(f"User: {user_id}")
    print(f"Sample: {sample_idx}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    eval_dataset = Qwen3VLEvalDataset([user_id], DATASET_ROOT)
    
    if sample_idx >= len(eval_dataset):
        print(f"Error: Sample index {sample_idx} out of range (dataset has {len(eval_dataset)} samples)")
        return
    
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
        return
    
    print(f"Image: {image_path}")
    print(f"Ground truth: {gt_content} at {gt_coord}")
    print()
    
    # Load vanilla model
    print("Loading vanilla Qwen3-VL model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    
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
    output_filename = f"user{user_id}_sample{sample_idx}_vanilla.png"
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
    return is_accurate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize vanilla Qwen3-VL predictions on GUI screenshots'
    )
    parser.add_argument(
        'gpu', type=str,
        help='GPU device to use (e.g., "0")'
    )
    parser.add_argument(
        '--user', '-u', type=int, required=True,
        help='User ID to visualize'
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
        '--output', '-o', type=str, default='./visualizations',
        help='Output directory for visualizations (default: ./visualizations)'
    )
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
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
            user_id=args.user,
            sample_idx=sample_idx,
            output_dir=args.output
        )
        total_count += 1
        if is_accurate:
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Completed! Generated {total_count} visualizations ({success_count} accurate, {total_count - success_count} inaccurate) in {args.output}")
    print(f"Accuracy: {success_count}/{total_count} = {100*success_count/total_count:.1f}%")
    print(f"{'='*80}")
