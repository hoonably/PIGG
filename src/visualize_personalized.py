"""
Visualize personalized model predictions on GUI screenshots
Shows predicted point, ground truth, and 14% accuracy circle for personalized models

Usage:
    # Visualize LoRA personalized model
    python visualize_personalized.py 0 --user 1 --fold 0 --sample 0 --lora
    python visualize_personalized.py 0 --user 1 --fold 0 --samples 0 1 2 3 --lora
    python visualize_personalized.py 0 --user 1 --fold 0 --all --lora
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
import pickle

from config import (
    DATASET_ROOT, 
    MAX_PIXELS, 
    CHECKPOINT_PERSON_LORA, 
    CHECKPOINT_PERSON_FULL,
    CHECKPOINT_GLOBAL_FULL,
    K_FOLDS
)
from utils.eval import parse_inference
from utils.kfold import create_k_folds, get_test_fold


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


def load_user_data(user_id: int, data_path: str):
    """Load all samples for a specific user"""
    pkl_path = os.path.join(data_path, f"user_{user_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"User data not found: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        user_data = pickle.load(f)
    
    return user_data


def get_fold_test_samples(user_id: int, fold_idx: int):
    """Get test samples for a specific fold"""
    user_samples = load_user_data(user_id, DATASET_ROOT)
    folds = create_k_folds(user_samples, k=K_FOLDS, seed=42)
    test_samples = get_test_fold(folds, fold_idx)
    return test_samples


def visualize_sample(user_id, fold_idx, sample_idx, is_lora=True, output_dir="./visualizations_personalized"):
    """
    Visualize a specific sample from personalized model's test fold.
    
    Args:
        user_id: User ID
        fold_idx: Fold index
        sample_idx: Index of sample within the test fold
        is_lora: Whether the model is LoRA or full fine-tuned
        output_dir: Directory to save visualizations
    """
    print("=" * 80)
    print("Personalized Model Prediction Visualization")
    print("=" * 80)
    print(f"User: {user_id}")
    print(f"Fold: {fold_idx}")
    print(f"Sample: {sample_idx}")
    print(f"Model type: {'LoRA' if is_lora else 'Full'}")
    print()
    
    # Get checkpoint path
    if is_lora:
        checkpoint_path = os.path.join(CHECKPOINT_PERSON_LORA, f"user_{user_id}", f"fold_{fold_idx}")
    else:
        checkpoint_path = os.path.join(CHECKPOINT_PERSON_FULL, f"user_{user_id}", f"fold_{fold_idx}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please run training first: python 4_finetune_person.py 0 --users {user_id} --fold {fold_idx} {'--lora' if is_lora else ''}")
        return None
    
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load test fold samples
    print("Loading test fold samples...")
    test_samples = get_fold_test_samples(user_id, fold_idx)
    print(f"Test fold has {len(test_samples)} samples")
    
    if sample_idx >= len(test_samples):
        print(f"Error: Sample index {sample_idx} out of range (test fold has {len(test_samples)} samples)")
        return None
    
    # Get sample
    sample = test_samples[sample_idx]
    
    # Extract image path and instruction from sample
    image_path = None
    instruction = None
    gt_text = None
    
    for msg in sample:
        if msg['role'] == 'system':
            for content_item in msg['content']:
                if content_item['type'] == 'image':
                    image_path = content_item['image']
                elif content_item['type'] == 'text':
                    instruction = content_item['text']
        elif msg['role'] == 'assistant':
            gt_text = msg['content'][0]['text'] if isinstance(msg['content'], list) else msg['content']
    
    if not image_path:
        print("Error: Could not find image path in sample")
        return None
    
    # Parse ground truth
    gt_content, gt_x, gt_y = parse_inference(gt_text, is_gt=True)
    
    print(f"Image: {image_path}")
    print(f"Instruction: {instruction}")
    print(f"Ground truth: {gt_content} at ({gt_x}, {gt_y})")
    print()
    
    # Load model
    print("Loading personalized model...")
    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        max_pixels=MAX_PIXELS
    )
    
    if is_lora:
        # Load base model + LoRA adapter
        print(f"  Loading base model from: {CHECKPOINT_GLOBAL_FULL}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CHECKPOINT_GLOBAL_FULL,
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
    
    # Prepare input (convert system role to user)
    messages = []
    for msg in sample:
        if msg['role'] == 'system':
            messages.append({
                'role': 'user',
                'content': msg['content']
            })
            break  # Only use first message for inference
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=[image_path],
        return_tensors="pt",
        padding=True
    ).to("cuda")
    
    # Run inference
    print("Running inference...")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
        )
    
    # Decode output
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nModel output:\n{decoded}\n")
    
    # Parse prediction
    pred_content, pred_x, pred_y = parse_inference(decoded, is_gt=False)
    print(f"Parsed prediction: {pred_content} at ({pred_x}, {pred_y})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    method = "lora" if is_lora else "full"
    output_filename = f"user{user_id}_fold{fold_idx}_sample{sample_idx}_{method}.png"
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
        description='Visualize personalized model predictions on GUI screenshots'
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
        '--fold', '-f', type=int, required=True,
        help=f'Fold index (0-{K_FOLDS-1})'
    )
    parser.add_argument(
        '--sample', '-s', type=int, default=None,
        help='Single sample index to visualize (within test fold)'
    )
    parser.add_argument(
        '--samples', type=int, nargs='+', default=None,
        help='Multiple sample indices to visualize (e.g., --samples 0 1 2 3)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Visualize all samples in the test fold'
    )
    parser.add_argument(
        '--lora', action='store_true',
        help='Use LoRA personalized model (default: full fine-tuned model)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./visualizations_personalized',
        help='Output directory for visualizations (default: ./visualizations_personalized)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.user not in range(1, 11):
        print(f"Error: user must be between 1 and 10, got {args.user}")
        sys.exit(1)
    
    if args.fold not in range(K_FOLDS):
        print(f"Error: fold must be between 0 and {K_FOLDS-1}, got {args.fold}")
        sys.exit(1)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Determine which samples to visualize
    if args.all:
        # Get all samples from test fold
        test_samples = get_fold_test_samples(args.user, args.fold)
        sample_indices = list(range(len(test_samples)))
        print(f"Visualizing all {len(sample_indices)} samples from test fold {args.fold}")
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
            fold_idx=args.fold,
            sample_idx=sample_idx,
            is_lora=args.lora,
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
