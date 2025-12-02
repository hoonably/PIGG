"""
Qwen3-VL Evaluation Script
"""

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from config import *
from utils.eval import parse_inference, evaluate_coordinates
from dataset import Qwen3VLEvalDataset
from peft import PeftModel
from tqdm import tqdm
import os
import natsort
import json
import gc


def eval_checkpoint(checkpoint, user_eval, save_root, is_lora = False, eval_dataset = None) :
    if not eval_dataset :
        eval_dataset = Qwen3VLEvalDataset(user_eval, DATASET_ROOT)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, num_workers=NUM_WORKERS)
    
    if is_lora :
        # Use global full checkpoint as base (matching training setup)
        print(f"Loading base model from: {CHECKPOINT_GLOBAL_FULL}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CHECKPOINT_GLOBAL_FULL, 
            dtype="auto", 
            trust_remote_code=True
        )
        print(f"Loading LoRA weights from: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
    else :
        model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint, dtype="auto", trust_remote_code=True)
    

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
    model.eval()
    model.to("cuda")

    gt_coords = []
    pred_coords = []
    gt_contents = []
    pred_contents = []
    out = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(eval_dataloader)):
            out_json = {
                
            }
            inputs, gt_content, gt_coord = batch
            
            inputs = inputs.to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
            inputs["input_ids"] = inputs["input_ids"].squeeze(0)
            inputs["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            out_json["pred_raw"] = decoded
            
            pred_content, pred_x, pred_y = parse_inference(decoded, is_gt=False)
            
            out_json["gt_coord"] = [c.item() if isinstance(c, torch.Tensor) else c for c in gt_coord]
            out_json["pred_coord"] = (pred_x, pred_y)
            out_json["gt_content"] = gt_content[0]
            out_json["pred_content"] = pred_content
                        
            gt_coords.append([c.item() if isinstance(c, torch.Tensor) else c for c in gt_coord])
            pred_coords.append((pred_x, pred_y))
            gt_contents.append(gt_content[0])
            pred_contents.append(pred_content)    
            out.append(out_json)
            
            with open(os.path.join(save_root, "out.json"), "w") as f :
                json.dump(out, f, ensure_ascii=False, indent=4)
    
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    
    return evaluate_coordinates(pred_coords,gt_coords, delta=140)

def eval_user(checkpoint_path, user_eval, save_root, is_lora = False, eval_dataset = None):
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    
    if not eval_dataset :
        eval_dataset = Qwen3VLEvalDataset(user_eval, DATASET_ROOT)
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, num_workers=NUM_WORKERS)
    if checkpoint_path != "Qwen/Qwen3-VL-8B-Instruct" :
        # Check if the path itself is already a checkpoint directory
        # For full models: config.json, For LoRA: adapter_config.json
        has_config = os.path.exists(os.path.join(checkpoint_path, "config.json"))
        has_adapter_config = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
        
        if has_config or has_adapter_config:
            # Path is already a checkpoint directory (e.g., checkpoint-500 or final model)
            checkpoint_path_lastest = checkpoint_path
            model_type = "LoRA adapter" if has_adapter_config else "full model"
            print(f"Using checkpoint directly ({model_type}): {checkpoint_path}")
        else:
            # Path is a parent directory containing checkpoints
            dir_list = os.listdir(checkpoint_path)
            checkpoint_list = []
            for d in dir_list :
                if "checkpoint" in d :
                    checkpoint_list.append(d)
            
            if not checkpoint_list:
                raise FileNotFoundError(
                    f"No checkpoint directories found in {checkpoint_path}. "
                    f"Available files/dirs: {dir_list}"
                )
            
            checkpoint_list = natsort.natsorted(checkpoint_list)
            checkpoint_path_lastest = os.path.join(checkpoint_path, checkpoint_list[-1])
            print(f"Using latest checkpoint: {checkpoint_path_lastest}")
    else :
        checkpoint_path_lastest = checkpoint_path
        is_lora = False
    
    os.makedirs(save_root, exist_ok=True)

    if is_lora :
        # Use global full checkpoint as base (matching training setup)
        print(f"Loading base model from: {CHECKPOINT_GLOBAL_FULL}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CHECKPOINT_GLOBAL_FULL,
            dtype="auto",
            trust_remote_code=True
        )
        print(f"Loading LoRA weights from: {checkpoint_path_lastest}")
        model = PeftModel.from_pretrained(model, checkpoint_path_lastest)
    else :
        model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint_path_lastest, dtype="auto", trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
    model.eval()
    model.to("cuda")

    gt_coords = []
    pred_coords = []
    gt_contents = []
    pred_contents = []
    out = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(eval_dataloader)):
            out_json = {
                
            }
            inputs, gt_content, gt_coord = batch
            
            inputs = inputs.to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
            inputs["input_ids"] = inputs["input_ids"].squeeze(0)
            inputs["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            out_json["pred_raw"] = decoded
            
            pred_content, pred_x, pred_y = parse_inference(decoded, is_gt=False)
            
            out_json["gt_coord"] = [c.item() if isinstance(c, torch.Tensor) else c for c in gt_coord]
            out_json["pred_coord"] = (pred_x, pred_y)
            out_json["gt_content"] = gt_content[0]
            out_json["pred_content"] = pred_content
                        
            gt_coords.append([c.item() if isinstance(c, torch.Tensor) else c for c in gt_coord])
            pred_coords.append((pred_x, pred_y))
            gt_contents.append(gt_content[0])
            pred_contents.append(pred_content)    
            out.append(out_json)
            
            with open(os.path.join(save_root, "out.json"), "w") as f :
                json.dump(out, f, ensure_ascii=False, indent=4)
    
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    
    return evaluate_coordinates(pred_coords,gt_coords, delta=140)
    