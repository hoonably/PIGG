"""
Training utilities for Qwen3-VL
Following official Qwen3-VL fine-tuning approach
"""

import torch
from typing import Dict, List, Sequence
from dataclasses import dataclass
from transformers import Qwen3VLProcessor, TrainingArguments
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IGNORE_INDEX


@dataclass
class Qwen3VLDataCollator:
    """
    Data collator for Qwen3-VL training.
    
    Based on official Qwen3-VL implementation from:
    https://github.com/QwenLM/Qwen2-VL/blob/main/qwenvl/data/dataset.py
    
    Key features:
    - Uses apply_chat_template(tokenize=True) for proper preprocessing
    - Creates labels with assistant-only masking
    - Handles vision inputs (images/videos)
    """
    
    processor: Qwen3VLProcessor
    
    def __call__(self, instances: Sequence[List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of message sequences.
        
        Args:
            instances: List of message lists from dataset
            
        Returns:
            Batch dictionary with input_ids, labels, attention_mask, pixel_values, image_grid_thw
        """
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []
        
        for idx, messages in enumerate(instances):
            # DEBUG: Check message roles (only first batch)
            if not hasattr(self, '_role_debug_printed'):
                print(f"\n[DEBUG] Input messages (sample 0):")
                for i, msg in enumerate(messages):
                    print(f"  Message {i}: role='{msg['role']}'")
                self._role_debug_printed = True
            
            # Convert 'system' to 'user' for Qwen3-VL compatibility
            converted_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    converted_messages.append({
                        'role': 'user',
                        'content': msg['content']
                    })
                else:
                    converted_messages.append(msg)
            
            # Step 1: Apply chat template with tokenization
            # This handles vision tokens and image processing automatically
            inputs = self.processor.apply_chat_template(
                converted_messages,
                tokenize=True,
                add_generation_prompt=False,  # Critical: False for training
                return_dict=True,
                return_tensors="pt"
            )
            
            # Step 2: Create labels (mask everything except assistant responses)
            input_ids = inputs["input_ids"].squeeze(0)
            labels = self._create_labels_with_assistant_masking(input_ids)
            
            # Step 3: Collect batch data
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(inputs["attention_mask"].squeeze(0))
            
            if "pixel_values" in inputs:
                all_pixel_values.append(inputs["pixel_values"])
            if "image_grid_thw" in inputs:
                all_image_grid_thw.append(inputs["image_grid_thw"])
        
        # Step 4: Pad sequences to max length in batch
        max_len = max(ids.size(0) for ids in all_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        # Pad input_ids, labels, attention_mask
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        
        for input_ids, labels, attention_mask in zip(all_input_ids, all_labels, all_attention_mask):
            padding_length = max_len - input_ids.size(0)
            
            # Pad to the right
            padded_input_ids.append(
                torch.cat([input_ids, torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype)])
            )
            padded_labels.append(
                torch.cat([labels, torch.full((padding_length,), IGNORE_INDEX, dtype=labels.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([attention_mask, torch.zeros(padding_length, dtype=attention_mask.dtype)])
            )
        
        # Step 5: Build final batch
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask),
        }
        
        # Concatenate vision inputs if present
        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        
        return batch
    
    def _create_labels_with_assistant_masking(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create labels with assistant-only masking.
        
        Official Qwen3-VL approach:
        - User inputs and system prompts: label = IGNORE_INDEX (-100)
        - Assistant responses only: label = input_ids
        
        Token structure in Qwen chat template:
        <|im_start|>user\n...user content...<|im_end|>\n
        <|im_start|>assistant\n...RESPONSE TOKENS...<|im_end|>\n
        
        We only train on RESPONSE TOKENS between:
        "<|im_start|>assistant" and "<|im_end|>"
        
        Note: The actual tokenization may vary - sometimes there's no \n after assistant
        """
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # Get special token IDs
        im_start_id = self.processor.tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        assistant_id = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
        im_end_id = self.processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        
        # Find all assistant response regions
        input_ids_list = input_ids.tolist()
        i = 0
        trainable_count = 0
        
        while i < len(input_ids_list) - 1:
            # Look for pattern: <|im_start|> + assistant (with or without \n)
            if (input_ids_list[i] == im_start_id and 
                i + 1 < len(input_ids_list) and
                input_ids_list[i+1] == assistant_id):
                
                # Found assistant start, skip past <|im_start|>assistant and any separator
                start_idx = i + 2
                
                # Skip optional newline or other separator token
                if start_idx < len(input_ids_list) and input_ids_list[start_idx] in [198, 271]:
                    start_idx += 1
                
                # Find the corresponding <|im_end|>
                end_idx = start_idx
                while end_idx < len(input_ids_list):
                    if input_ids_list[end_idx] == im_end_id:
                        # Unmask assistant response tokens (excluding <|im_end|>)
                        labels[start_idx:end_idx] = input_ids[start_idx:end_idx]
                        trainable_count += (end_idx - start_idx)
                        break
                    end_idx += 1
                
                i = end_idx + 1 if end_idx < len(input_ids_list) else len(input_ids_list)
            else:
                i += 1
        
        # DEBUG: Print warning if no trainable tokens found (only once)
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG] Label Masking Check:")
            print(f"  Input length: {len(input_ids_list)}")
            print(f"  im_start: {im_start_id}, assistant: {assistant_id}, im_end: {im_end_id}")
            print(f"  Trainable tokens found: {trainable_count}")
            
            # Search for assistant token in entire sequence
            if assistant_id in input_ids_list:
                idx_assistant = input_ids_list.index(assistant_id)
                print(f"  Found assistant token at index {idx_assistant}")
                print(f"  Context: {input_ids_list[max(0,idx_assistant-3):idx_assistant+8]}")
            
            if trainable_count == 0:
                print(f"  ⚠️  WARNING: No trainable labels created! Training will fail.")
            else:
                print(f"  ✓ Labels created successfully")
            self._debug_printed = True
        
        return labels


def get_training_arguments(output_dir: str, **kwargs) -> TrainingArguments:
    """
    Create TrainingArguments with Qwen3-VL recommended settings.
    
    Based on official Qwen3-VL training scripts.
    """
    from config import (
        BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LR, EPOCHS,
        SAVE_STEPS, WARMUP_STEPS, WEIGHT_DECAY
    )
    
    default_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LR,
        "num_train_epochs": EPOCHS,
        "bf16": True,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": 3,
        "logging_steps": 10,
        "logging_first_step": True,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,  # Critical: keep all columns
        "dataloader_num_workers": 0,  # Avoid multiprocessing issues
        "report_to": [],  # Disable reporting (tensorboard not installed)
        "load_best_model_at_end": True if kwargs.get("eval_strategy") == "steps" else False,
        "metric_for_best_model": "eval_loss" if kwargs.get("eval_strategy") == "steps" else None,
    }
    
    # Override with user-provided kwargs
    default_args.update(kwargs)
    
    return TrainingArguments(**default_args)


def prepare_model_for_training(model):
    """
    Prepare model for memory-efficient training.
    
    - Enable gradient checkpointing
    - Enable input gradients (required for some PEFT methods)
    """
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model
