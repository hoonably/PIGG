"""
Qwen3-VL Dataset Classes
Following official Qwen3-VL data format
"""

import os
import pickle
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any
from transformers import AutoProcessor
from utils.eval import parse_inference
from PIL import Image
from config import MAX_PIXELS


class Qwen3VLTrainDataset(Dataset):
    """
    Training dataset for Qwen3-VL.
    
    Returns raw messages - all processing is done in the data collator.
    This follows the official Qwen3-VL approach.
    """
    
    def __init__(self, user_ids: List[int], data_path: str):
        super().__init__()
        self.data_path = data_path
        self.samples = []
        
        # Load data for specified users
        for user_id in user_ids:
            pkl_path = os.path.join(data_path, f"user_{user_id}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    user_data = pickle.load(f)
                self.samples.extend(user_data)
            else:
                print(f"Warning: {pkl_path} not found, skipping user {user_id}")
    
    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        """
        Returns messages in Qwen3-VL format.
        
        Expected format:
        [
            {
                "role": "user" or "system",
                "content": [
                    {"type": "image", "image": "path/to/image.jpg"},
                    {"type": "text", "text": "instruction..."}
                ]
            },
            {
                "role": "assistant",
                "content": "response..."
            }
        ]
        
        Important: Qwen3-VL requires vision content to have role='user'
        """
        messages = self.samples[idx]
        
        # Convert 'system' role to 'user' for Qwen3-VL compatibility
        converted_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                # Create new dict with role changed to 'user'
                converted_messages.append({
                    'role': 'user',
                    'content': msg['content']
                })
            else:
                converted_messages.append(msg)
        
        return converted_messages
    
    def __len__(self) -> int:
        return len(self.samples)


class Qwen3VLEvalDataset(Dataset):
    """
    Evaluation dataset - returns processed inputs + ground truth for evaluation
    """
    
    def __init__(self, user_ids: List[int], data_path: str):
        super().__init__()
        self.data_path = data_path
        self.samples = []
        self.gt_contents = []
        self.gt_coords = []
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", 
            trust_remote_code=True, 
            max_pixels=MAX_PIXELS
        )
        
        # Load data for specified users
        for user_id in user_ids:
            pkl_path = os.path.join(data_path, f"user_{user_id}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    user_data = pickle.load(f)
                self.samples.extend(user_data)
            else:
                print(f"Warning: {pkl_path} not found, skipping user {user_id}")
        
        # Preprocess to extract ground truth
        self._extract_ground_truth()
    
    def _extract_ground_truth(self):
        """Extract ground truth content and coordinates from assistant responses"""
        for sample in self.samples:
            # Find assistant message
            for msg in sample:
                if msg['role'] == 'assistant':
                    gt_text = msg['content'][0]['text'] if isinstance(msg['content'], list) else msg['content']
                    gt_content, gt_x, gt_y = parse_inference(gt_text)
                    self.gt_contents.append(gt_content)
                    self.gt_coords.append((gt_x, gt_y))
                    break
    
    def __getitem__(self, idx: int):
        """
        Returns (processed_inputs, gt_content, gt_coord) for evaluation
        """
        messages = self.samples[idx]
        
        # Convert first message (with image) to user role
        user_message = messages[0].copy()
        user_message['role'] = 'user'
        
        # Apply chat template with generation prompt (for inference)
        text = self.processor.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Extract image path
        image_path = None
        for content_item in user_message['content']:
            if content_item['type'] == 'image':
                image_path = content_item['image']
                break
        
        # Process text and image
        processed = self.processor(
            text=[text],
            images=[image_path] if image_path else None,
            return_tensors="pt",
            padding=True
        )
        
        return processed, self.gt_contents[idx], self.gt_coords[idx]
    
    def __len__(self) -> int:
        return len(self.samples)
