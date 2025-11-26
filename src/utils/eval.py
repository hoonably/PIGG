"""

"""

import re
import os
import numpy as np

def parse_inference(text, is_gt = True):
    if not is_gt :
        text = text.split("assistant")[-1]
    # Referring content
    referring_match = re.findall(r'```referring\n(.*?)\n```', text, re.DOTALL)
    content = referring_match[-1] if referring_match else "None"

    x = y = 0

    # 1. grounding 블록으로 split
    if 'grounding' in text :
        blocks = text.split('```grounding')
    else :
        blocks = [text]
        
    if len(blocks) > 1:
        # 첫 번째 블록 이후가 실제 grounding 블록들
        for block in blocks[1:]:
            if '```' in block:
                block_content = block.split('```')[0]

                # 2. 다양한 좌표 패턴 적용
                patterns = [
                    r'\(?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)?',                  # 4값 좌표
                    r'\(?(\d+)\s*,\s*(\d+)\)?\s*to\s*\(?(\d+)\s*,\s*(\d+)\)?',           # (x1,y1) to (x2,y2)
                    r'(?:Top-left|Bottom-right)[^:]*:\s*\(?(\d+)\s*,\s*(\d+)\)?',        # Top-left / Bottom-right
                    r'\[?(\d+)\s*,\s*(\d+)\]?',                                         # [x,y]
                    r'\(?(\d+)\s*,\s*(\d+)\)?'                                          # (x,y)
                ]

                for pattern in patterns:
                    match = re.search(pattern, block_content)
                    if match:
                        nums = [int(g) for g in match.groups() if g and g.isdigit()]
                        if len(nums) >= 2:
                            x, y = nums[0], nums[1]  # 첫 좌표 기준
                        break
                break  # 첫 번째 블록만 처리하고 싶으면 여기서 break

    return content, x, y



def save_results(result, save_path) :
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "results.txt"), "w") as f :
        f.write(str(result))

def evaluate_coordinates(pred_coords, gt_coords, delta=140):
    """    
    Parameters:
        pred_coords (list of tuple): [(x1, y1), (x2, y2), ...] 예측 좌표
        gt_coords (list of tuple): [(x1, y1), (x2, y2), ...] 정답 좌표
        delta (float, optional): Click Accuracy 기준 거리(px)
    
    Returns:
        dict: {
            "click_accuracy": float,
            "mean_l2_error": float,
            "median_l2_error": float
        }
    """
    pred_coords = np.array(pred_coords)
    gt_coords = np.array(gt_coords)
    
    # L2 거리 계산
    distances = np.linalg.norm(pred_coords - gt_coords, axis=1)
    
    # Click Accuracy@δ
    click_acc = np.mean(distances <= delta)
    
    # Mean / Median L2 Error
    mean_l2 = np.mean(distances)
    median_l2 = np.median(distances)
    
    return {
        "accuracy": click_acc,
        "mean_l2": mean_l2,
        "median_l2": median_l2
    }

