"""
K-Fold utilities for personalized training
Ensures consistent fold splits across training and evaluation
"""

import random
from typing import List


def create_k_folds(samples: List, k: int = 5, seed: int = 42) -> List[List]:
    """
    Split samples into K folds randomly with a fixed seed.
    
    CRITICAL: This function must be used by both training and evaluation
    to ensure the same fold splits are used consistently.
    
    Args:
        samples: List of all samples to split
        k: Number of folds
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of K folds, where each fold is a list of samples
    """
    random.seed(seed)
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    fold_size = len(shuffled_samples) // k
    folds = []
    
    for i in range(k):
        if i == k - 1:
            # Last fold gets remaining samples
            fold = shuffled_samples[i * fold_size:]
        else:
            fold = shuffled_samples[i * fold_size:(i + 1) * fold_size]
        folds.append(fold)
    
    return folds


def get_training_folds(folds: List[List], test_fold_idx: int) -> List:
    """
    Get training samples by combining all folds except the test fold.
    
    Args:
        folds: List of K folds from create_k_folds()
        test_fold_idx: Index of fold to hold out for testing (0 to K-1)
    
    Returns:
        Combined list of training samples
    """
    training_samples = []
    for i, fold in enumerate(folds):
        if i != test_fold_idx:
            training_samples.extend(fold)
    return training_samples


def get_test_fold(folds: List[List], test_fold_idx: int) -> List:
    """
    Get test samples for a specific fold.
    
    Args:
        folds: List of K folds from create_k_folds()
        test_fold_idx: Index of fold to use for testing (0 to K-1)
    
    Returns:
        Test fold samples
    """
    return folds[test_fold_idx]
