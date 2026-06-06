#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper utilities for DCFCL.

Provides common utility functions for:
- Random seed setup
- Logging configuration
- Metric computation
"""

import os
import random
import logging
import numpy as np
import torch


def setup_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # For hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(result_path: str, log_level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        result_path: Directory to save log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logger
    logger = logging.getLogger('DCFCL')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = os.path.join(result_path, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        Accuracy value
    """
    return np.mean(predictions == labels)


def compute_forgetting(acc_matrix: np.ndarray) -> float:
    """
    Compute average forgetting rate.
    
    Args:
        acc_matrix: Accuracy matrix of shape (num_tasks, num_tasks)
                   where acc_matrix[i, j] is accuracy on task j after learning task i
                   
    Returns:
        Average forgetting rate
    """
    num_tasks = acc_matrix.shape[0]
    forgetting = 0.0
    count = 0
    
    for j in range(num_tasks - 1):
        # Best accuracy on task j before final task
        best_acc = np.max(acc_matrix[j:num_tasks-1, j])
        # Final accuracy on task j
        final_acc = acc_matrix[-1, j]
        forgetting += max(0, best_acc - final_acc)
        count += 1
    
    return forgetting / count if count > 0 else 0.0


def print_summary(results: dict):
    """
    Print training summary.
    
    Args:
        results: Dictionary containing training results
    """
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    if 'final_accuracy' in results:
        print(f"Final Accuracy: {results['final_accuracy']:.4f}")
    
    if 'forgetting_rate' in results:
        print(f"Forgetting Rate: {results['forgetting_rate']:.4f}")
    
    if 'all_accuracies' in results:
        print(f"\nPer-round Accuracies:")
        for i, acc in enumerate(results['all_accuracies']):
            print(f"  Round {i}: {acc:.4f}")
    
    print("=" * 60)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
