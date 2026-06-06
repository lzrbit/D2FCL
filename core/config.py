#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration management for DCFCL.

This module provides a clean configuration interface that supports
both command-line arguments and YAML config files.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Config:
    """Configuration class for DCFCL experiments."""
    
    # ==================== Basic Settings ====================
    seed: int = 42
    device: str = 'auto'
    
    # ==================== Dataset Settings ====================
    dataset: str = 'EMNIST-Letters'
    datadir: str = './datasets'
    data_split_file: str = 'split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl'
    
    # ==================== FL Settings ====================
    num_users: int = 8
    num_tasks: int = 6
    num_rounds: int = 60
    local_epochs: int = 100
    batch_size: int = 64
    
    # ==================== Algorithm Settings ====================
    algorithm: str = 'DCFCL'
    
    # ==================== Optimizer Settings ====================
    lr: float = 1e-4  # Paper C.4: 1e-4 for EMNIST, 1e-3 for CIFAR100
    weight_decay: float = 1e-5  # Paper C.4: weightdecay = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    
    # ==================== Algorithm-Specific ====================
    # FedProx
    mu: float = 0.005
    
    # SCAFFOLD
    glo_lr: float = 1.0
    scaffold_lr: float = 0.005  # Local LR for SCAFFOLD (matches original code)
    
    # PerAvg / pFedMe
    beta: float = 0.1
    personal_lr: float = 0.09
    lamda: float = 5.0
    K: int = 3
    
    # ==================== DCFCL Specific ====================
    alpha: float = 1.0  # KD loss weight for baselines (FedLwF, SCAFFOLD, etc.)
    lambda_kd: float = 0.2  # KD loss weight for DCFCL (paper Eq.6, Table 3: λ=0.2)
    sw: float = 0.2     # Similarity weight (paper ε=0.2 in Eq.7, Table 3)
    temp: float = 0.1   # Temperature
    ema_global: float = 0.9
    global_weight: float = 0.9
    lambda_proto_aug: float = 2.0
    proto_queue_length: int = 100
    dcfcl_broadcast: int = 0  # 0=pure coalition (paper), 1=EMA+broadcast (AFCL code), 2=hybrid
    ema_blend: float = 0.5    # Blend weight for hybrid mode: final = (1-ema_blend)*coalition + ema_blend*EMA
    label_smoothing: float = 0.0  # Label smoothing for CE loss (regularization)
    
    # ==================== DER (Dark Experience Replay) Settings ====================
    use_der: bool = True        # Master switch: enable/disable DER++ replay module
    buffer_size: int = 500      # Replay buffer capacity per client
    der_alpha: float = 0.5      # Weight for DER logit-matching loss (MSE on stored logits)
    der_beta: float = 0.5       # Weight for DER++ replay CE loss (CE on stored labels)
    
    # ==================== Directed Collaboration Settings ====================
    directed_collaboration: bool = False  # Enable directed collaboration mechanism
    directed_threshold: float = 0.0       # Threshold for directed collaboration (0 means accept all positive)
    directed_mode: str = 'gradient'       # Mode: 'gradient', 'task_aware', 'hybrid'
    directed_temperature: float = 1.0     # Temperature for softmax weighting
    directed_self_weight: float = 0.5     # Weight for client's own model in personalized aggregation
    
    # ==================== Coalition Mask Settings ====================
    use_coalition_mask: bool = False       # Enable coalition formation constraints
    coalition_forbidden_pairs: List = field(default_factory=list)  # List of (i,j) pairs that cannot be in same coalition
    coalition_mask_type: str = 'custom'    # 'custom' (use forbidden_pairs), 'random' (random mask), 'group' (group-based)
    coalition_mask_density: float = 0.2    # For 'random' type: probability of forbidden pair
    num_client_groups: int = 2             # For 'group' type: number of client groups
    
    # ==================== Model Settings ====================
    model: str = 'cnn'
    feature_dim: int = 512
    
    # ==================== Logging ====================
    result_path: str = './results'
    result_dir: str = './results'  # Alias for result_path
    log_level: str = 'INFO'
    save_model: bool = False
    
    # ==================== Emergence Evaluation Settings ====================
    evaluate_emergence: bool = False       # Enable emergence phenomenon evaluation
    save_emergence_samples: bool = True    # Save emergence samples for analysis
    
    def __init__(self, args=None, **kwargs):
        """Initialize config from argparse namespace, kwargs, or use defaults.
        
        Args:
            args: argparse namespace object
            **kwargs: keyword arguments (from YAML config)
        """
        # Handle argparse namespace
        if args is not None:
            for key, value in vars(args).items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Handle keyword arguments (from YAML)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Sync result_path and result_dir
        if 'result_dir' in kwargs:
            self.result_path = self.result_dir
        
        # Compute derived settings
        self._compute_derived()
    
    def _compute_derived(self):
        """Compute derived configuration values."""
        # Rounds per task
        self.rounds_per_task = self.num_rounds // self.num_tasks
        
        # Number of classes based on dataset
        self.num_classes = self._get_num_classes()
        
        # Feature size based on model
        self.feature_size = self._get_feature_size()
    
    def _get_num_classes(self) -> int:
        """Get number of classes for the dataset."""
        dataset_classes = {
            'EMNIST-Letters': 26,
            'EMNIST-Letters-shuffle': 26,
            'CIFAR100': 100,
            'MNIST-SVHN-FASHION': 20,
            'TEST-noniid': 10,
        }
        return dataset_classes.get(self.dataset, 26)
    
    def _get_feature_size(self) -> int:
        """Get feature dimension based on model."""
        if self.model == 'resnet18':
            return 512
        elif self.model == 'cnn':
            return 512
        else:
            return self.feature_dim
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config._compute_derived()
        return config
