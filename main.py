#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCFCL: Decentralized Dynamic Cooperation of Personalized Models for Federated Continual Learning

Main entry point for training and evaluation.

Paper: "Decentralized Dynamic Cooperation of Personalized Models for Federated Continual Learning", NeurIPS'25

Usage:
    # Using DCFCL algorithm (default, the proposed method)
    python main.py --algorithm DCFCL --dataset EMNIST-Letters
    
    # Using baseline methods
    python main.py --algorithm FedAvg --dataset EMNIST-Letters
    python main.py --algorithm FedProx --dataset EMNIST-Letters
    python main.py --algorithm Local --dataset EMNIST-Letters
"""
  
import argparse
import logging
import os
import random
import numpy as np
import torch
import yaml
from datetime import datetime

from core.server import DCFCLServer
from core.config import Config
from utils.helpers import setup_seed, setup_logging


def parse_args():
    """Parse command line arguments.
    
    When `--config` is provided, the YAML values are used as parser defaults,
    while any explicitly supplied CLI flags still take precedence.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        with open(pre_args.config, 'r') as f:
            config_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(
        description='DCFCL: Decentralized Dynamic Cooperation for Federated Continual Learning'
    )
    
    # ==================== Basic Settings ====================
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (overrides command line args)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (auto will use CUDA if available)')
    
    # ==================== Dataset Settings ====================
    parser.add_argument('--dataset', type=str, default='EMNIST-Letters',
                        choices=['EMNIST-Letters', 'EMNIST-Letters-shuffle', 'CIFAR100', 
                                 'MNIST-SVHN-FASHION', 'TEST-noniid'],
                        help='Dataset to use')
    parser.add_argument('--datadir', type=str, default='./datasets',
                        help='Directory for datasets')
    parser.add_argument('--data_split_file', type=str, 
                        default='split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl',
                        help='Path to data split file (relative to datadir)')
    
    # ==================== FL Settings ====================
    parser.add_argument('--num_users', type=int, default=8,
                        help='Number of clients')
    parser.add_argument('--num_tasks', type=int, default=6,
                        help='Number of tasks for continual learning')
    parser.add_argument('--num_rounds', type=int, default=60,
                        help='Total number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=100,
                        help='Number of local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    
    # ==================== Algorithm Selection ====================
    parser.add_argument('--algorithm', type=str, default='DCFCL',
                        choices=['DCFCL', 'D2FCL', 'FedAvg', 'FedProx', 'FedLwF', 'Local', 
                                 'SCAFFOLD', 'PerAvg', 'pFedMe', 'ClusterFL'],
                        help='Federated learning algorithm')
    
    # ==================== Optimizer Settings ====================
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (paper C.4: 1e-4 for EMNIST)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (paper C.4: 1e-5)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    
    # ==================== Algorithm-Specific Settings ====================
    # FedProx
    parser.add_argument('--mu', type=float, default=0.005,
                        help='FedProx proximal term coefficient')
    
    # SCAFFOLD
    parser.add_argument('--glo_lr', type=float, default=1.0,
                        help='SCAFFOLD global learning rate')
    
    # PerAvg / pFedMe
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Moving average parameter for pFedMe / second LR for PerAvg')
    parser.add_argument('--personal_lr', type=float, default=0.09,
                        help='Personalized learning rate for pFedMe')
    parser.add_argument('--lamda', type=float, default=5,
                        help='Regularization term for pFedMe')
    parser.add_argument('--K', type=int, default=3,
                        help='Number of personalization steps for pFedMe')
    
    # ==================== DCFCL Specific Settings ====================
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='KD loss weight for baselines (FedLwF, etc.)')
    parser.add_argument('--lambda_kd', type=float, default=0.2,
                        help='KD loss weight for DCFCL (paper Eq.6, Table 3: lambda=0.2)')
    parser.add_argument('--sw', type=float, default=0.2,
                        help='Similarity weight (paper epsilon=0.2 in Eq.7)')
    parser.add_argument('--temp', type=float, default=0.1,
                        help='Temperature for knowledge distillation')
    parser.add_argument('--ema_global', type=float, default=0.9,
                        help='EMA smoothing factor for global prototypes')
    parser.add_argument('--global_weight', type=float, default=0.9,
                        help='Weight for global model aggregation')
    parser.add_argument('--lambda_proto_aug', type=float, default=2.0,
                        help='Prototype augmentation loss weight (best: 2.0 for EMNIST)')
    parser.add_argument('--proto_queue_length', type=int, default=100,
                        help='Length of prototype queue')
    parser.add_argument('--dcfcl_broadcast', type=int, default=0,
                        help='0=pure coalition (paper), 1=EMA+broadcast (AFCL code), 2=hybrid')
    parser.add_argument('--ema_blend', type=float, default=0.5,
                        help='Blend weight for hybrid mode: final = (1-ema_blend)*coalition + ema_blend*EMA')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for CE loss (regularization)')
    
    # ==================== DER (Dark Experience Replay) Settings ====================
    parser.add_argument('--use_der', action='store_true', default=True,
                        help='Enable DER++ replay module for D2FCL (master switch)')
    parser.add_argument('--no_use_der', dest='use_der', action='store_false',
                        help='Disable DER++ replay module (ablation: D2FCL → DCFCL behavior)')
    parser.add_argument('--buffer_size', type=int, default=500,
                        help='Replay buffer size per client (D2FCL)')
    parser.add_argument('--der_alpha', type=float, default=0.5,
                        help='DER logit-matching loss weight (D2FCL)')
    parser.add_argument('--der_beta', type=float, default=0.5,
                        help='DER++ replay CE loss weight (D2FCL)')
    
    # ==================== Directed Collaboration Settings ====================
    parser.add_argument('--directed_collaboration', action='store_true',
                        help='Enable directed collaboration mechanism')
    parser.add_argument('--directed_threshold', type=float, default=0.0,
                        help='Threshold for directed collaboration scores')
    parser.add_argument('--directed_mode', type=str, default='gradient',
                        choices=['gradient', 'task_aware', 'hybrid'],
                        help='Directed collaboration mode')
    parser.add_argument('--directed_temperature', type=float, default=1.0,
                        help='Temperature for softmax weighting in directed aggregation')
    parser.add_argument('--directed_self_weight', type=float, default=0.5,
                        help='Weight for client own model in personalized aggregation')
    
    # ==================== Coalition Mask Settings ====================
    parser.add_argument('--use_coalition_mask', action='store_true',
                        help='Enable coalition formation constraints (some clients cannot cooperate)')
    parser.add_argument('--coalition_mask_type', type=str, default='custom',
                        choices=['custom', 'random', 'group'],
                        help='Type of coalition mask: custom (use forbidden_pairs), random, or group-based')
    parser.add_argument('--coalition_mask_density', type=float, default=0.2,
                        help='For random mask: probability of forbidden pair')
    parser.add_argument('--num_client_groups', type=int, default=2,
                        help='For group mask: number of client groups')
    
    # ==================== Model Settings ====================
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension')
    
    # ==================== Logging ====================
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained models')
    
    # ==================== Emergence Evaluation ====================
    parser.add_argument('--evaluate_emergence', action='store_true',
                        help='Enable emergence phenomenon evaluation')
    parser.add_argument('--save_emergence_samples', action='store_true', default=True,
                        help='Save emergence samples for analysis')

    if config_defaults:
        valid_defaults = {k: v for k, v in config_defaults.items() if k in {a.dest for a in parser._actions}}
        parser.set_defaults(**valid_defaults)
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments (YAML supplies defaults; explicit CLI args still win)
    args = parse_args()
    
    # Setup seed for reproducibility
    setup_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    
    # Create result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(args.result_dir, f"{args.algorithm}_{args.dataset}_{timestamp}")
    os.makedirs(result_path, exist_ok=True)
    args.result_path = result_path
    
    # Setup logging
    setup_logging(result_path, args.log_level)
    logger = logging.getLogger('DCFCL')
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("DCFCL: Federated Continual Learning")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    logger.info(f"Clients: {args.num_users}, Tasks: {args.num_tasks}, Rounds: {args.num_rounds}")
    logger.info(f"Local epochs: {args.local_epochs}, Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Result path: {result_path}")
    logger.info("=" * 60)
    
    # Create config object
    config = Config(args)
    
    # Create and run server
    server = DCFCLServer(config, device)
    
    # Train
    logger.info("Starting training...")
    results = server.train()
    
    # Log final results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Final Accuracy: {results['final_accuracy']:.4f}")
    if 'forgetting_rate' in results:
        logger.info(f"Forgetting Rate: {results['forgetting_rate']:.4f}")
    
    # Log detailed per-task metrics
    if 'per_task_acc' in results and results['per_task_acc']:
        pta = results['per_task_acc']
        ptf = results['per_task_forget']
        num_tasks = len(pta)
        rpt = args.num_rounds // args.num_tasks  # rounds per task
        all_acc = results.get('all_accuracies', [])
        all_forget = results.get('all_forgetting', [])
        
        logger.info("-" * 60)
        logger.info("Per-Task Accuracy (after each task phase):")
        for phase in range(num_tasks):
            accs_str = ", ".join(f"T{t}={pta[phase][t]:.4f}" for t in range(len(pta[phase])))
            # Use all_acc (weighted) for the overall average at end of phase
            round_idx = phase * rpt + rpt - 1
            overall = all_acc[round_idx] if round_idx < len(all_acc) else 0.0
            logger.info(f"  After Task {phase}: [{accs_str}]  overall_avg={overall:.4f}")
        
        logger.info("Per-Task Forgetting (after each task phase):")
        for phase in range(num_tasks):
            fgt_str = ", ".join(f"T{t}={ptf[phase][t]:.4f}" for t in range(len(ptf[phase])))
            past_forgets = [ptf[phase][t] for t in range(phase)]
            avg_forget = sum(past_forgets) / len(past_forgets) if past_forgets else 0.0
            logger.info(f"  After Task {phase}: [{fgt_str}]  avg_forget={avg_forget:.4f}")
        
        # Summary — avg_task_accuracy = average of end-of-phase overall accuracies
        phase_overall = [all_acc[p * rpt + rpt - 1] for p in range(num_tasks) if p * rpt + rpt - 1 < len(all_acc)]
        avg_task_acc = sum(phase_overall) / len(phase_overall) if phase_overall else 0.0
        avg_all_forget = sum(all_forget) / len(all_forget) if all_forget else 0.0
        logger.info("-" * 60)
        logger.info(f"Phase-end overall accuracies: {[round(a,4) for a in phase_overall]}")
        logger.info(f"Avg Task Accuracy (weighted): {avg_task_acc:.4f}")
        logger.info(f"Avg Forgetting: {avg_all_forget:.4f}")
    
    logger.info("=" * 60)
    
    # Save results
    import json
    
    def _serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(os.path.join(result_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_serialize)
    
    return results


if __name__ == "__main__":
    main()
