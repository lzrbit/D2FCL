#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coalition-mechanism ablation script.

Validates the coalition aggregation mechanism inside DCFCL / DynDFCL.
Supports EMNIST-Letters and CIFAR-100.

Ablation modes:
1. DynDFCL (Full)             - similarity-driven dynamic coalitions (the proposed method)
2. DynDFCL (FedAvg Agg)       - FedAvg aggregation instead of coalitions
3. DynDFCL (Random Coalition) - randomly formed coalitions (not similarity-driven)
4. DynDFCL (Singleton)        - one coalition per client (pure local training)
5. DynDFCL (Global Union)     - all clients in a single union

Usage:
  python scripts/run_coalition_ablation.py                  # EMNIST (default)
  python scripts/run_coalition_ablation.py --dataset cifar100
"""

import sys
import os
import copy
import logging
import random
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed


def setup_logger(name, log_file=None):
    """Configure a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def run_experiment(config, coalition_mode='full', seed=42, logger=None):
    """
    Run a single experiment.

    Args:
        config:         configuration object
        coalition_mode: one of
            - 'full':      similarity-driven dynamic coalitions (default)
            - 'fedavg':    FedAvg aggregation (no coalitions)
            - 'random':    randomly formed coalitions
            - 'singleton': one coalition per client
            - 'global':    a single union containing all clients
        seed:           random seed
        logger:         optional logger

    Returns:
        dict of metrics.
    """
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Suppress noisy logs.
    logging.getLogger('DCFCL').setLevel(logging.WARNING)
    logging.getLogger('DCFCL.Server').setLevel(logging.WARNING)

    server = DCFCLServer(config, device)
    num_tasks = config.num_tasks
    rounds_per_task = config.rounds_per_task

    # Save the original aggregation hooks.
    original_aggregate_dcfcl = server._aggregate_dcfcl
    original_form_coalition_initial = server._form_coalition_initial
    original_form_coalition_dynamic = server._form_coalition_dynamic

    # Swap in the appropriate aggregation hook for the chosen mode.
    if coalition_mode == 'fedavg':
        # FedAvg aggregation: no coalition, simple global average.
        def fedavg_aggregate(train_results, global_round):
            # Aggregate prototypes (preserve original behaviour).
            if config.algorithm == 'DCFCL':
                server.proto_global = server._aggregate_prototypes(train_results['proto_locals'])
                server.radius_global = server._aggregate_radius(train_results['radius_locals'])

            # FedAvg model aggregation.
            server._zero_model_parameters(server.model)
            total_samples = sum(c.train_samples for c in server.clients)
            for client in server.clients:
                ratio = client.train_samples / total_samples
                server._add_parameters(client.model, ratio)
            server._broadcast_parameters()

        server._aggregate_dcfcl = fedavg_aggregate

    elif coalition_mode == 'random':
        # Random coalitions: split clients randomly into groups of 2-4.
        def random_coalition_initial():
            n = server.num_clients
            clients = list(range(n))
            random.shuffle(clients)

            unions = []
            i = 0
            while i < n:
                group_size = min(random.randint(2, 4), n - i)
                unions.append(tuple(clients[i:i+group_size]))
                i += group_size

            server.unions = tuple(unions)
            if logger:
                logger.info(f"Random coalitions: {server.unions}")

        def random_coalition_dynamic():
            # Re-randomize occasionally during training.
            if random.random() < 0.3:
                random_coalition_initial()

        server._form_coalition_initial = random_coalition_initial
        server._form_coalition_dynamic = random_coalition_dynamic

    elif coalition_mode == 'singleton':
        # Singletons: each client is its own coalition.
        def singleton_coalition():
            server.unions = tuple((i,) for i in range(server.num_clients))
            if logger:
                logger.info(f"Singleton coalitions: {server.unions}")

        server._form_coalition_initial = singleton_coalition
        server._form_coalition_dynamic = singleton_coalition

    elif coalition_mode == 'global':
        # Single global coalition (equivalent to FedAvg but via the coalition path).
        def global_coalition():
            server.unions = (tuple(range(server.num_clients)),)
            if logger:
                logger.info(f"Global coalition: {server.unions}")

        server._form_coalition_initial = global_coalition
        server._form_coalition_dynamic = global_coalition

    # Training loop.
    all_acc = []
    all_accs = []
    all_forget = []
    coalition_history = []

    for task in range(num_tasks):
        if logger:
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {task}")
            logger.info(f"{'='*60}")

        if task > 0:
            server._update_clients_for_new_task(task)
        server._update_available_labels()

        for round_in_task in range(rounds_per_task):
            global_round = task * rounds_per_task + round_in_task

            if config.algorithm != 'Local' and global_round == 0:
                server._broadcast_parameters()

            train_results = server._local_training(global_round, task)
            server._server_aggregation(global_round, task, train_results)

            # Record the coalition structure.
            if hasattr(server, 'unions') and server.unions is not None:
                coalition_history.append({
                    'round': global_round,
                    'task': task,
                    'unions': server.unions
                })

            accs, avg_acc, _ = server._evaluate()
            all_acc.append(avg_acc)

            if round_in_task == rounds_per_task - 1:
                all_accs.append(accs)
                if task > 0:
                    forget_rate = server._compute_forgetting()
                    all_forget.append(forget_rate)
                    if logger:
                        logger.info(f"Task {task} - Acc: {avg_acc:.4f}, Forget: {forget_rate:.4f}")
                else:
                    if logger:
                        logger.info(f"Task {task} - Acc: {avg_acc:.4f}")

    # Aggregate per-task metrics.
    task_end_accs = [all_acc[rounds_per_task * (t + 1) - 1] for t in range(num_tasks)]
    avg_task_acc = sum(task_end_accs) / len(task_end_accs)
    avg_forget = sum(all_forget) / len(all_forget) if all_forget else 0.0

    results = {
        'coalition_mode': coalition_mode,
        'final_accuracy': all_acc[-1],
        'avg_task_accuracy': avg_task_acc,
        'avg_forgetting': avg_forget,
        'task_end_accs': task_end_accs,
        'all_accuracies': all_acc,
        'all_forgetting': all_forget,
        'coalition_history': coalition_history,
    }

    return results


def _build_config_emnist():
    """Base config: EMNIST-Letters (Shuffle split)."""
    config = Config()
    config.dataset = "EMNIST-Letters"
    config.data_split_file = "split_files/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl"
    config.num_users = 8
    config.num_tasks = 6
    config.num_rounds = 60
    config.local_epochs = 100
    config.batch_size = 64
    config.model = "cnn"
    config.lr = 1e-4
    config.weight_decay = 1e-5
    config.seed = 42
    config.rounds_per_task = config.num_rounds // config.num_tasks
    config.algorithm = "DynDFCL"
    config.sw = 0.1
    config.lambda_kd = 0.2
    config.lambda_proto_aug = 2.0
    config.global_weight = 0.9
    config.ema_global = 0.9
    config.dcfcl_broadcast = 1
    config.buffer_size = 500
    config.der_alpha = 0.5
    config.der_beta = 0.5
    return config


def _build_config_cifar100():
    """Base config: CIFAR-100 (random-sample 4x20 split)."""
    config = Config()
    config.dataset = "CIFAR100"
    config.datadir = "./datasets"
    config.data_split_file = "split_files/CIFAR100_split_cn10_tn4_cet20_s2571.pkl"
    config.num_users = 10
    config.num_tasks = 4
    config.num_rounds = 40
    config.local_epochs = 50
    config.batch_size = 64
    config.model = "resnet18"
    config.feature_dim = 512
    config.lr = 0.001
    config.weight_decay = 0.001
    config.seed = 42
    config.rounds_per_task = config.num_rounds // config.num_tasks
    config.algorithm = "DynDFCL"
    config.sw = 0.1
    config.lambda_kd = 0.2
    config.lambda_proto_aug = 0.1
    config.global_weight = 0.9
    config.ema_global = 0.9
    config.dcfcl_broadcast = 1
    config.buffer_size = 500
    config.der_alpha = 0.5
    config.der_beta = 0.5
    config.proto_queue_length = 100
    config._compute_derived()
    return config


def main():
    """Entry point: run the coalition ablation."""
    import argparse
    parser = argparse.ArgumentParser(description="Coalition ablation")
    parser.add_argument("--dataset", choices=["emnist", "cifar100"], default="emnist",
                        help="dataset (emnist | cifar100)")
    args = parser.parse_args()

    use_cifar = args.dataset == "cifar100"
    dataset_name = "CIFAR100" if use_cifar else "EMNIST-Letters"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = "_cifar100" if use_cifar else ""
    result_dir = f"results/coalition_ablation{suffix}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    logger = setup_logger('CoalitionAblation', os.path.join(result_dir, 'experiment.log'))

    logger.info("=" * 80)
    logger.info(f"Coalition aggregation ablation  -  {dataset_name}")
    logger.info("=" * 80)
    logger.info(f"Result directory: {result_dir}")

    config = _build_config_cifar100() if use_cifar else _build_config_emnist()

    logger.info(f"num_classes: {config.num_classes}, rounds_per_task: {config.rounds_per_task}")

    ablation_modes = [
        ('full',      'DynDFCL (full: similarity-driven dynamic coalitions)'),
        ('fedavg',    'DynDFCL (FedAvg aggregation, no coalitions)'),
        ('random',    'DynDFCL (random coalitions)'),
        ('singleton', 'DynDFCL (singleton: one coalition per client)'),
        ('global',    'DynDFCL (global: one union across all clients)'),
    ]

    results = {}

    for mode, description in ablation_modes:
        logger.info(f"\n{'='*60}\nExperiment: {description}\nMode: {mode}\n{'='*60}")

        exp_config = copy.deepcopy(config)
        exp_results = run_experiment(exp_config, coalition_mode=mode, seed=42, logger=logger)
        results[mode] = {'description': description, **exp_results}

        logger.info(f"\nResults:")
        logger.info(f"  Final accuracy:     {exp_results['final_accuracy']*100:.2f}%")
        logger.info(f"  Avg task accuracy:  {exp_results['avg_task_accuracy']*100:.2f}%")
        logger.info(f"  Avg forgetting:     {exp_results['avg_forgetting']*100:.2f}%")
        logger.info(f"  Per-task accuracy:  {[f'{a*100:.1f}%' for a in exp_results['task_end_accs']]}")

    # --- Summary ---
    logger.info("\n" + "=" * 80)
    logger.info("Ablation comparison")
    logger.info("=" * 80)
    logger.info(f"\n{'Mode':<25} {'Final Acc':>12} {'Avg Task Acc':>15} {'Forgetting':>12}")
    logger.info("-" * 70)
    for mode, data in results.items():
        logger.info(f"{mode:<25} {data['final_accuracy']*100:>11.2f}%"
                    f" {data['avg_task_accuracy']*100:>14.2f}%"
                    f" {data['avg_forgetting']*100:>11.2f}%")

    baseline = results['full']
    logger.info("\n" + "-" * 70)
    logger.info("Deltas vs the full configuration")
    logger.info("-" * 70)
    for mode, data in results.items():
        if mode == 'full':
            continue
        acc_diff    = (data['final_accuracy']    - baseline['final_accuracy'])    * 100
        avg_diff    = (data['avg_task_accuracy'] - baseline['avg_task_accuracy']) * 100
        forget_diff = (data['avg_forgetting']    - baseline['avg_forgetting'])    * 100
        logger.info(f"{mode:<25} {acc_diff:>+11.2f}% {avg_diff:>+14.2f}% {forget_diff:>+11.2f}%")

    # --- Save ---
    import json
    for mode in results:
        results[mode]['coalition_history'] = [
            {**h, 'unions': [list(u) for u in h['unions']]}
            for h in results[mode]['coalition_history']
        ]

    json_path = os.path.join(result_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDone. Results saved to: {result_dir}")
    return results


if __name__ == '__main__':
    main()
