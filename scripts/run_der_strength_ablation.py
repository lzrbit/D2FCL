#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DER++ strength vs. coalition mechanism ablation.

Hypothesis: a strong DER++ buffer (zero forgetting) can mask the value
of the coalition mechanism. By shrinking buffer_size we let forgetting
re-appear and observe whether the coalition mechanism still provides
extra protection.

Experiment design:
- Dataset:     EMNIST-Letters (Shuffle split, heterogeneous data)
- buffer_size: 0 (DER++ disabled), 50, 100, 200, 500
- For each buffer size we run two aggregation modes: full (dynamic
  coalitions) vs fedavg (global average).
- Expectation: as buffer_size shrinks, forgetting grows and the gap
  between `full` and `fedavg` should widen.
"""

import sys
import os
import copy
import logging
import random
import numpy as np
import torch
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed


def setup_logger(name, log_file=None):
    """Configure a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # drop pre-existing handlers

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
        coalition_mode: 'full' (dynamic coalitions) or 'fedavg' (no coalitions)
        seed:           random seed
        logger:         optional logger
    """
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Suppress noisy logs.
    logging.getLogger('DCFCL').setLevel(logging.WARNING)
    logging.getLogger('DCFCL.Server').setLevel(logging.WARNING)

    server = DCFCLServer(config, device)
    num_tasks = config.num_tasks
    rounds_per_task = config.rounds_per_task

    # Swap in the FedAvg aggregator if requested.
    if coalition_mode == 'fedavg':
        # FedAvg aggregation: no coalition, global average.
        def fedavg_aggregate(train_results, global_round):
            # Aggregate prototypes (preserve original behaviour).
            if config.algorithm == 'DCFCL':
                server.proto_global = server._aggregate_prototypes(train_results['proto_locals'])
                server.radius_global = server._aggregate_radius(train_results['radius_locals'])

            # FedAvg.
            server._zero_model_parameters(server.model)
            total_samples = sum(c.train_samples for c in server.clients)
            for client in server.clients:
                ratio = client.train_samples / total_samples
                server._add_parameters(client.model, ratio)
            server._broadcast_parameters()

        server._aggregate_dcfcl = fedavg_aggregate

    # Training loop.
    all_acc = []
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
            if coalition_mode == 'full' and hasattr(server, 'unions') and server.unions is not None:
                coalition_history.append({
                    'round': global_round,
                    'task': task,
                    'unions': [list(u) for u in server.unions]
                })

            accs, avg_acc, _ = server._evaluate()
            all_acc.append(avg_acc)

            if round_in_task == rounds_per_task - 1:
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

    return {
        'coalition_mode': coalition_mode,
        'final_accuracy': all_acc[-1],
        'avg_task_accuracy': avg_task_acc,
        'avg_forgetting': avg_forget,
        'task_end_accs': task_end_accs,
        'all_accuracies': all_acc,
        'all_forgetting': all_forget,
        'coalition_history': coalition_history,
    }


def main():
    """Entry point: run the DER++ strength ablation."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"results/der_strength_ablation_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    logger = setup_logger('DERStrengthAblation', os.path.join(result_dir, 'experiment.log'))

    logger.info("=" * 80)
    logger.info("DER++ strength vs. coalition mechanism ablation")
    logger.info("=" * 80)

    # buffer_size=0 means DER++ disabled (we set der_alpha=der_beta=0).
    buffer_sizes = [0, 50, 100, 200, 500]
    seed = 42

    # Base configuration (EMNIST-Letters, Shuffle split).
    base_config = Config()
    base_config.dataset = "EMNIST-Letters"
    base_config.data_split_file = "split_files/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl"
    base_config.num_users = 8
    base_config.num_tasks = 6
    base_config.num_rounds = 60
    base_config.local_epochs = 100
    base_config.batch_size = 64
    base_config.model = "cnn"
    base_config.lr = 1e-4
    base_config.weight_decay = 1e-5
    base_config.seed = seed
    base_config.rounds_per_task = base_config.num_rounds // base_config.num_tasks

    # DynDFCL parameters.
    base_config.algorithm = "DynDFCL"
    base_config.sw = 0.1
    base_config.lambda_kd = 0.2
    base_config.lambda_proto_aug = 2.0
    base_config.global_weight = 0.9
    base_config.ema_global = 0.9
    base_config.dcfcl_broadcast = 1

    all_results = {}

    for buffer_size in buffer_sizes:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Buffer Size: {buffer_size}")
        logger.info(f"{'#'*80}")

        all_results[f'buf_{buffer_size}'] = {}

        for mode in ['full', 'fedavg']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Coalition Mode: {mode}")
            logger.info(f"{'='*60}")

            config = copy.deepcopy(base_config)

            # buffer_size=0: keep buffer_size > 0 to avoid runtime errors,
            # and zero out der_alpha/der_beta to effectively disable DER++.
            if buffer_size == 0:
                config.buffer_size = 10
                config.der_alpha = 0.0
                config.der_beta = 0.0
            else:
                config.buffer_size = buffer_size
                config.der_alpha = 0.5
                config.der_beta = 0.5

            results = run_experiment(config, coalition_mode=mode, seed=seed, logger=logger)
            all_results[f'buf_{buffer_size}'][mode] = results

            logger.info(f"\nResults (buffer_size={buffer_size}, mode={mode}):")
            logger.info(f"  Final accuracy:    {results['final_accuracy']*100:.2f}%")
            logger.info(f"  Avg task accuracy: {results['avg_task_accuracy']*100:.2f}%")
            logger.info(f"  Avg forgetting:    {results['avg_forgetting']*100:.2f}%")
            logger.info(f"  Per-task accuracy: {[f'{a*100:.1f}%' for a in results['task_end_accs']]}")

    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    generate_report(all_results, result_dir, logger)

    logger.info(f"\nDone. Results saved to: {result_dir}")


def generate_report(results, result_dir, logger):
    """Write the ablation report."""
    report = []
    report.append("# DER++ Strength vs. Coalition Mechanism Ablation Report\n\n")
    report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("**Dataset**: EMNIST-Letters (Shuffle split, heterogeneous data)\n\n")

    report.append("## Hypothesis\n\n")
    report.append("A strong DER++ buffer (zero forgetting) can mask the value of the coalition mechanism.\n")
    report.append("By shrinking buffer_size we let forgetting re-appear and check whether the\n")
    report.append("coalition mechanism still provides extra protection.\n\n")

    report.append("## Results\n\n")
    report.append("| Buffer Size | Mode | Final Acc | Avg Task Acc | Avg Forgetting |\n")
    report.append("|-------------|------|-----------|--------------|----------------|\n")

    for buf_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        buffer_size = buf_key.split('_')[1]
        for mode in ['full', 'fedavg']:
            if mode in results[buf_key]:
                r = results[buf_key][mode]
                report.append(f"| {buffer_size} | {mode} | {r['final_accuracy']*100:.2f}% | {r['avg_task_accuracy']*100:.2f}% | {r['avg_forgetting']*100:.2f}% |\n")

    report.append("\n## Coalition effect analysis (full - fedavg)\n\n")
    report.append("| Buffer Size | Acc delta | Forgetting delta | Coalition helpful? |\n")
    report.append("|-------------|-----------|------------------|--------------------|\n")

    for buf_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        buffer_size = buf_key.split('_')[1]
        if 'full' in results[buf_key] and 'fedavg' in results[buf_key]:
            full = results[buf_key]['full']
            fedavg = results[buf_key]['fedavg']

            acc_diff = full['final_accuracy'] - fedavg['final_accuracy']
            forget_diff = full['avg_forgetting'] - fedavg['avg_forgetting']

            # Coalition is helpful if accuracy is higher or forgetting lower.
            effective = "Yes" if acc_diff > 0.005 or forget_diff < -0.005 else "No"

            report.append(f"| {buffer_size} | {acc_diff*100:+.2f}% | {forget_diff*100:+.2f}% | {effective} |\n")

    report.append("\n## Conclusion\n\n")
    report.append("(Auto-generated from the results above.)\n\n")

    # Trend check.
    buf_sizes = sorted([int(k.split('_')[1]) for k in results.keys()])
    acc_diffs = []
    forget_diffs = []

    for buf_size in buf_sizes:
        buf_key = f'buf_{buf_size}'
        if 'full' in results[buf_key] and 'fedavg' in results[buf_key]:
            acc_diff = results[buf_key]['full']['final_accuracy'] - results[buf_key]['fedavg']['final_accuracy']
            forget_diff = results[buf_key]['full']['avg_forgetting'] - results[buf_key]['fedavg']['avg_forgetting']
            acc_diffs.append(acc_diff)
            forget_diffs.append(forget_diff)

    if acc_diffs:
        if acc_diffs[0] > acc_diffs[-1]:
            report.append("- **Hypothesis supported**: as buffer_size shrinks (DER++ weakens), the coalition advantage grows.\n")
        else:
            report.append("- **Hypothesis not supported**: the coalition advantage does not grow as DER++ weakens.\n")

        report.append(f"- Smallest buffer (buf={buf_sizes[0]}): full vs fedavg = {acc_diffs[0]*100:+.2f}%\n")
        report.append(f"- Largest buffer  (buf={buf_sizes[-1]}): full vs fedavg = {acc_diffs[-1]*100:+.2f}%\n")

    with open(os.path.join(result_dir, 'ABLATION_REPORT.md'), 'w') as f:
        f.writelines(report)

    logger.info("\n" + "="*60)
    logger.info("Report written: ABLATION_REPORT.md")
    logger.info("="*60)


if __name__ == '__main__':
    main()
