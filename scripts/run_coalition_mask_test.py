#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coalition-mask ablation experiment.

Tests the coalition-mask feature and compares performance under:
1. baseline:     no mask (all clients can cooperate)
2. group_mask:   group-based mask (clients in different groups cannot cooperate)
3. random_mask:  randomly forbidden client pairs

Goals:
1. Verify the coalition mask is wired in correctly.
2. Understand how restricting coalitions affects accuracy and forgetting.
"""

import os
import sys
import yaml
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(name: str, config: dict, output_dir: str) -> dict:
    """Run a single experiment."""
    exp_dir = os.path.join(output_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = ['python', 'main.py', '--config', config_path]
    cmd.extend(['--result_dir', exp_dir])

    logger.info(f"Running: {name}")
    logger.info(f"Config: use_coalition_mask={config.get('use_coalition_mask', False)}, "
                f"mask_type={config.get('coalition_mask_type', 'custom')}")

    log_file = os.path.join(exp_dir, 'training.log')
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        final_accuracy = None
        forgetting_rate = None

        for line in process.stdout:
            f.write(line)
            f.flush()

            if 'Final Accuracy:' in line:
                try:
                    final_accuracy = float(line.split('Final Accuracy:')[1].strip())
                except Exception:
                    pass
            if 'Forgetting Rate:' in line:
                try:
                    forgetting_rate = float(line.split('Forgetting Rate:')[1].strip())
                except Exception:
                    pass

        process.wait()

    return {
        'name': name,
        'final_accuracy': final_accuracy,
        'forgetting_rate': forgetting_rate,
        'exit_code': process.returncode
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/coalition_mask_test_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Coalition Mask Ablation")
    logger.info("=" * 70)

    base_config = {
        'dataset': 'EMNIST-Letters',
        'datadir': './datasets',
        'data_split_file': 'split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl',
        'algorithm': 'DynDFCL',
        'num_users': 8,
        'num_tasks': 6,
        'num_rounds': 60,
        'local_epochs': 100,
        'batch_size': 64,
        'lr': 1e-4,
        'buffer_size': 100,  # smaller buffer lets some forgetting occur
        'der_alpha': 0.5,
        'der_beta': 0.5,
        'seed': 42,
    }

    experiments = {
        'baseline': {
            **base_config,
            'use_coalition_mask': False,
        },
        'group_mask_2': {
            **base_config,
            'use_coalition_mask': True,
            'coalition_mask_type': 'group',
            'num_client_groups': 2,  # 8 clients -> 2 groups of 4
        },
        'group_mask_4': {
            **base_config,
            'use_coalition_mask': True,
            'coalition_mask_type': 'group',
            'num_client_groups': 4,  # 8 clients -> 4 groups of 2
        },
        'random_mask_20': {
            **base_config,
            'use_coalition_mask': True,
            'coalition_mask_type': 'random',
            'coalition_mask_density': 0.2,  # 20% pairs forbidden
        },
        'random_mask_50': {
            **base_config,
            'use_coalition_mask': True,
            'coalition_mask_type': 'random',
            'coalition_mask_density': 0.5,  # 50% pairs forbidden
        },
    }

    results = []
    for name, config in experiments.items():
        logger.info("=" * 60)
        logger.info(f"Running: {name}")
        logger.info("=" * 60)

        result = run_experiment(name, config, output_dir)
        results.append(result)

        if result['final_accuracy'] is not None:
            logger.info(f"[ok] {name}: Final Acc = {result['final_accuracy']:.4f}")
        else:
            logger.info(f"[fail] {name}: exit code {result['exit_code']}")

    logger.info("\n" + "=" * 70)
    logger.info("Comparison summary")
    logger.info("=" * 70)

    logger.info("\n{:<25} {:<15} {:<15}".format("Method", "Final Acc", "Forgetting"))
    logger.info("-" * 55)

    baseline_acc = None
    for r in results:
        if r['final_accuracy'] is not None:
            acc_str = f"{r['final_accuracy']:.4f}"
            forget_str = f"{r['forgetting_rate']:.4f}" if r['forgetting_rate'] else "N/A"
            logger.info(f"{r['name']:<25} {acc_str:<15} {forget_str:<15}")

            if r['name'] == 'baseline':
                baseline_acc = r['final_accuracy']

    if baseline_acc is not None:
        logger.info("-" * 55)
        logger.info("\nDeltas vs baseline:")
        for r in results:
            if r['name'] != 'baseline' and r['final_accuracy'] is not None:
                diff = r['final_accuracy'] - baseline_acc
                pct = (diff / baseline_acc) * 100
                sign = '+' if diff >= 0 else ''
                logger.info(f"  {r['name']}: {sign}{diff:.4f} ({sign}{pct:.2f}%)")

    report_path = os.path.join(output_dir, 'COALITION_MASK_REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Coalition Mask Ablation Report\n\n")
        f.write("## Setup\n")
        f.write("- Dataset: EMNIST-Letters\n")
        f.write("- Algorithm: DynDFCL\n")
        f.write("- buffer_size: 100\n")
        f.write("- Number of clients: 8\n")
        f.write("- Number of tasks: 6\n\n")

        f.write("## Results\n\n")
        f.write("| Method | Final Acc | Forgetting | Note |\n")
        f.write("|--------|-----------|------------|------|\n")

        descriptions = {
            'baseline':       'No constraint; all clients can cooperate',
            'group_mask_2':   '2 groups (clients 0-3 vs 4-7)',
            'group_mask_4':   '4 groups (pairs)',
            'random_mask_20': '20% random forbidden pairs',
            'random_mask_50': '50% random forbidden pairs',
        }

        for r in results:
            if r['final_accuracy'] is not None:
                acc = f"{r['final_accuracy']:.4f}"
                forget = f"{r['forgetting_rate']:.4f}" if r['forgetting_rate'] else "N/A"
                desc = descriptions.get(r['name'], '')
                f.write(f"| {r['name']} | {acc} | {forget} | {desc} |\n")

        f.write("\n## Conclusion\n\n")
        if baseline_acc is not None:
            worst_result = min([r for r in results if r['final_accuracy']],
                              key=lambda x: x['final_accuracy'] if x['final_accuracy'] else float('inf'))
            best_result = max([r for r in results if r['final_accuracy']],
                             key=lambda x: x['final_accuracy'] if x['final_accuracy'] else 0)

            f.write(f"- Best: {best_result['name']} ({best_result['final_accuracy']:.4f})\n")
            f.write(f"- Worst: {worst_result['name']} ({worst_result['final_accuracy']:.4f})\n")

            f.write("\n### Impact of the constraint\n\n")
            for r in results:
                if r['name'] != 'baseline' and r['final_accuracy'] is not None:
                    diff = r['final_accuracy'] - baseline_acc
                    if diff < -0.01:
                        f.write(f"- **{r['name']}**: dropped by {abs(diff):.4f}; the constraint blocks useful cooperation.\n")
                    elif diff > 0.01:
                        f.write(f"- **{r['name']}**: gained {diff:.4f}; the constraint may filter out negative transfer.\n")
                    else:
                        f.write(f"- **{r['name']}**: essentially unchanged; constraint has limited effect.\n")

    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
