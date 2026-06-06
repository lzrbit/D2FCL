#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation of ALL algorithms on EMNIST-Letters and CIFAR100.

Usage:
    python run_all_algorithms.py --debug                          # Quick sanity check
    python run_all_algorithms.py --datasets EMNIST-Letters        # Full EMNIST only
    python run_all_algorithms.py --datasets CIFAR100              # Full CIFAR100 only
    python run_all_algorithms.py                                   # Full both datasets
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed, setup_logging


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset configurations
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    'EMNIST-Letters': {
        'dataset': 'EMNIST-Letters',
        'data_split_file': 'split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl',
        'num_users': 8, 'num_tasks': 6, 'num_rounds': 60,
        'local_epochs': 100, 'batch_size': 64,
        'model': 'cnn', 'lr': 1e-4, 'weight_decay': 1e-5, 'sw': 0.1,
    },
    'CIFAR100': {
        'dataset': 'CIFAR100',
        'data_split_file': 'split_files/CIFAR100_split_cn10_tn10_cet10_cs1_s2571.pkl',
        'num_users': 10, 'num_tasks': 10, 'num_rounds': 100,
        'local_epochs': 50, 'batch_size': 64,
        'model': 'resnet18', 'lr': 1e-3, 'weight_decay': 1e-3, 'sw': 0.1,
    },
}

# Per-dataset debug rounds (1 round per task to avoid division-by-zero)
DEBUG_ROUNDS = {
    'EMNIST-Letters': {'num_rounds': 6, 'num_tasks': 6},
    'CIFAR100':       {'num_rounds': 10, 'num_tasks': 10},
}

# All runnable algorithms (L2C excluded: _aggregate_l2c not implemented)
BASELINE_ALGORITHMS = ['FedAvg', 'FedProx', 'FedLwF', 'Local', 'SCAFFOLD', 'PerAvg', 'pFedMe', 'ClusterFL']

# DCFCL lambda_kd sweep
DCFCL_LAMBDA_KD_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def resolve_device(device_str):
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def run_single_experiment(dataset_name, algorithm, extra_cfg=None, debug=False,
                          device='auto', seed=42):
    """Run one experiment. Returns result dict or None on error."""
    ds_cfg = copy.deepcopy(DATASET_CONFIGS[dataset_name])

    if debug:
        ds_cfg['local_epochs'] = 2
        if dataset_name in DEBUG_ROUNDS:
            ds_cfg.update(DEBUG_ROUNDS[dataset_name])

    if extra_cfg:
        ds_cfg.update(extra_cfg)

    config = Config(algorithm=algorithm, seed=seed, device=device, **ds_cfg)
    setup_seed(config.seed)
    dev = resolve_device(config.device)

    label = algorithm
    if algorithm == 'DCFCL' and extra_cfg and 'lambda_kd' in extra_cfg:
        label = f"DCFCL(λ_kd={extra_cfg['lambda_kd']})"

    mode_str = f"debug({config.num_rounds}r×{config.local_epochs}e)" if debug \
        else f"{config.num_rounds}r×{config.local_epochs}e"

    print(f"\n{'='*70}")
    print(f"  {dataset_name} | {label} | {mode_str} | {dev}")
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    try:
        server = DCFCLServer(config, dev)
        results = server.train()
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  !! ERROR after {elapsed:.0f}s: {e}", flush=True)
        traceback.print_exc()
        return None

    elapsed = time.time() - t0
    final_acc = results.get('final_accuracy', 0.0)
    forgetting = results.get('forgetting_rate', 0.0)
    all_acc = results.get('all_accuracies', [])

    # per-task accuracies at task boundary
    rounds_per_task = config.num_rounds // config.num_tasks
    task_accuracies = {}
    for t in range(config.num_tasks):
        idx = (t + 1) * rounds_per_task - 1
        if idx < len(all_acc):
            task_accuracies[t] = all_acc[idx]

    # Average accuracy across all task boundaries (Avg Acc)
    if task_accuracies:
        avg_accuracy = float(np.mean(list(task_accuracies.values())))
    else:
        avg_accuracy = final_acc

    print(f"  => {label}: final={final_acc*100:.1f}%, avg={avg_accuracy*100:.1f}%, forget={forgetting*100:.1f}%, time={elapsed:.0f}s",
          flush=True)

    return {
        'dataset': dataset_name,
        'algorithm': algorithm,
        'label': label,
        'final_accuracy': final_acc,
        'avg_accuracy': avg_accuracy,
        'forgetting_rate': forgetting,
        'time_seconds': elapsed,
        'all_accuracies': [float(a) for a in all_acc],
        'task_accuracies': {str(k): float(v) for k, v in task_accuracies.items()},
        'extra_cfg': extra_cfg or {},
        'config': {
            'num_rounds': config.num_rounds, 'num_tasks': config.num_tasks,
            'local_epochs': config.local_epochs, 'lr': config.lr,
            'weight_decay': config.weight_decay, 'model': config.model, 'seed': seed,
        },
    }


def run_dataset_experiments(dataset_name, debug=False, device='auto', seed=42):
    """Run ALL algorithms on one dataset. Returns dict of results."""
    results = {'baselines': [], 'dcfcl_results': [], 'errors': []}

    # 1) Baseline algorithms
    for algo in BASELINE_ALGORITHMS:
        print(f"\n{'━'*70}")
        print(f"  Running {algo} on {dataset_name}")
        print(f"{'━'*70}", flush=True)

        r = run_single_experiment(dataset_name, algo, debug=debug,
                                  device=device, seed=seed)
        if r:
            results['baselines'].append(r)
        else:
            results['errors'].append({'algorithm': algo, 'error': 'failed'})

    # 2) DCFCL with different lambda_kd
    for lkd in DCFCL_LAMBDA_KD_VALUES:
        print(f"\n{'━'*70}")
        print(f"  Running DCFCL(λ_kd={lkd}) on {dataset_name}")
        print(f"{'━'*70}", flush=True)

        r = run_single_experiment(dataset_name, 'DCFCL',
                                  extra_cfg={'lambda_kd': lkd},
                                  debug=debug, device=device, seed=seed)
        if r:
            results['dcfcl_results'].append(r)
        else:
            results['errors'].append({'algorithm': f'DCFCL(λ_kd={lkd})', 'error': 'failed'})

    return results


def generate_report(all_results, output_path):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# Comprehensive Algorithm Evaluation Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ─── Overview ───
    lines.append("## 1. Experiment Setup")
    lines.append("")
    lines.append("### Datasets")
    lines.append("")
    lines.append("| Dataset | Clients | Tasks | Classes/Task | Total Classes | Model | Rounds | Epochs | LR | WD |")
    lines.append("|---------|---------|-------|-------------|---------------|-------|--------|--------|-----|-----|")
    for ds_name in all_results:
        cfg = DATASET_CONFIGS[ds_name]
        total_cls = {'EMNIST-Letters': 26, 'CIFAR100': 100}.get(ds_name, '?')
        cpt = {'EMNIST-Letters': 2, 'CIFAR100': 10}.get(ds_name, '?')
        lines.append(f"| {ds_name} | {cfg['num_users']} | {cfg['num_tasks']} | "
                     f"{cpt} | {total_cls} | {cfg['model']} | {cfg['num_rounds']} | "
                     f"{cfg['local_epochs']} | {cfg['lr']} | {cfg['weight_decay']} |")
    lines.append("")

    lines.append("### Algorithms")
    lines.append("")
    lines.append("| Algorithm | Type | Key Feature |")
    lines.append("|-----------|------|-------------|")
    lines.append("| FedAvg | Centralized | Simple weighted averaging |")
    lines.append("| FedProx | Centralized | Proximal term (μ=0.005) |")
    lines.append("| FedLwF | Centralized | Knowledge distillation (α=1.0) |")
    lines.append("| Local | Decentralized | No aggregation, local training only |")
    lines.append("| SCAFFOLD | Centralized | Control variates for variance reduction |")
    lines.append("| PerAvg | Personalized | Two-step personalized averaging |")
    lines.append("| pFedMe | Personalized | Moreau envelope personalization |")
    lines.append("| ClusterFL | Clustered | Gradient-similarity clustering |")
    lines.append("| **DCFCL** | **Coalition** | **Dynamic coalition formation + prototypes** |")
    lines.append("")

    # ─── Per-dataset results ───
    for ds_name, ds_results in all_results.items():
        baselines = ds_results.get('baselines', [])
        dcfcl_results = ds_results.get('dcfcl_results', [])
        errors = ds_results.get('errors', [])

        lines.append(f"## 2. Results: {ds_name}")
        lines.append("")

        # Main results table
        lines.append("### Main Comparison Table")
        lines.append("")
        lines.append("| Algorithm | Final Acc (%) | Avg Acc (%) | Forgetting (%) | Time (s) |")
        lines.append("|-----------|:------------------:|:------------------:|:-------------------:|:--------:|")

        all_entries = []
        for r in baselines:
            all_entries.append((r['label'], r['final_accuracy'], r.get('avg_accuracy', r['final_accuracy']), r['forgetting_rate'], r['time_seconds']))
        for r in dcfcl_results:
            all_entries.append((r['label'], r['final_accuracy'], r.get('avg_accuracy', r['final_accuracy']), r['forgetting_rate'], r['time_seconds']))

        # Sort by avg accuracy descending
        all_entries.sort(key=lambda x: x[2], reverse=True)

        best_avg = all_entries[0][2] if all_entries else 0
        for label, acc, avg_acc, fgt, t in all_entries:
            marker = " **★**" if avg_acc == best_avg else ""
            lines.append(f"| {label}{marker} | {acc*100:.1f} | {avg_acc*100:.1f} | {fgt*100:.1f} | {t:.0f} |")
        lines.append("")

        if errors:
            lines.append("**Errors:**")
            for e in errors:
                lines.append(f"- {e['algorithm']}: {e['error']}")
            lines.append("")

        # DCFCL lambda_kd analysis
        if dcfcl_results:
            lines.append("### DCFCL: Impact of λ_kd")
            lines.append("")
            lines.append("| λ_kd | Final Acc (%) | Avg Acc (%) | Forgetting (%) |")
            lines.append("|:----:|:------------:|:------------:|:--------------:|")

            base_avg = None
            for r in dcfcl_results:
                lkd = r['extra_cfg'].get('lambda_kd', 0)
                acc = r['final_accuracy']
                avg_acc = r.get('avg_accuracy', acc)
                fgt = r['forgetting_rate']
                if base_avg is None:
                    base_avg = avg_acc
                lines.append(f"| {lkd} | {acc*100:.1f} | {avg_acc*100:.1f} | {fgt*100:.1f} |")
            lines.append("")

        # Per-task accuracy progression for key algorithms
        lines.append("### Per-Task Accuracy Progression")
        lines.append("")

        # Collect key algorithms to show
        key_algos = []
        for r in baselines:
            if r['algorithm'] in ['FedAvg', 'FedProx', 'FedLwF', 'SCAFFOLD', 'Local']:
                key_algos.append(r)
        # Add best DCFCL
        if dcfcl_results:
            best_dcfcl = max(dcfcl_results, key=lambda x: x['final_accuracy'])
            key_algos.append(best_dcfcl)

        if key_algos:
            num_tasks = DATASET_CONFIGS[ds_name]['num_tasks']
            header = "| Algorithm |"
            sep = "|-----------|"
            for t in range(num_tasks):
                header += f" Task {t} |"
                sep += ":------:|"
            lines.append(header)
            lines.append(sep)

            for r in key_algos:
                row = f"| {r['label']} |"
                for t in range(num_tasks):
                    val = r['task_accuracies'].get(str(t), None)
                    if val is not None:
                        row += f" {val*100:.1f} |"
                    else:
                        row += " - |"
                lines.append(row)
            lines.append("")

        # Analysis
        lines.append("### Analysis")
        lines.append("")

        if baselines and dcfcl_results:
            best_baseline = max(baselines, key=lambda x: x.get('avg_accuracy', x['final_accuracy']))
            best_dcfcl_r = max(dcfcl_results, key=lambda x: x.get('avg_accuracy', x['final_accuracy']))
            baseline_acc = best_baseline.get('avg_accuracy', best_baseline['final_accuracy'])
            dcfcl_acc = best_dcfcl_r.get('avg_accuracy', best_dcfcl_r['final_accuracy'])
            gap = (dcfcl_acc - baseline_acc) * 100

            lines.append(f"- **Best baseline**: {best_baseline['label']} "
                        f"({baseline_acc*100:.1f}%)")
            lines.append(f"- **Best DCFCL**: {best_dcfcl_r['label']} "
                        f"({dcfcl_acc*100:.1f}%)")
            if gap > 0:
                lines.append(f"- **DCFCL advantage**: +{gap:.1f}% over best baseline")
            else:
                lines.append(f"- **DCFCL gap**: {gap:.1f}% vs best baseline")

            # Compare forgetting
            best_fgt_baseline = min(baselines, key=lambda x: x['forgetting_rate'])
            lines.append(f"- **Lowest forgetting (baseline)**: {best_fgt_baseline['label']} "
                        f"({best_fgt_baseline['forgetting_rate']*100:.1f}%)")
            lines.append(f"- **DCFCL forgetting**: "
                        f"{best_dcfcl_r['forgetting_rate']*100:.1f}%")

            # lambda_kd insight
            best_lkd = best_dcfcl_r['extra_cfg'].get('lambda_kd', 0)
            lines.append(f"- **Optimal λ_kd**: {best_lkd}")
            lines.append("")

    # ─── Paper comparison ───
    lines.append("## 3. Comparison with Paper Results")
    lines.append("")
    lines.append("### EMNIST-Letters (Paper Table 1, LTP setting)")
    lines.append("")
    lines.append("| Algorithm | Paper Accuracy | Our Accuracy | Δ |")
    lines.append("|-----------|:--------------:|:------------:|:---:|")

    paper_emnist = {
        'FedAvg': 40.4, 'FedProx': 39.3, 'SCAFFOLD': 35.6,
        'PerAvg': 38.2, 'pFedMe': 38.1, 'FedLwF': 48.3, 'Local': 17.1,
        'ClusterFL': 39.2, 'DCFCL': 52.5,
    }
    if 'EMNIST-Letters' in all_results:
        emnist = all_results['EMNIST-Letters']
        our_results = {}
        for r in emnist.get('baselines', []):
            our_results[r['algorithm']] = r.get('avg_accuracy', r['final_accuracy']) * 100
        if emnist.get('dcfcl_results'):
            best_dcfcl_r = max(emnist['dcfcl_results'], key=lambda x: x.get('avg_accuracy', x['final_accuracy']))
            our_results['DCFCL'] = best_dcfcl_r.get('avg_accuracy', best_dcfcl_r['final_accuracy']) * 100

        for algo, paper_val in paper_emnist.items():
            if algo in our_results:
                ours = our_results[algo]
                delta = ours - paper_val
                sign = "+" if delta >= 0 else ""
                lines.append(f"| {algo} | {paper_val:.1f} | {ours:.1f} | {sign}{delta:.1f} |")
            else:
                lines.append(f"| {algo} | {paper_val:.1f} | - | - |")
        lines.append("")

    lines.append("### CIFAR-100 (Paper Table 1, LTP setting)")
    lines.append("")
    lines.append("| Algorithm | Paper Accuracy | Our Accuracy | Δ |")
    lines.append("|-----------|:--------------:|:------------:|:---:|")

    paper_cifar = {
        'FedAvg': 12.3, 'FedProx': 12.6, 'SCAFFOLD': 9.3,
        'PerAvg': 13.1, 'pFedMe': 12.1, 'FedLwF': 12.7, 'Local': 4.1,
        'ClusterFL': 12.5, 'DCFCL': 15.8,
    }
    if 'CIFAR100' in all_results:
        cifar = all_results['CIFAR100']
        our_results = {}
        for r in cifar.get('baselines', []):
            our_results[r['algorithm']] = r.get('avg_accuracy', r['final_accuracy']) * 100
        if cifar.get('dcfcl_results'):
            best_dcfcl_r = max(cifar['dcfcl_results'], key=lambda x: x.get('avg_accuracy', x['final_accuracy']))
            our_results['DCFCL'] = best_dcfcl_r.get('avg_accuracy', best_dcfcl_r['final_accuracy']) * 100

        for algo, paper_val in paper_cifar.items():
            if algo in our_results:
                ours = our_results[algo]
                delta = ours - paper_val
                sign = "+" if delta >= 0 else ""
                lines.append(f"| {algo} | {paper_val:.1f} | {ours:.1f} | {sign}{delta:.1f} |")
            else:
                lines.append(f"| {algo} | {paper_val:.1f} | - | - |")
        lines.append("")

    # ─── Summary ───
    lines.append("## 4. Summary")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    for ds_name, ds_results in all_results.items():
        baselines = ds_results.get('baselines', [])
        dcfcl_results = ds_results.get('dcfcl_results', [])
        if baselines and dcfcl_results:
            best_bl = max(baselines, key=lambda x: x.get('avg_accuracy', x['final_accuracy']))
            best_dc = max(dcfcl_results, key=lambda x: x.get('avg_accuracy', x['final_accuracy']))
            best_bl_acc = best_bl.get('avg_accuracy', best_bl['final_accuracy'])
            best_dc_acc = best_dc.get('avg_accuracy', best_dc['final_accuracy'])
            lines.append(f"**{ds_name}:**")
            lines.append(f"- Best baseline: {best_bl['label']} = {best_bl_acc*100:.1f}%")
            lines.append(f"- Best DCFCL: {best_dc['label']} = {best_dc_acc*100:.1f}%")
            lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by run_all_algorithms.py*")

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='Comprehensive FL algorithm evaluation')
    parser.add_argument('--debug', action='store_true', help='Quick debug mode')
    parser.add_argument('--datasets', nargs='+',
                        default=['EMNIST-Letters', 'CIFAR100'],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--output', type=str, default=None,
                        help='Output MD file path')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    mode = "DEBUG" if args.debug else "FULL"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.output is None:
        args.output = f"results/all_algorithms_{'debug' if args.debug else 'full'}_{timestamp}.md"

    json_path = args.output.replace('.md', '.json')

    print(f"╔{'═'*68}╗")
    print(f"║  All-Algorithm Evaluation — {mode} MODE{' '*(36-len(mode))}║")
    print(f"║  Datasets: {', '.join(args.datasets):<56}║")
    print(f"║  Baselines: {len(BASELINE_ALGORITHMS)} | DCFCL λ_kd: {len(DCFCL_LAMBDA_KD_VALUES)} values{' '*23}║")
    print(f"╚{'═'*68}╝", flush=True)

    all_results = {}
    total_t0 = time.time()

    for ds_name in args.datasets:
        print(f"\n{'━'*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'━'*70}", flush=True)

        ds_results = run_dataset_experiments(ds_name, debug=args.debug,
                                              device=args.device, seed=args.seed)
        all_results[ds_name] = ds_results

        # Save intermediate JSON
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    total_elapsed = time.time() - total_t0

    # Generate report
    generate_report(all_results, args.output)

    # Final summary
    print(f"\n{'═'*70}")
    print(f"  COMPLETED (total: {total_elapsed:.0f}s)")
    print(f"{'═'*70}")
    for ds_name, ds_results in all_results.items():
        baselines = ds_results.get('baselines', [])
        dcfcl_results = ds_results.get('dcfcl_results', [])
        print(f"\n  {ds_name}:")
        for r in sorted(baselines, key=lambda x: -x['final_accuracy']):
            print(f"    {r['label']:<15} {r['final_accuracy']*100:>5.1f}%  (forget={r['forgetting_rate']*100:.1f}%)")
        for r in sorted(dcfcl_results, key=lambda x: -x['final_accuracy']):
            print(f"    {r['label']:<15} {r['final_accuracy']*100:>5.1f}%  (forget={r['forgetting_rate']*100:.1f}%)")

    print(f"\n  JSON:   {json_path}")
    print(f"  Report: {args.output}", flush=True)


if __name__ == '__main__':
    main()
