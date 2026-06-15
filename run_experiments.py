#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive experiment runner for DCFCL project.

Part 1: Run all algorithms on EMNIST-Letters and CIFAR100.
Part 2: Ablation study for D2FCL.

Results are saved to results/ and summarized in EXPERIMENT_RESULTS.md.
"""

import subprocess
import sys
import json
import os
import time
from datetime import datetime

PYTHON = sys.executable
MAIN = "main.py"

# ============================================================
# Dataset configurations
# ============================================================
EMNIST_BASE = dict(
    dataset="EMNIST-Letters",
    data_split_file="split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl",
    num_users=8, num_tasks=6, num_rounds=60,
    local_epochs=100, batch_size=64,
    model="cnn", lr=1e-4, weight_decay=1e-5,
    device="cuda", seed=42,
    # DCFCL / D2FCL defaults
    sw=0.1, lambda_kd=0.2, lambda_proto_aug=2.0,
    global_weight=0.9, ema_global=0.9,
    dcfcl_broadcast=1,
    buffer_size=500, der_alpha=0.5, der_beta=0.5,
)

CIFAR100_BASE = dict(
    dataset="CIFAR100",
    data_split_file="split_files/CIFAR100_split_cn10_tn10_cet10_cs1_s2571.pkl",
    num_users=10, num_tasks=10, num_rounds=100,
    local_epochs=50, batch_size=64,
    model="resnet18", lr=1e-3, weight_decay=1e-3,
    device="cuda", seed=42,
    sw=0.1, lambda_kd=0.2, lambda_proto_aug=2.0,
    global_weight=0.9, ema_global=0.9,
    dcfcl_broadcast=1,
    buffer_size=500, der_alpha=0.5, der_beta=0.5,
)

ALL_ALGORITHMS = [
    'DCFCL', 'D2FCL', 'FedAvg', 'FedProx', 'FedLwF',
    'Local', 'SCAFFOLD', 'PerAvg', 'pFedMe', 'ClusterFL', 'L2C',
]

# ============================================================
# Ablation configurations for D2FCL on EMNIST-Letters
# ============================================================
ABLATION_CONFIGS = {
    # baseline (full D2FCL)
    "full":            dict(lambda_kd=0.2, lambda_proto_aug=2.0, der_alpha=0.5, der_beta=0.5, buffer_size=500),
    # remove DER (alpha=0, beta=0)  →  degenerates to DCFCL
    "no_DER":          dict(lambda_kd=0.2, lambda_proto_aug=2.0, der_alpha=0.0, der_beta=0.0, buffer_size=500),
    # remove DER++ CE only (beta=0)  →  DER without replay CE
    "no_DER_CE":       dict(lambda_kd=0.2, lambda_proto_aug=2.0, der_alpha=0.5, der_beta=0.0, buffer_size=500),
    # remove DER logit only (alpha=0) → only replay CE
    "no_DER_logit":    dict(lambda_kd=0.2, lambda_proto_aug=2.0, der_alpha=0.0, der_beta=0.5, buffer_size=500),
    # remove KD
    "no_KD":           dict(lambda_kd=0.0, lambda_proto_aug=2.0, der_alpha=0.5, der_beta=0.5, buffer_size=500),
    # remove proto_aug
    "no_ProtoAug":     dict(lambda_kd=0.2, lambda_proto_aug=0.0, der_alpha=0.5, der_beta=0.5, buffer_size=500),
    # remove both KD + proto_aug (only DER++)
    "only_DER":        dict(lambda_kd=0.0, lambda_proto_aug=0.0, der_alpha=0.5, der_beta=0.5, buffer_size=500),
    # buffer size 200
    "buf200":          dict(lambda_kd=0.2, lambda_proto_aug=2.0, der_alpha=0.5, der_beta=0.5, buffer_size=200),
    # buffer size 1000
    "buf1000":         dict(lambda_kd=0.2, lambda_proto_aug=2.0, der_alpha=0.5, der_beta=0.5, buffer_size=1000),
}


def build_cmd(base_config, algorithm, **overrides):
    """Build command line from config dict."""
    cfg = {**base_config, **overrides, "algorithm": algorithm}
    cmd = [PYTHON, MAIN]
    for k, v in cfg.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def run_one(name, cmd, log_path=None):
    """Run a single experiment, return parsed results or None on failure."""
    print(f"\n{'='*60}")
    print(f"[START] {name}")
    print(f"  cmd: {' '.join(cmd[-8:])}")  # show last few args
    print(f"{'='*60}")
    t0 = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"[FAIL] {name} (exit={result.returncode}, {elapsed:.0f}s)")
            print(result.stderr[-500:] if result.stderr else "no stderr")
            return None
        print(f"[DONE] {name} ({elapsed:.0f}s)")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name}")
        return None

    # Find the latest results.json
    results_dir = "results"
    prefix = f"{algorithm_from_cmd(cmd)}_{dataset_from_cmd(cmd)}_"
    candidates = sorted(
        [d for d in os.listdir(results_dir) if d.startswith(prefix)],
        reverse=True,
    )
    if not candidates:
        print(f"[WARN] No result dir found for {name}")
        return None

    json_path = os.path.join(results_dir, candidates[0], "results.json")
    if not os.path.exists(json_path):
        print(f"[WARN] No results.json in {candidates[0]}")
        return None

    with open(json_path) as f:
        data = json.load(f)

    # Compute avg_task_accuracy from all_accuracies
    all_acc = data.get("all_accuracies", [])
    num_tasks = int(cfg_val(cmd, "--num_tasks", 6))
    rpt = len(all_acc) // num_tasks if num_tasks > 0 else 0
    if rpt > 0:
        phase_end = [all_acc[rpt * (t + 1) - 1] for t in range(num_tasks)]
        avg_task = sum(phase_end) / len(phase_end)
    else:
        phase_end = []
        avg_task = data.get("final_accuracy", 0)

    return {
        "name": name,
        "final_accuracy": data.get("final_accuracy", 0),
        "forgetting_rate": data.get("forgetting_rate", 0),
        "avg_task_accuracy": avg_task,
        "phase_end_accs": phase_end,
        "elapsed_s": elapsed,
        "result_dir": candidates[0],
    }


def algorithm_from_cmd(cmd):
    for i, a in enumerate(cmd):
        if a == "--algorithm" and i + 1 < len(cmd):
            return cmd[i + 1]
    return "unknown"

def dataset_from_cmd(cmd):
    for i, a in enumerate(cmd):
        if a == "--dataset" and i + 1 < len(cmd):
            return cmd[i + 1]
    return "unknown"

def cfg_val(cmd, key, default):
    for i, a in enumerate(cmd):
        if a == key and i + 1 < len(cmd):
            return cmd[i + 1]
    return default


def write_markdown(all_results, ablation_results, md_path="EXPERIMENT_RESULTS.md"):
    """Write results summary markdown."""
    lines = []
    lines.append("# DCFCL Experiment Results\n")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # ---- Part 1: All algorithms ----
    for ds_name, results in all_results.items():
        lines.append(f"\n## {ds_name} — All Algorithms\n")
        lines.append("| Algorithm | Avg Task Acc (%) | Final Acc (%) | Avg Forgetting (%) | Time (s) |")
        lines.append("|-----------|:----------------:|:-------------:|:------------------:|:--------:|")
        for r in sorted(results, key=lambda x: -x["avg_task_accuracy"]):
            lines.append(
                f"| {r['name']} | {r['avg_task_accuracy']*100:.2f} | "
                f"{r['final_accuracy']*100:.2f} | "
                f"{r['forgetting_rate']*100:.2f} | "
                f"{r['elapsed_s']:.0f} |"
            )
        lines.append("")

        # Phase-end table
        if results and results[0]["phase_end_accs"]:
            num_tasks = len(results[0]["phase_end_accs"])
            hdr = "| Algorithm | " + " | ".join(f"Task {t}" for t in range(num_tasks)) + " |"
            sep = "|-----------|" + "|".join(":------:" for _ in range(num_tasks)) + "|"
            lines.append(f"### {ds_name} — Phase-End Accuracy by Task\n")
            lines.append(hdr)
            lines.append(sep)
            for r in sorted(results, key=lambda x: -x["avg_task_accuracy"]):
                cols = " | ".join(f"{a*100:.2f}" for a in r["phase_end_accs"])
                lines.append(f"| {r['name']} | {cols} |")
            lines.append("")

    # ---- Part 2: Ablation ----
    if ablation_results:
        lines.append("\n## D2FCL Ablation Study (EMNIST-Letters)\n")
        lines.append("| Variant | Avg Task Acc (%) | Final Acc (%) | Avg Forgetting (%) | Buffer | α_der | β_der | λ_kd | λ_pa |")
        lines.append("|---------|:----------------:|:-------------:|:------------------:|:------:|:-----:|:-----:|:----:|:----:|")
        for tag, r in ablation_results.items():
            if r is None:
                lines.append(f"| {tag} | FAILED | - | - | - | - | - | - | - |")
                continue
            cfg = ABLATION_CONFIGS[tag]
            lines.append(
                f"| {tag} | {r['avg_task_accuracy']*100:.2f} | "
                f"{r['final_accuracy']*100:.2f} | "
                f"{r['forgetting_rate']*100:.2f} | "
                f"{cfg['buffer_size']} | {cfg['der_alpha']} | {cfg['der_beta']} | "
                f"{cfg['lambda_kd']} | {cfg['lambda_proto_aug']} |"
            )
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVED] {md_path}")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_results = {}
    total_experiments = len(ALL_ALGORITHMS) * 2 + len(ABLATION_CONFIGS)
    done = 0

    # ================================================================
    # Part 1: EMNIST-Letters — all algorithms
    # ================================================================
    ds_name = "EMNIST-Letters"
    print(f"\n{'#'*60}")
    print(f"# Part 1a: {ds_name} — {len(ALL_ALGORITHMS)} algorithms")
    print(f"{'#'*60}")
    emnist_results = []
    for alg in ALL_ALGORITHMS:
        done += 1
        remaining = total_experiments - done
        print(f"\n>>> [{done}/{total_experiments}] {ds_name} / {alg}  (remaining: {remaining})")
        cmd = build_cmd(EMNIST_BASE, alg)
        r = run_one(alg, cmd)
        if r:
            emnist_results.append(r)
        # Write intermediate results
        all_results[ds_name] = emnist_results
        write_markdown(all_results, {})

    # ================================================================
    # Part 1: CIFAR100 — all algorithms
    # ================================================================
    ds_name = "CIFAR100"
    print(f"\n{'#'*60}")
    print(f"# Part 1b: {ds_name} — {len(ALL_ALGORITHMS)} algorithms")
    print(f"{'#'*60}")
    cifar_results = []
    for alg in ALL_ALGORITHMS:
        done += 1
        remaining = total_experiments - done
        print(f"\n>>> [{done}/{total_experiments}] {ds_name} / {alg}  (remaining: {remaining})")
        cmd = build_cmd(CIFAR100_BASE, alg)
        r = run_one(alg, cmd)
        if r:
            cifar_results.append(r)
        all_results[ds_name] = cifar_results
        write_markdown(all_results, {})

    # ================================================================
    # Part 2: Ablation study for D2FCL on EMNIST-Letters
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# Part 2: D2FCL Ablation Study — {len(ABLATION_CONFIGS)} configs")
    print(f"{'#'*60}")
    ablation_results = {}
    for tag, overrides in ABLATION_CONFIGS.items():
        done += 1
        remaining = total_experiments - done
        print(f"\n>>> [{done}/{total_experiments}] Ablation: {tag}  (remaining: {remaining})")
        cmd = build_cmd(EMNIST_BASE, "D2FCL", **overrides)
        r = run_one(f"D2FCL ({tag})", cmd)
        ablation_results[tag] = r
        write_markdown(all_results, ablation_results)

    # Final summary
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_experiments} total)")
    print(f"Results saved to EXPERIMENT_RESULTS.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
