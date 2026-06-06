#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DynDFCL module ablation script.

Goal: verify the contribution of each module in the full DynDFCL method.

Module definitions
------------------
A. DER++         (use_der)                : dark experience replay; mitigates forgetting
B. Directed      (directed_collaboration) : asymmetric coalition aggregation
C. Mask          (use_coalition_mask)     : restricts coalition formation for some pairs

Ablation matrix (7 experiments)
-------------------------------
| Experiment          | DER++ | Directed | Mask |
|---------------------|-------|----------|------|
| DCFCL (baseline)    |   x   |     x    |   x  |
| DynDFCL-base        |   v   |     x    |   x  |
| DynDFCL+Directed    |   v   |     v    |   x  |
| DynDFCL+Mask        |   v   |     x    |   v  |
| DynDFCL+D+M (Full)  |   v   |     v    |   v  |
| (extra) no-DER+Dir  |   x   |     v    |   x  |
| (extra) no-DER+Mask |   x   |     x    |   v  |

Usage
-----
  # Quick check (fewer rounds)
  python scripts/run_module_ablation.py --quick

  # Full ablation on EMNIST
  python scripts/run_module_ablation.py

  # Pick a dataset
  python scripts/run_module_ablation.py --dataset cifar100

  # Run only the 5 core variants (skip the 2 extras)
  python scripts/run_module_ablation.py --core_only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

PYTHON = sys.executable
MAIN = str(REPO_DIR / "main.py")

# ---------------------------------------------------------------------------
# Base configurations (EMNIST-Letters)
# ---------------------------------------------------------------------------
EMNIST_BASE = dict(
    dataset="EMNIST-Letters",
    data_split_file="split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl",
    num_users=8,
    num_tasks=6,
    num_rounds=60,
    local_epochs=100,
    batch_size=64,
    model="cnn",
    lr=1e-4,
    weight_decay=1e-5,
    seed=42,
    # DCFCL shared parameters
    sw=0.1,
    lambda_kd=0.2,
    lambda_proto_aug=2.0,
    global_weight=0.9,
    ema_global=0.9,
    dcfcl_broadcast=0,
    # DER++ parameters (use_der is set per experiment)
    buffer_size=500,
    der_alpha=0.5,
    der_beta=0.5,
    # Directed-coalition parameters (directed_collaboration is set per experiment)
    directed_mode="gradient",
    directed_temperature=1.0,
    directed_self_weight=0.5,
    # Mask parameters (use_coalition_mask is set per experiment)
    coalition_mask_type="group",
    num_client_groups=2,
)

CIFAR100_BASE = dict(
    dataset="CIFAR100",
    data_split_file="split_files/CIFAR100_split_cn10_tn10_cet10_cs1_s2571.pkl",
    num_users=10,
    num_tasks=10,
    num_rounds=100,
    local_epochs=50,
    batch_size=64,
    model="resnet18",
    lr=1e-3,
    weight_decay=1e-3,
    seed=42,
    sw=0.1,
    lambda_kd=0.2,
    lambda_proto_aug=2.0,
    global_weight=0.9,
    ema_global=0.9,
    dcfcl_broadcast=0,
    buffer_size=500,
    der_alpha=0.5,
    der_beta=0.5,
    directed_mode="gradient",
    directed_temperature=1.0,
    directed_self_weight=0.5,
    coalition_mask_type="group",
    num_client_groups=2,
)

# ---------------------------------------------------------------------------
# Ablation matrix (module toggles).
# Each entry key maps to a main.py flag; store_true flags handled as bools.
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = {
    # ----- 5 core experiments -----
    "DCFCL_baseline": dict(
        algorithm="DCFCL",   # original DCFCL: no DER, no directed, no mask
    ),
    "DynDFCL_base": dict(
        algorithm="DynDFCL",
        use_der=True,
        directed_collaboration=False,
        use_coalition_mask=False,
    ),
    "DynDFCL_Directed": dict(
        algorithm="DynDFCL",
        use_der=True,
        directed_collaboration=True,
        use_coalition_mask=False,
    ),
    "DynDFCL_Mask": dict(
        algorithm="DynDFCL",
        use_der=True,
        directed_collaboration=False,
        use_coalition_mask=True,
    ),
    "DynDFCL_Full": dict(
        algorithm="DynDFCL",
        use_der=True,
        directed_collaboration=True,
        use_coalition_mask=True,
    ),
    # ----- 2 extras: DER independence from the other modules -----
    "DCFCL_NoDER_Directed": dict(
        algorithm="DynDFCL",
        use_der=False,
        directed_collaboration=True,
        use_coalition_mask=False,
    ),
    "DCFCL_NoDER_Mask": dict(
        algorithm="DynDFCL",
        use_der=False,
        directed_collaboration=False,
        use_coalition_mask=True,
    ),
}

CORE_VARIANTS = [
    "DCFCL_baseline",
    "DynDFCL_base",
    "DynDFCL_Directed",
    "DynDFCL_Mask",
    "DynDFCL_Full",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_cmd(base_config: dict, variant_cfg: dict, result_dir: str) -> list:
    """Build the CLI command from a base config + variant overrides."""
    merged = {**base_config, **variant_cfg}
    cmd = [PYTHON, MAIN]

    for k, v in merged.items():
        if k == "algorithm":
            cmd += ["--algorithm", str(v)]
        elif isinstance(v, bool):
            if k == "use_der":
                if v:
                    cmd.append("--use_der")
                else:
                    cmd.append("--no_use_der")
            elif k == "directed_collaboration":
                if v:
                    cmd.append("--directed_collaboration")
                # False -> omit the flag (default off).
            elif k == "use_coalition_mask":
                if v:
                    cmd.append("--use_coalition_mask")
        else:
            cmd += [f"--{k}", str(v)]

    cmd += ["--result_dir", result_dir]
    return cmd


def run_experiment(name: str, cmd: list, timeout: int = 7200) -> dict | None:
    """Run one experiment and parse its results."""
    print(f"\n{'='*60}")
    print(f"[RUN]  {name}")
    # Only echo the salient flags.
    key_args = ["--algorithm", "--use_der", "--no_use_der",
                "--directed_collaboration", "--use_coalition_mask"]
    shown = []
    i = 0
    while i < len(cmd):
        if cmd[i] in key_args:
            shown.append(cmd[i])
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                shown.append(cmd[i + 1])
                i += 2
            else:
                i += 1
        else:
            i += 1
    print(f"       {' '.join(shown)}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_DIR),
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name}")
        return None

    if result.returncode != 0:
        print(f"[FAIL]  {name}  (exit={result.returncode}, {elapsed:.0f}s)")
        if result.stderr:
            print(result.stderr[-800:])
        return None

    print(f"[DONE]  {name}  ({elapsed:.0f}s)")

    # Parse results.json from the latest run directory.
    metrics = _parse_results(cmd, elapsed)
    if metrics:
        metrics["name"] = name
        metrics["elapsed_s"] = elapsed
    return metrics


def _parse_results(cmd: list, elapsed: float) -> dict | None:
    """Locate the latest result directory and parse results.json."""
    algorithm = _flag_val(cmd, "--algorithm", "")
    dataset = _flag_val(cmd, "--dataset", "")
    result_dir_base = _flag_val(cmd, "--result_dir", "./results")

    prefix = f"{algorithm}_{dataset}_"
    try:
        candidates = sorted(
            [d for d in os.listdir(result_dir_base) if d.startswith(prefix)],
            reverse=True,
        )
    except FileNotFoundError:
        return None

    if not candidates:
        return None

    json_path = os.path.join(result_dir_base, candidates[0], "results.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        data = json.load(f)

    num_tasks = int(_flag_val(cmd, "--num_tasks", "6"))
    all_acc = data.get("all_accuracies", [])
    rpt = len(all_acc) // num_tasks if num_tasks > 0 else 0
    if rpt > 0:
        phase_end = [all_acc[rpt * (t + 1) - 1] for t in range(num_tasks)]
        avg_task = sum(phase_end) / len(phase_end)
    else:
        phase_end = []
        avg_task = data.get("final_accuracy", 0)

    return {
        "final_accuracy": data.get("final_accuracy", 0),
        "forgetting_rate": data.get("forgetting_rate", 0),
        "avg_task_accuracy": avg_task,
        "phase_end_accs": phase_end,
        "result_dir": candidates[0],
    }


def _flag_val(cmd: list, flag: str, default: str) -> str:
    """Pull the value of a flag from the command list."""
    try:
        idx = cmd.index(flag)
        return cmd[idx + 1]
    except (ValueError, IndexError):
        return default


def print_summary(all_results: dict):
    """Print the ablation summary table."""
    print("\n" + "=" * 80)
    print("DynDFCL module ablation summary")
    print("=" * 80)

    header = f"{'Experiment':<30} {'Final Acc':>10} {'Avg Task Acc':>13} {'Forgetting':>11}"
    print(header)
    print("-" * 70)

    for name, r in all_results.items():
        if r is None:
            print(f"  {name:<28} {'FAILED':>10}")
            continue
        fa = r.get("final_accuracy", 0)
        at = r.get("avg_task_accuracy", 0)
        fg = r.get("forgetting_rate", 0)
        print(f"  {name:<28} {fa*100:>9.2f}%  {at*100:>12.2f}%  {fg*100:>10.2f}%")

    print("=" * 80)

    # Per-variant deltas relative to the base configuration.
    base_key = "DynDFCL_base"

    if base_key in all_results and all_results[base_key]:
        base_acc = all_results[base_key]["final_accuracy"]
        print(f"\nDeltas vs {base_key} (Final Acc):")
        for k, r in all_results.items():
            if k == base_key or r is None:
                continue
            delta = r["final_accuracy"] - base_acc
            sign = "+" if delta >= 0 else ""
            print(f"  {k:<28}  {sign}{delta*100:.2f}%")


def save_results(all_results: dict, output_dir: str):
    """Save ablation results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"module_ablation_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DynDFCL module ablation")
    parser.add_argument("--dataset", choices=["emnist", "cifar100"], default="emnist",
                        help="dataset (emnist | cifar100)")
    parser.add_argument("--quick", action="store_true",
                        help="quick mode: shorter run (num_rounds=12, local_epochs=20)")
    parser.add_argument("--core_only", action="store_true",
                        help="only run the 5 core variants; skip the 2 extras")
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="root result directory")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="per-experiment timeout (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Pick the base config.
    base_cfg = EMNIST_BASE.copy() if args.dataset == "emnist" else CIFAR100_BASE.copy()

    # Quick mode.
    if args.quick:
        base_cfg.update(num_rounds=12, local_epochs=20)
        print("[quick] num_rounds=12, local_epochs=20")

    # Pick the variants to run.
    variants = CORE_VARIANTS if args.core_only else list(ABLATION_VARIANTS.keys())
    print(f"\nVariants to run: {len(variants)}")
    print(f"{'5 core' if args.core_only else '7 total'}: {variants}")

    result_dir = os.path.abspath(args.result_dir)
    all_results = {}

    for name in variants:
        variant_cfg = ABLATION_VARIANTS[name]
        cmd = build_cmd(base_cfg, variant_cfg, result_dir)
        metrics = run_experiment(name, cmd, timeout=args.timeout)
        all_results[name] = metrics

    # Print summary.
    print_summary(all_results)

    # Save JSON.
    ablation_dir = os.path.join(result_dir, "module_ablation_results")
    save_results(all_results, ablation_dir)


if __name__ == "__main__":
    main()
