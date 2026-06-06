#!/usr/bin/env python3
"""
Directed-collaboration comparison.

Purpose: validate whether directed collaboration outperforms standard
symmetric coalition aggregation.

Design:
1. baseline:           standard symmetric coalition (directed_collaboration=False)
2. directed_gradient:  gradient-based directed collaboration

We use a smaller buffer_size (100) to let some forgetting occur, so the
personalization advantage of directed collaboration can show through.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import yaml
import json
import re
from datetime import datetime
from pathlib import Path
import logging

# Logging setup.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(config_overrides: dict, exp_name: str, base_config: str, result_dir: Path):
    """Run a single experiment with config overrides."""
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)

    config.update(config_overrides)
    config['result_dir'] = str(result_dir / exp_name)

    temp_config_path = result_dir / f'{exp_name}_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"{'='*60}")
    logger.info(f"Running: {exp_name}")
    logger.info(f"Config: directed_collaboration={config.get('directed_collaboration', False)}, "
                f"buffer_size={config.get('buffer_size', 500)}")
    logger.info(f"{'='*60}")

    result = subprocess.run(
        ['python', 'main.py', '--config', str(temp_config_path)],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    if result.returncode != 0:
        logger.error(f"Error running {exp_name}:")
        logger.error(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return None

    output = result.stdout + result.stderr
    metrics = parse_metrics(output)
    metrics['exp_name'] = exp_name
    metrics['config'] = config_overrides

    return metrics


def parse_metrics(output: str) -> dict:
    """Parse metrics from the experiment output."""
    metrics = {}

    final_acc_match = re.search(r'Final.*[Aa]ccuracy[:\s]+(\d+\.?\d*)', output)
    if final_acc_match:
        metrics['final_accuracy'] = float(final_acc_match.group(1))

    avg_acc_match = re.search(r'[Aa]verage.*[Aa]ccuracy[:\s]+(\d+\.?\d*)', output)
    if avg_acc_match:
        metrics['avg_accuracy'] = float(avg_acc_match.group(1))

    task_acc_match = re.search(r'Per-task accuracy.*?:\s*\[([\d.,\s]+)\]', output)
    if task_acc_match:
        try:
            accs = [float(x.strip()) for x in task_acc_match.group(1).split(',')]
            metrics['task_accuracies'] = accs
        except Exception:
            pass

    forget_match = re.search(r'[Ff]orget(?:ting)?[:\s]+(\d+\.?\d*)', output)
    if forget_match:
        metrics['forgetting'] = float(forget_match.group(1))

    acc_values = re.findall(r'Average accuracy:\s*(\d+\.?\d*)', output)
    if acc_values:
        metrics['final_round_accuracy'] = float(acc_values[-1])
        metrics['all_round_accuracies'] = [float(x) for x in acc_values]

    return metrics


def main():
    """Run the directed-collaboration comparison."""
    logger.info("="*70)
    logger.info("DIRECTED COLLABORATION COMPARISON")
    logger.info("="*70)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = Path(f'./results/directed_comparison_{timestamp}')
    result_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(result_dir / 'experiment.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    base_config = './configs/dcfcl_emnist.yaml'

    # buffer_size=100 lets some forgetting occur.
    experiments = [
        {
            'name': 'baseline',
            'overrides': {
                'algorithm': 'DynDFCL',
                'directed_collaboration': False,
                'buffer_size': 100,
            }
        },
        {
            'name': 'directed_gradient',
            'overrides': {
                'algorithm': 'DynDFCL',
                'directed_collaboration': True,
                'directed_mode': 'gradient',
                'directed_threshold': 0.0,
                'directed_temperature': 1.0,
                'directed_self_weight': 0.5,
                'buffer_size': 100,
            }
        },
    ]

    results = []

    for exp in experiments:
        result = run_experiment(
            exp['overrides'],
            exp['name'],
            base_config,
            result_dir
        )
        if result:
            results.append(result)
            logger.info(f"[ok] {exp['name']}: Final Acc = {result.get('final_round_accuracy', 'N/A')}")
        else:
            logger.error(f"[fail] {exp['name']}")

    with open(result_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("Comparison summary")
    logger.info("="*70)

    baseline_result = next((r for r in results if r['exp_name'] == 'baseline'), None)
    directed_result = next((r for r in results if r['exp_name'] == 'directed_gradient'), None)

    if baseline_result and directed_result:
        baseline_acc = baseline_result.get('final_round_accuracy', 0)
        directed_acc = directed_result.get('final_round_accuracy', 0)
        improvement = directed_acc - baseline_acc

        logger.info(f"\n{'Method':<25} {'Final Acc':<15}")
        logger.info("-" * 40)
        logger.info(f"{'Baseline (symmetric)':<25} {baseline_acc:.4f}")
        logger.info(f"{'Directed (asymmetric)':<25} {directed_acc:.4f}")
        logger.info("-" * 40)
        logger.info(f"{'Delta':<25} {improvement:+.4f} ({improvement*100:+.2f}%)")

        report = f"""# Directed Collaboration Comparison Report

## Setup
- Dataset: EMNIST-Letters
- Algorithm: DynDFCL
- buffer_size: 100 (reduced so forgetting can occur)
- Number of clients: 8
- Number of tasks: 6

## Results

| Method | Final Acc |
|--------|-----------|
| Baseline (symmetric)  | {baseline_acc:.4f} |
| Directed (asymmetric) | {directed_acc:.4f} |

## Conclusion

Delta: **{improvement:+.4f}** ({improvement*100:+.2f}%)

{"Directed collaboration improves accuracy." if improvement > 0 else "Directed collaboration did not improve accuracy under these settings; further tuning may be needed."}

## Details

### Baseline
- Setting: directed_collaboration=False
- Final accuracy: {baseline_acc:.4f}

### Directed Gradient
- Setting: directed_collaboration=True, mode=gradient
- Final accuracy: {directed_acc:.4f}
"""
        with open(result_dir / 'COMPARISON_REPORT.md', 'w') as f:
            f.write(report)

        logger.info(f"\nReport saved to: {result_dir / 'COMPARISON_REPORT.md'}")

    logger.info("\n" + "="*70)
    logger.info(f"All results saved to: {result_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
