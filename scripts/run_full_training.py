#!/usr/bin/env python
"""Full 60-round training for FedAvg, Local, and DCFCL to match paper results."""
import sys
import os
import logging
import torch
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed

def run_full_training(algorithm, num_rounds=60, num_tasks=6, local_epochs=100):
    """Run full training and report per-task results."""
    setup_seed(42)
    
    config = Config()
    config.algorithm = algorithm
    config.num_rounds = num_rounds
    config.num_tasks = num_tasks
    config.local_epochs = local_epochs
    config.rounds_per_task = num_rounds // num_tasks
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    log_fmt = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.WARNING, format=log_fmt)
    logger = logging.getLogger('train')
    
    print(f"\n{'='*70}")
    print(f"FULL TRAINING: {algorithm}")
    print(f"lr={config.lr}, wd={config.weight_decay}, epochs={local_epochs}")
    print(f"rounds={num_rounds}, tasks={num_tasks}, device={device}")
    if algorithm == 'DCFCL':
        print(f"lambda_kd={config.lambda_kd}, sw={config.sw}, gw={config.global_weight}")
    print(f"{'='*70}")
    
    server = DCFCLServer(config, device)
    
    rounds_per_task = config.rounds_per_task
    all_accs = []
    task_end_accs = []
    
    start_time = datetime.now()
    
    for task in range(num_tasks):
        print(f"\n=== Task {task} ===")
        
        if task > 0:
            server._update_clients_for_new_task(task)
        server._update_available_labels()
        
        # Log task info
        for c in server.clients:
            if c.id == 0:
                print(f"  Client 0 classes so far: {c.classes_so_far}")
        
        for round_in_task in range(rounds_per_task):
            global_round = task * rounds_per_task + round_in_task
            
            if config.algorithm != 'Local' and global_round == 0:
                server._broadcast_parameters()
            
            train_results = server._local_training(global_round, task)
            server._server_aggregation(global_round, task, train_results)
            
            accs, avg_acc, num_samples = server._evaluate()
            all_accs.append(avg_acc)
            server.all_acc.append(avg_acc)
            
            if round_in_task == rounds_per_task - 1:
                server.all_accs.append(accs)
                if task > 0:
                    forget = server._compute_forgetting()
                    server.all_forget.append(forget)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  Round {global_round:2d} (T{task}R{round_in_task}): "
                  f"acc={avg_acc:.4f}  [{elapsed:.0f}s]")
        
        # Print per-task accuracy at end of this task
        task_end_accs.append(avg_acc)
        if task in server.all_accs[-1]:
            per_task = server.all_accs[-1]
            print(f"  Per-client per-task accuracy:")
            for cid in sorted(per_task.keys()):
                task_accs = per_task[cid]
                print(f"    Client {cid}: {[f'{a:.3f}' for a in task_accs]}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Final results
    final_acc = all_accs[-1]
    final_forget = server.all_forget[-1] if server.all_forget else 0.0
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS [{algorithm}]")
    print(f"{'='*70}")
    print(f"Average Accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"Forgetting Rate:  {final_forget:.4f} ({final_forget*100:.1f}%)")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Task-end accuracies: {[f'{a:.4f}' for a in task_end_accs]}")
    print(f"{'='*70}")
    
    return {
        'algorithm': algorithm,
        'final_accuracy': final_acc,
        'forgetting_rate': final_forget,
        'all_accuracies': [float(a) for a in all_accs],
        'task_end_accs': [float(a) for a in task_end_accs],
        'time_seconds': total_time
    }


if __name__ == '__main__':
    algorithms = sys.argv[1:] if len(sys.argv) > 1 else ['FedAvg', 'DCFCL']
    
    all_results = {}
    for algo in algorithms:
        try:
            result = run_full_training(algo)
            all_results[algo] = result
        except Exception as e:
            print(f"\nERROR [{algo}]: {e}")
            import traceback
            traceback.print_exc()
            all_results[algo] = {'error': str(e)}
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Algorithm':15s} {'Accuracy':>10s} {'Forgetting':>12s} {'Time':>8s}")
    print(f"{'-'*45}")
    for algo, res in all_results.items():
        if 'error' in res:
            print(f"{algo:15s} ERROR: {res['error']}")
        else:
            print(f"{algo:15s} {res['final_accuracy']*100:9.1f}% "
                  f"{res['forgetting_rate']*100:10.1f}% "
                  f"{res['time_seconds']:7.0f}s")
    
    print(f"\nPaper reference (EMNIST-LTP):")
    print(f"  FedAvg: 32.5%, DCFCL: 52.5%, Local: 12.3%")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'results/full_training_{timestamp}.json'
    os.makedirs('results', exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")
