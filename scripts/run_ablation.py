#!/usr/bin/env python
"""Ablation study: test DCFCL components individually to identify what's helping/hurting."""
import sys, os, logging, torch, numpy as np, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed

def run_ablation(name, config_overrides, num_rounds=60, num_tasks=6):
    """Run training with specific config and return final accuracy."""
    setup_seed(42)
    config = Config()
    config.num_rounds = num_rounds
    config.num_tasks = num_tasks 
    config.local_epochs = 100
    config.rounds_per_task = num_rounds // num_tasks
    
    for k, v in config_overrides.items():
        setattr(config, k, v)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.getLogger('DCFCL').setLevel(logging.WARNING)
    logging.getLogger('DCFCL.Server').setLevel(logging.WARNING)
    
    server = DCFCLServer(config, device)
    
    # Override aggregation for ablation
    original_aggregate = server._server_aggregation
    skip_coalition = config_overrides.get('_skip_coalition', False)
    
    if skip_coalition:
        # Replace DCFCL aggregation with FedAvg-style broadcast
        def fedavg_aggregate(global_round, task, train_results):
            server._zero_model_parameters(server.model)
            total_samples = sum(c.train_samples for c in server.clients)
            for client in server.clients:
                ratio = client.train_samples / total_samples
                server._add_parameters(client.model, ratio)
            server._broadcast_parameters()
        
        if config.algorithm == 'DCFCL':
            server._server_aggregation = lambda gr, t, tr: fedavg_aggregate(gr, t, tr)
    
    # Training
    task_accs = []
    for task in range(num_tasks):
        if task > 0:
            server._update_clients_for_new_task(task)
        server._update_available_labels()
        
        for round_in_task in range(config.rounds_per_task):
            global_round = task * config.rounds_per_task + round_in_task
            
            if config.algorithm != 'Local' and global_round == 0:
                server._broadcast_parameters()
            
            train_results = server._local_training(global_round, task)
            server._server_aggregation(global_round, task, train_results)
            
            accs, avg_acc, _ = server._evaluate()
            
            if round_in_task == config.rounds_per_task - 1:
                server.all_accs.append(accs)
                task_accs.append(avg_acc)
    
    final_acc = task_accs[-1]
    print(f"  {name:40s}: {final_acc*100:5.1f}%  (task progression: {[f'{a:.3f}' for a in task_accs]})")
    return final_acc

if __name__ == '__main__':
    print("="*80)
    print("ABLATION STUDY: Isolating DCFCL component effects")
    print("="*80)
    
    results = {}
    
    # 1. FedAvg baseline (no KD, no coalition)
    results['FedAvg'] = run_ablation('FedAvg (no KD, no coalition)', 
        {'algorithm': 'FedAvg'})
    
    # 2. FedAvg + KD (FedLwF style, alpha=1.0)
    results['FedLwF'] = run_ablation('FedLwF (FedAvg + KD alpha=1.0)',
        {'algorithm': 'FedLwF'})
    
    # 3. DCFCL training (CE loss + KD lambda=0.2) but FedAvg aggregation (no coalition)
    results['DCFCL-no-coalition'] = run_ablation('DCFCL train + FedAvg agg (no coalition)',
        {'algorithm': 'DCFCL', '_skip_coalition': True})
    
    # 4. DCFCL training with KD=0 but WITH coalition
    results['DCFCL-no-KD'] = run_ablation('DCFCL + coalition, no KD (lambda_kd=0)',
        {'algorithm': 'DCFCL', 'lambda_kd': 0.0})
    
    # 5. Full DCFCL (KD + coalition)
    results['DCFCL'] = run_ablation('DCFCL full (KD=0.2 + coalition)',
        {'algorithm': 'DCFCL'})
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, acc in results.items():
        print(f"  {name:25s}: {acc*100:.1f}%")
    print(f"\nPaper reference (EMNIST-LTP):")
    print(f"  FedAvg=32.5%, w/o KD=50.3%, w/o CE-FedAvg=32.5%, DCFCL=52.5%")
