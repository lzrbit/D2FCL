#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Server module for DCFCL.

This module implements the federated learning server that orchestrates
training across clients and performs model aggregation with coalition formation.
"""

import copy
import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from .models import create_model
from .config import Config
from utils.data_loader import get_dataset, read_user_data

logger = logging.getLogger('DCFCL.Server')


class ProtoQueue:
    """Queue for storing and aggregating prototypes across rounds."""
    
    def __init__(self, n_classes: int, max_length: int = 100):
        self.n_classes = n_classes
        self.queue = {i: [] for i in range(n_classes)}
        self.max_length = max_length
        self.global_proto = {i: 0 for i in range(n_classes)}
    
    def insert(self, local_proto: Dict, local_radius: float, num_samples: Dict):
        """Insert local prototypes into the queue."""
        for cls_id in local_proto.keys():
            self.queue[cls_id].append((local_proto[cls_id], local_radius, num_samples.get(cls_id, 0)))
            
            # Keep queue bounded
            while len(self.queue[cls_id]) > self.max_length:
                self.queue[cls_id].pop(0)
    
    def compute_mean(self) -> Dict:
        """Compute weighted mean of prototypes."""
        for cls_id in range(self.n_classes):
            if len(self.queue[cls_id]) > 1:
                total_weight = 0
                weighted_sum = 0
                for proto, radius, weight in self.queue[cls_id]:
                    weighted_sum += proto * weight
                    total_weight += weight
                
                if total_weight > 0:
                    self.global_proto[cls_id] = weighted_sum / total_weight
        
        return self.global_proto


class DCFCLServer:
    """
    DCFCL Server for Federated Continual Learning.
    
    Orchestrates training across clients with:
    - Coalition formation based on model similarity
    - Prototype aggregation for continual learning
    - Various FL aggregation strategies
    
    Attributes:
        config: Configuration object
        device: Torch device
        model: Global model
        clients: List of client objects
    """
    
    def __init__(self, config: Config, device: torch.device):
        """
        Initialize DCFCL server.
        
        Args:
            config: Configuration object
            device: Torch device to use
        """
        self.config = config
        self.device = device
        
        # Create global model
        self.model = create_model(config)
        self.model.to(device)
        
        # Load data
        logger.info("Loading dataset...")
        self.data = get_dataset(config)
        
        # Initialize clients
        self._init_clients()
        
        # Coalition formation state
        self.similarity_matrix = None
        self.unions = None
        self.pay_table = {}
        self.all_partitions = []
        self.stable_state = None
        self.coalition_mask = None  # Mask for forbidden coalition pairs
        
        # Prototype aggregation
        self.proto_queue = ProtoQueue(config.num_classes, config.proto_queue_length)
        self.proto_global = None
        self.radius_global = None
        
        # SCAFFOLD state
        self.server_control = None
        self.client_control = {}
        
        # Results tracking
        self.all_acc = []
        self.all_accs = []
        self.all_forget = []
        
        # Initialize algorithm-specific components
        self._init_algorithm()
        
        logger.info(f"Server initialized with {len(self.clients)} clients")
    
    def _init_clients(self):
        """Initialize all clients with their data."""
        # Lazy import to avoid the FL_model -> core -> core.server -> FL_model cycle.
        from FL_model import create_client

        self.clients = []
        client_names = self.data['client_names']
        
        for i, name in enumerate(client_names):
            # Get initial task data
            client_id, train_data, test_data, label_info = read_user_data(
                i, self.data, self.config.dataset, task=0
            )
            
            # Instantiate the algorithm-specific client based on config.algorithm.
            client = create_client(
                config=self.config,
                client_id=i,
                model=self.model,
                train_data=train_data,
                test_data=test_data,
                label_info=label_info,
                unique_labels=self.data['unique_labels']
            )
            self.clients.append(client)
            
            logger.debug(f"Initialized client {i} with labels: {label_info.get('labels', [])}")
        
        self.num_clients = len(self.clients)
    
    def _init_algorithm(self):
        """Initialize algorithm-specific components."""
        if self.config.algorithm == 'SCAFFOLD':
            self.server_control = self._init_control(self.model)
            self.client_control = {c.id: self._init_control(self.model) for c in self.clients}
        
        # Create partition table for coalition formation
        if self.config.algorithm in ['DCFCL', 'DynDFCL', 'ClusterFL']:
            # Initialize coalition mask if enabled
            if getattr(self.config, 'use_coalition_mask', False):
                self._init_coalition_mask()
            self._create_partition_table()
    
    def _init_coalition_mask(self):
        """Initialize coalition mask matrix.
        
        The mask is a boolean matrix where mask[i][j] = True means
        clients i and j CANNOT be in the same coalition.
        """
        n = self.num_clients
        self.coalition_mask = np.zeros((n, n), dtype=bool)
        
        mask_type = getattr(self.config, 'coalition_mask_type', 'custom')
        
        if mask_type == 'custom':
            # Use explicitly specified forbidden pairs
            forbidden_pairs = getattr(self.config, 'coalition_forbidden_pairs', [])
            for i, j in forbidden_pairs:
                if 0 <= i < n and 0 <= j < n:
                    self.coalition_mask[i][j] = True
                    self.coalition_mask[j][i] = True  # Symmetric
                    
        elif mask_type == 'random':
            # Random forbidden pairs based on density
            density = getattr(self.config, 'coalition_mask_density', 0.2)
            np.random.seed(self.config.seed)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.random.random() < density:
                        self.coalition_mask[i][j] = True
                        self.coalition_mask[j][i] = True
                        
        elif mask_type == 'group':
            # Group-based: clients in different groups cannot cooperate
            num_groups = getattr(self.config, 'num_client_groups', 2)
            clients_per_group = n // num_groups
            
            # Assign clients to groups
            client_groups = [i // clients_per_group for i in range(n)]
            # Handle remainder
            for i in range(n):
                if client_groups[i] >= num_groups:
                    client_groups[i] = num_groups - 1
            
            # Clients in different groups cannot cooperate
            for i in range(n):
                for j in range(i + 1, n):
                    if client_groups[i] != client_groups[j]:
                        self.coalition_mask[i][j] = True
                        self.coalition_mask[j][i] = True
        
        # Log mask info
        num_forbidden = np.sum(self.coalition_mask) // 2  # Divide by 2 for symmetric
        total_pairs = n * (n - 1) // 2
        logger.info(f"Coalition mask initialized: {num_forbidden}/{total_pairs} pairs forbidden (type={mask_type})")
    
    def _is_valid_coalition(self, coalition: set) -> bool:
        """Check if a coalition is valid given the mask constraints.
        
        A coalition is valid if no two clients in it are forbidden from cooperating.
        """
        if self.coalition_mask is None:
            return True
        
        clients = list(coalition)
        for i in range(len(clients)):
            for j in range(i + 1, len(clients)):
                if self.coalition_mask[clients[i]][clients[j]]:
                    return False
        return True
    
    def _is_valid_partition(self, partition: List[set]) -> bool:
        """Check if a partition is valid (all coalitions are valid)."""
        return all(self._is_valid_coalition(c) for c in partition)
    
    def _init_control(self, model) -> Dict[str, torch.Tensor]:
        """Initialize control variates for SCAFFOLD."""
        return {name: torch.zeros_like(p.data).cpu() 
                for name, p in model.state_dict().items()}
    
    def _create_partition_table(self):
        """Create partition table for coalition formation.
        
        If coalition mask is enabled, filter out invalid partitions.
        """
        n = self.num_clients
        clients = set(range(n))
        
        # Generate all possible partitions
        all_partitions = list(self._enumerate_partitions(clients))
        
        # Filter partitions if mask is enabled
        if getattr(self.config, 'use_coalition_mask', False) and self.coalition_mask is not None:
            valid_partitions = [p for p in all_partitions if self._is_valid_partition(p)]
            logger.info(f"Partition filtering: {len(all_partitions)} total -> {len(valid_partitions)} valid")
            self.all_partitions = valid_partitions
        else:
            self.all_partitions = all_partitions
        
        # Create pay table
        self.pay_table = {
            tuple(tuple(p) for p in part): [0] * n 
            for part in self.all_partitions
        }
    
    def _enumerate_partitions(self, s: set):
        """Enumerate all possible partitions of a set."""
        if not s:
            yield []
            return
        first = s.pop()
        for smaller in self._enumerate_partitions(s.copy()):
            for i, subset in enumerate(smaller):
                yield smaller[:i] + [subset | {first}] + smaller[i+1:]
            yield [{first}] + smaller
        s.add(first)  # Restore set
    
    # =========================================================================
    # Training
    # =========================================================================
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary with training results
        """
        rounds_per_task = self.config.rounds_per_task
        
        for task in range(self.config.num_tasks):
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Task {task}")
            logger.info(f"{'='*60}")
            
            # Initialize new task for clients (except first task)
            if task > 0:
                self._update_clients_for_new_task(task)
            
            # Update available labels
            self._update_available_labels()
            
            # Log task info
            for c in self.clients:
                logger.debug(f"Client {c.id} - Classes so far: {c.classes_so_far}")
            
            # Train for rounds_per_task rounds
            for round_in_task in range(rounds_per_task):
                global_round = task * rounds_per_task + round_in_task
                
                logger.info(f"\nRound {global_round} (Task {task}, Round {round_in_task})")
                
                # Distribute global model to clients (except for Local algorithm)
                if self.config.algorithm != 'Local' and global_round == 0:
                    self._broadcast_parameters()
                
                # Local training
                train_results = self._local_training(global_round, task)
                
                # Server aggregation
                self._server_aggregation(global_round, task, train_results)
                
                # Evaluation
                accs, avg_acc, num_samples = self._evaluate()
                self.all_acc.append(avg_acc)
                
                logger.info(f"Average accuracy: {avg_acc:.4f}")
                
                # Track per-task accuracy at end of task
                if round_in_task == rounds_per_task - 1:
                    self.all_accs.append(accs)
                    if task > 0:
                        forget_rate = self._compute_forgetting()
                        self.all_forget.append(forget_rate)
                        logger.info(f"Forgetting rate: {forget_rate:.4f}")
                    
                    # Log per-task accuracy breakdown
                    task_acc_summary = self._compute_per_task_accuracy(accs)
                    logger.info(f"Per-task accuracy after Task {task}: {task_acc_summary}")
        
        # Final results
        final_accuracy = self.all_acc[-1] if self.all_acc else 0.0
        forgetting_rate = self.all_forget[-1] if self.all_forget else 0.0
        
        # Compute detailed per-task metrics
        per_task_acc, per_task_forget = self._compute_detailed_metrics()
        
        # Evaluate emergence phenomenon if enabled
        emergence_results = None
        if getattr(self.config, 'evaluate_emergence', False):
            logger.info("\n" + "=" * 60)
            logger.info("Evaluating Emergence Phenomenon")
            logger.info("=" * 60)
            
            emergence_results = self.evaluate_emergence()
            
            logger.info(f"Global Emergence Rate (unseen class accuracy): {emergence_results['global_emergence_rate']:.4f}")
            logger.info(f"Seen Class Accuracy: {emergence_results['seen_accuracy']:.4f}")
            logger.info(f"Total Emergence Samples: {emergence_results['total_emergence_samples']}")
            logger.info(f"Unseen: {emergence_results['global_unseen_correct']}/{emergence_results['global_unseen_total']}")
            logger.info(f"Seen: {emergence_results['global_seen_correct']}/{emergence_results['global_seen_total']}")
            
            # Per-client emergence
            logger.info("\nPer-client Emergence:")
            for cid, result in emergence_results['per_client_emergence'].items():
                logger.info(f"  Client {cid}: unseen_acc={result['unseen_accuracy']:.4f} "
                           f"({result['unseen_correct']}/{result['unseen_total']}), "
                           f"seen_acc={result['seen_accuracy']:.4f}, "
                           f"local_classes={result['local_seen_classes']}")
            
            # Knowledge transfer matrix
            logger.info("\nKnowledge Transfer Matrix (row=receiver, col=source):")
            transfer_matrix = emergence_results['knowledge_transfer_matrix']
            for i, row in enumerate(transfer_matrix):
                logger.info(f"  Client {i}: {row}")
            
            # Save emergence data if result_dir is set
            if hasattr(self.config, 'result_dir') and self.config.result_dir:
                self.save_emergence_data(emergence_results, self.config.result_dir)
        
        # Per-client per-task accuracy at the FINAL evaluation (after the
        # last aggregation step). Shape: { client_id: [acc_task0, ..., acc_taskT] }
        per_client_per_task_final = {}
        if self.all_accs:
            for cid, accs_list in self.all_accs[-1].items():
                per_client_per_task_final[int(cid)] = [float(a) for a in accs_list]

        # Full per-task-phase per-client matrix:
        # per_client_per_task_history[phase][cid] = [acc_task0, ..., acc_task_phase]
        per_client_per_task_history = []
        for phase_dict in self.all_accs:
            phase_serialized = {}
            for cid, accs_list in phase_dict.items():
                phase_serialized[int(cid)] = [float(a) for a in accs_list]
            per_client_per_task_history.append(phase_serialized)

        # Bubble-matrix-friendly views derived from the same source:
        # per_client_diag_acc[cid][t] = accuracy of client cid on task t,
        #     measured right after task t finished (before subsequent stream).
        # per_client_final_acc[cid][t] = end-of-stream accuracy of client cid
        #     on task t, after the full continual stream + aggregation.
        per_client_diag_acc = {}
        per_client_final_acc = {}
        if self.all_accs:
            client_ids = sorted(self.all_accs[-1].keys())
            num_phases = len(self.all_accs)
            for cid in client_ids:
                diag = []
                for t in range(num_phases):
                    accs_t = self.all_accs[t]
                    if cid in accs_t and t < len(accs_t[cid]):
                        diag.append(float(accs_t[cid][t]))
                    else:
                        diag.append(0.0)
                per_client_diag_acc[str(cid)] = diag
                final_row = self.all_accs[-1].get(cid, [])
                per_client_final_acc[str(cid)] = [float(x) for x in final_row]

        return {
            'final_accuracy': final_accuracy,
            'forgetting_rate': forgetting_rate,
            'all_accuracies': self.all_acc,
            'all_forgetting': self.all_forget,
            'per_task_acc': per_task_acc,
            'per_task_forget': per_task_forget,
            'per_client_per_task_final': per_client_per_task_final,
            'per_client_per_task_history': per_client_per_task_history,
            'per_client_diag_acc': per_client_diag_acc,
            'per_client_final_acc': per_client_final_acc,
            'emergence_results': emergence_results,
        }
    
    def _update_clients_for_new_task(self, task: int):
        """Update clients with data for new task."""
        for i, client in enumerate(self.clients):
            client_id, train_data, test_data, label_info = read_user_data(
                i, self.data, self.config.dataset, task=task
            )
            client.next_task(train_data, test_data, label_info)
    
    def _update_available_labels(self):
        """Update available labels across all clients."""
        available_labels = set()
        available_labels_current = set()
        
        for c in self.clients:
            available_labels.update(c.classes_so_far)
            available_labels_current.update(c.current_labels)
        
        for c in self.clients:
            c.available_labels = list(available_labels)
            c.available_labels_current = list(available_labels_current)
    
    def _local_training(self, global_round: int, task: int) -> Dict:
        """
        Perform local training on all clients.
        
        Returns:
            Dictionary with training results from all clients
        """
        results = {
            'proto_locals': {},
            'radius_locals': {},
            'w_locals': [],
            'delta_models': {},
            'delta_controls': {}
        }
        
        start_time = time.time()
        
        for client in self.clients:
            # Prepare algorithm-specific arguments
            kwargs = {}
            
            if self.config.algorithm == 'SCAFFOLD':
                kwargs['server_control'] = self.server_control
                kwargs['client_control'] = self.client_control
            
            if self.config.algorithm in ['DCFCL', 'DynDFCL']:
                kwargs['proto_queue'] = self.proto_queue
                
                # Update client with global prototypes
                if self.proto_global:
                    client.prototype["global"] = copy.deepcopy(self.proto_global)
                    client.radius["global"] = copy.deepcopy(self.radius_global)
            
            # Local training
            train_result = client.train(global_round, task, **kwargs)
            
            # Collect results based on algorithm
            if self.config.algorithm in ['DCFCL', 'DynDFCL']:
                # Save prototypes
                radius, prototype, class_label = client.compute_prototypes()
                results['proto_locals'][client.id] = {
                    'sample_num': client.get_sample_number(),
                    'prototype': prototype,
                    'num_samples_class': train_result.get('num_sample_class', {})
                }
                results['radius_locals'][client.id] = {
                    'sample_num': client.get_sample_number(),
                    'radius': radius
                }
                
                # Update client's local prototypes
                client.prototype["local"].update(prototype)
                client.radius["local"] = radius
                
                # Insert into proto queue
                self.proto_queue.insert(
                    prototype, radius, 
                    train_result.get('num_sample_class', {})
                )
            
            # Save model weights
            results['w_locals'].append((
                client.get_sample_number(),
                copy.deepcopy(client.model.state_dict()),
                client.id
            ))
            
            # SCAFFOLD specific
            if self.config.algorithm == 'SCAFFOLD':
                results['delta_models'][client.id] = client.delta_model
                results['delta_controls'][client.id] = client.delta_control
                self.client_control[client.id] = client.client_control
            
            # For coalition formation
            client.compute_l2_norm()
        
        train_time = time.time() - start_time
        logger.debug(f"Local training completed in {train_time:.2f}s")
        
        return results
    
    def _server_aggregation(self, global_round: int, task: int, train_results: Dict):
        """Perform server-side aggregation."""
        algorithm = self.config.algorithm
        
        if algorithm == 'Local':
            # No aggregation for local training
            pass
        
        elif algorithm == 'SCAFFOLD':
            self._aggregate_scaffold(train_results)
        
        elif algorithm in ['FedAvg', 'FedLwF', 'FedProx']:
            self._aggregate_fedavg()
        
        elif algorithm in ['PerAvg', 'pFedMe']:
            self._aggregate_personalized()
        
        elif algorithm in ['DCFCL', 'DynDFCL', 'ClusterFL']:
            self._aggregate_dcfcl(train_results, global_round)
    
    def _aggregate_fedavg(self):
        """FedAvg aggregation."""
        self._zero_model_parameters(self.model)
        total_samples = sum(c.train_samples for c in self.clients)
        
        for client in self.clients:
            ratio = client.train_samples / total_samples
            self._add_parameters(client.model, ratio)
        
        self._broadcast_parameters()
    
    def _aggregate_scaffold(self, train_results: Dict):
        """SCAFFOLD aggregation with control variate update."""
        # Update global model
        delta_models = train_results['delta_models']
        
        state_dict = {}
        for name, param in self.model.state_dict().items():
            deltas = torch.stack([delta_models[c][name].to(param.device) for c in delta_models])
            if param.is_floating_point() or param.is_complex():
                mean_delta = deltas.mean(dim=0)
                state_dict[name] = param - self.config.glo_lr * mean_delta
            else:
                mean_delta = deltas.float().mean(dim=0)
                state_dict[name] = (param.float() - self.config.glo_lr * mean_delta).to(param.dtype)
        
        self.model.load_state_dict(state_dict)
        
        # Update server control
        delta_controls = train_results['delta_controls']
        for name, c in self.server_control.items():
            mean_ci = torch.stack([delta_controls[cid][name].cpu() for cid in delta_controls]).mean(dim=0)
            self.server_control[name] = c - mean_ci
        
        self._broadcast_parameters()
    
    def _aggregate_personalized(self):
        """Aggregation for personalized FL (PerAvg, pFedMe)."""
        if self.config.algorithm == 'pFedMe':
            # Save previous model
            previous = copy.deepcopy(self.model)
            self._aggregate_fedavg()
            
            # Blend with previous model
            for prev_p, new_p in zip(previous.parameters(), self.model.parameters()):
                new_p.data = (1 - self.config.beta) * prev_p.data + self.config.beta * new_p.data
        else:
            # Simple averaging for PerAvg
            self._zero_model_parameters(self.model)
            ratio = 1.0 / len(self.clients)
            for client in self.clients:
                self._add_parameters(client.model, ratio)
        
        self._broadcast_parameters()
    
    def _aggregate_dcfcl(self, train_results: Dict, global_round: int):
        """DCFCL aggregation with coalition formation.
        
        Modes controlled by config.dcfcl_broadcast:
        - 1 (default): EMA + broadcast before coalition (legacy AFCL style)
        - 0: Pure coalition on local models (legacy FCL style)
        - 2: Hybrid — EMA global (no broadcast) + coalition on locals + blend
        """
        # Aggregate prototypes (always, for proto_aug)
        if self.config.algorithm == 'DCFCL':
            self.proto_global = self._aggregate_prototypes(train_results['proto_locals'])
            self.radius_global = self._aggregate_radius(train_results['radius_locals'])
        
        broadcast_mode = getattr(self.config, 'dcfcl_broadcast', 1)
        
        if broadcast_mode == 1:
            # Mode 1: EMA + broadcast (legacy AFCL)
            w_global = self._aggregate_weights(train_results['w_locals'])
            old_model = copy.deepcopy(self.model)
            self.model.load_state_dict(w_global)
            gw = self.config.global_weight
            for old_param, new_param in zip(old_model.parameters(), self.model.parameters()):
                new_param.data = old_param.data * (1 - gw) + new_param.data.clone() * gw
            
            # Save differ before broadcast (set_parameters resets it)
            saved_differs = {c.id: c.differ.copy() for c in self.clients}
            self._broadcast_parameters()
            for c in self.clients:
                c.differ = saved_differs[c.id]
        elif broadcast_mode == 2:
            # Mode 2: Hybrid — compute EMA global but do NOT broadcast
            # Local models stay diverse for meaningful coalition formation
            w_global = self._aggregate_weights(train_results['w_locals'])
            old_model = copy.deepcopy(self.model)
            self.model.load_state_dict(w_global)
            gw = self.config.global_weight
            for old_param, new_param in zip(old_model.parameters(), self.model.parameters()):
                new_param.data = old_param.data * (1 - gw) + new_param.data.clone() * gw
            # self.model now holds EMA global, but clients keep local models
        # else (mode 0): no EMA, no broadcast, locals stay as-is
        
        # Compute similarity matrix for coalition formation
        self.similarity_matrix = self._compute_similarity_matrix()
        
        # Update pay table
        self._update_pay_table()
        
        # Form coalitions
        if global_round == 0:
            self._form_coalition_initial()
        else:
            self._form_coalition_dynamic()
        
        logger.info(f"Coalitions: {self.unions}")
        
        # Compute directed collaboration matrix if enabled
        if getattr(self.config, 'directed_collaboration', False):
            current_task = self.clients[0].current_task if self.clients else 0
            self.directed_matrix = self._compute_directed_collaboration_matrix(current_task)
            logger.info(f"Directed collaboration matrix computed (mode={self.config.directed_mode})")
        
        # Create coalition models and distribute
        self._aggregate_coalitions()
        
        if broadcast_mode == 2:
            # Hybrid: blend coalition model with EMA global before distributing
            ema_blend = getattr(self.config, 'ema_blend', 0.5)
            for union_id in self.coalition_models:
                coal_model = self.coalition_models[union_id]
                for coal_p, ema_p in zip(coal_model.parameters(), self.model.parameters()):
                    coal_p.data = (1 - ema_blend) * coal_p.data + ema_blend * ema_p.data
        
        self._distribute_coalition_models()
    
    def _aggregate_weights(self, w_locals: List) -> Dict:
        """Aggregate model weights."""
        total_samples = sum(w[0] for w in w_locals)
        
        averaged_params = copy.deepcopy(w_locals[0][1])
        for key in averaged_params:
            averaged_params[key] = sum(
                w[1][key] * (w[0] / total_samples) for w in w_locals
            )
        
        return averaged_params
    
    def _aggregate_prototypes(self, proto_locals: Dict) -> Dict:
        """Aggregate client prototypes."""
        global_classes = set()
        for client_id in proto_locals:
            global_classes.update(proto_locals[client_id]['prototype'].keys())
        
        proto_global = {k: np.zeros(self.config.feature_size) for k in global_classes}
        weight_sums = {k: 0 for k in global_classes}
        
        for client_id in proto_locals:
            local_proto = proto_locals[client_id]['prototype']
            num_samples = proto_locals[client_id].get('num_samples_class', {})
            
            for cls in global_classes:
                if cls in local_proto and not np.all(local_proto[cls] == 0):
                    w = num_samples.get(cls, 1)
                    proto_global[cls] += local_proto[cls] * w
                    weight_sums[cls] += w
        
        for cls in global_classes:
            if weight_sums[cls] > 0:
                proto_global[cls] /= weight_sums[cls]
        
        # EMA with previous prototypes
        if self.proto_global:
            ema = self.config.ema_global
            for cls in self.proto_global:
                if cls in proto_global:
                    proto_global[cls] = ema * proto_global[cls] + (1 - ema) * self.proto_global[cls]
        
        return proto_global
    
    def _aggregate_radius(self, radius_locals: Dict) -> float:
        """Aggregate client radii."""
        total_samples = sum(r['sample_num'] for r in radius_locals.values())
        
        radius_global = sum(
            r['radius'] * (r['sample_num'] / total_samples)
            for r in radius_locals.values()
        )
        
        return radius_global

    # =========================================================================
    # Coalition Formation
    # =========================================================================
    
    def _compute_similarity_matrix(self) -> np.ndarray:
        """Compute similarity matrix between clients."""
        n = self.num_clients
        sw = self.config.sw
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Similarity based on per-round update direction.
                # Use the fresh local update from the current communication round.
                diff_i = self.clients[i].differ
                diff_j = self.clients[j].differ
                if np.allclose(diff_i, 0) or np.allclose(diff_j, 0):
                    sim_diff = 0.0
                else:
                    sim_diff = cosine_similarity(diff_i.reshape(1, -1), diff_j.reshape(1, -1))[0, 0]
                
                # Similarity based on current synchronized model parameters.
                vec_i = self.clients[i]._get_param_vector().reshape(1, -1)
                vec_j = self.clients[j]._get_param_vector().reshape(1, -1)
                sim_vec = cosine_similarity(vec_i, vec_j)[0, 0]
                
                # Combined similarity
                if self.config.algorithm == 'ClusterFL':
                    similarity = sim_diff  # Only gradient similarity
                else:
                    similarity = sim_diff + sw * sim_vec
                
                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity
        
        return sim_matrix
    
    def _compute_directed_collaboration_matrix(self, current_task: int) -> np.ndarray:
        """
        Compute directed collaboration matrix between clients.
        
        D[i][j] represents how much client i is willing to receive knowledge from client j.
        This is an asymmetric matrix: D[i][j] != D[j][i]
        
        The directed collaboration captures:
        1. Gradient alignment: Does j's update direction help i's optimization?
        2. Task relevance: Does j's knowledge benefit i's current learning?
        3. Knowledge transfer potential: Can j's experience improve i's performance?
        
        Args:
            current_task: Current task index.
            
        Returns:
            Asymmetric collaboration matrix of shape (n, n).
        """
        n = self.num_clients
        mode = getattr(self.config, 'directed_mode', 'gradient')
        
        # Initialize directed collaboration matrix
        directed_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-collaboration is always 1.0 (full trust in own model)
                    directed_matrix[i][j] = 1.0
                    continue
                
                # Compute directed collaboration score based on mode
                if mode == 'gradient':
                    score = self._compute_gradient_directed_score(i, j)
                elif mode == 'task_aware':
                    score = self._compute_task_aware_directed_score(i, j, current_task)
                elif mode == 'hybrid':
                    grad_score = self._compute_gradient_directed_score(i, j)
                    task_score = self._compute_task_aware_directed_score(i, j, current_task)
                    score = 0.5 * grad_score + 0.5 * task_score
                else:
                    # Default to symmetric similarity
                    score = self.similarity_matrix[i, j]
                
                directed_matrix[i][j] = score
        
        return directed_matrix
    
    def _compute_gradient_directed_score(self, i: int, j: int) -> float:
        """
        Compute gradient-based directed collaboration score.
        
        Score measures how much j's gradient update can help i's optimization.
        High score when:
        - j's gradient aligns with i's gradient (both moving in similar direction)
        - j has larger gradient magnitude (more informative update)
        
        Args:
            i: Target client (receiver)
            j: Source client (sender)
            
        Returns:
            Directed collaboration score (can be negative if harmful).
        """
        diff_i = self.clients[i].differ
        diff_j = self.clients[j].differ
        
        if np.allclose(diff_i, 0) or np.allclose(diff_j, 0):
            return 0.0
        
        # Cosine similarity of gradients
        cos_sim = cosine_similarity(
            diff_i.reshape(1, -1), 
            diff_j.reshape(1, -1)
        )[0, 0]
        
        # Gradient magnitude ratio: prefer larger gradients from j
        # This captures "informativeness" of j's update
        norm_i = np.linalg.norm(diff_i)
        norm_j = np.linalg.norm(diff_j)
        
        if norm_i < 1e-8:
            mag_ratio = 1.0
        else:
            # Softly encourage learning from clients with larger updates
            mag_ratio = np.sqrt(norm_j / (norm_i + 1e-8))
            mag_ratio = np.clip(mag_ratio, 0.5, 2.0)  # Bound the ratio
        
        # Directed score: alignment * relative magnitude
        # If cos_sim < 0, j's update is harmful to i
        directed_score = cos_sim * mag_ratio
        
        return directed_score
    
    def _compute_task_aware_directed_score(self, i: int, j: int, current_task: int) -> float:
        """
        Compute task-aware directed collaboration score.
        
        Score measures how relevant j's knowledge is for i's learning.
        High score when:
        - j has learned classes that i needs to learn
        - j has more diverse or complementary knowledge
        
        Args:
            i: Target client (receiver)
            j: Source client (sender)
            current_task: Current task index
            
        Returns:
            Directed collaboration score in [0, 1].
        """
        # Get classes each client has learned
        classes_i = set(self.clients[i].learned_classes if hasattr(self.clients[i], 'learned_classes') else [])
        classes_j = set(self.clients[j].learned_classes if hasattr(self.clients[j], 'learned_classes') else [])
        
        # Get current task classes for client i
        current_classes_i = set(self.clients[i].current_classes if hasattr(self.clients[i], 'current_classes') else [])
        
        if not classes_j:
            return 0.0
        
        # Factor 1: Class overlap - j has classes that i is learning
        overlap_with_current = len(classes_j & current_classes_i)
        if current_classes_i:
            relevance = overlap_with_current / len(current_classes_i)
        else:
            relevance = 0.0
        
        # Factor 2: Knowledge diversity - j has complementary knowledge
        # High score if j has classes that i hasn't learned yet
        new_knowledge = len(classes_j - classes_i)
        diversity = new_knowledge / (len(classes_j) + 1e-8)
        
        # Factor 3: Experience level - prefer learning from more experienced clients
        # (clients that have seen more tasks/classes)
        experience_ratio = len(classes_j) / (len(classes_i) + 1e-8)
        experience_score = np.tanh(experience_ratio - 1)  # Positive if j more experienced
        
        # Combine factors
        task_score = 0.4 * relevance + 0.3 * diversity + 0.3 * max(0, experience_score)
        
        return task_score

    def _update_pay_table(self):
        """Update payoff table for coalition formation."""
        for part, payoffs in self.pay_table.items():
            for union in part:
                if len(union) > 1:
                    for c_id in union:
                        payoffs[c_id] = self._compute_coalition_reward(union, c_id)
                else:
                    payoffs[list(union)[0]] = 0
    
    def _compute_coalition_reward(self, union: tuple, c_id: int) -> float:
        """Compute reward for client in coalition."""
        others = list(set(union) - {c_id})
        
        if not others:
            return 0.0
        
        # Cross-correlation term
        cross_term = 0.0
        for i in range(len(others) - 1):
            for j in range(i + 1, len(others)):
                ci, cj = others[i], others[j]
                l2_i = self.clients[ci].l2_norm or 1e-8
                l2_j = self.clients[cj].l2_norm or 1e-8
                cross_term += 2 * l2_i * l2_j * self.similarity_matrix[ci, cj]
        
        # Correlation with target client
        target_term = 0.0
        norm_term = 0.0
        for i in others:
            l2_i = self.clients[i].l2_norm or 1e-8
            target_term += l2_i * self.similarity_matrix[i, c_id]
            norm_term += l2_i ** 2
        
        # Reward = cosine similarity between aggregated gradient and target
        denominator = np.sqrt(cross_term + norm_term)
        if denominator < 1e-8:
            return 0.0
        
        return target_term / denominator
    
    def _form_coalition_initial(self):
        """Initial coalition formation."""
        trans_list = []
        single_rewards = [0] * self.num_clients
        
        for idx, (state, payoffs) in enumerate(self.pay_table.items()):
            next_state = self._single_step_transfer(idx, state, payoffs, single_rewards)
            trans_list.append((idx, next_state))
        
        # Find absorbing states
        absorbing = [i for i, (s, n) in enumerate(trans_list) if s == n]
        
        # Select best absorbing state
        pay_list = list(self.pay_table.keys())
        if len(absorbing) > 1:
            best_welfare = -float('inf')
            best_state = absorbing[0]
            for s_id in absorbing:
                welfare = sum(self.pay_table[pay_list[s_id]])
                if welfare > best_welfare:
                    best_welfare = welfare
                    best_state = s_id
            self.stable_state = best_state
        else:
            self.stable_state = absorbing[0] if absorbing else 0
        
        self.unions = pay_list[self.stable_state]
    
    def _form_coalition_dynamic(self):
        """Dynamic coalition update."""
        pay_list = list(self.pay_table.items())
        
        current_state = self.stable_state
        state, payoffs = pay_list[current_state]
        
        # Try to find better state
        next_state = self._single_step_transfer(current_state, state, payoffs, [0] * self.num_clients)
        
        iterations = 0
        while next_state != current_state and iterations < 100:
            current_state = next_state
            state, payoffs = pay_list[current_state]
            next_state = self._single_step_transfer(current_state, state, payoffs, [0] * self.num_clients)
            iterations += 1
        
        self.stable_state = next_state
        self.unions = pay_list[self.stable_state][0]
    
    def _single_step_transfer(self, s_id: int, state: tuple, payoffs: List, 
                               single_rewards: List) -> int:
        """Find next state in coalition formation (matches official code)."""
        next_state = s_id
        
        for idx, (part, part_payoffs) in enumerate(self.pay_table.items()):
            if part != state:
                # Case 1: Check if a singleton client can benefit from leaving
                # (official code: if a client has negative payoff in current state,
                #  they prefer to be alone with payoff 0)
                for i in range(self.num_clients):
                    if (i,) in part:
                        if single_rewards[i] > payoffs[i]:
                            return idx
                
                # Case 2: Check if a coalition blocks current state
                for union in part:
                    union_reward = [part_payoffs[u] for u in union]
                    state_reward = [payoffs[u] for u in union]
                    
                    if (all(ur >= sr for ur, sr in zip(union_reward, state_reward)) and
                        any(ur > sr for ur, sr in zip(union_reward, state_reward))):
                        return idx
        
        return next_state
    
    def _aggregate_coalitions(self):
        """Create aggregated models for each coalition.
        
        If directed_collaboration is enabled, each client gets a personalized
        aggregated model based on the directed collaboration matrix.
        Otherwise, all clients in a coalition share the same aggregated model.
        """
        # Check if directed collaboration is enabled
        use_directed = getattr(self.config, 'directed_collaboration', False)
        
        if use_directed:
            self._aggregate_coalitions_directed()
        else:
            self._aggregate_coalitions_standard()
    
    def _aggregate_coalitions_standard(self):
        """Standard coalition aggregation - all members share one model."""
        self.coalition_models = {}
        
        for union_id, union in enumerate(self.unions):
            union_model = copy.deepcopy(self.model)
            self._zero_model_parameters(union_model)
            
            total_samples = sum(self.clients[c].train_samples for c in union)
            
            for client_id in union:
                client = self.clients[client_id]
                ratio = client.train_samples / total_samples
                
                for agg_p, client_p in zip(union_model.parameters(), client.model.parameters()):
                    agg_p.data += client_p.data * ratio
            
            self.coalition_models[union_id] = union_model
    
    def _aggregate_coalitions_directed(self):
        """
        Directed coalition aggregation - each client gets personalized model.
        
        For each client i in a coalition:
        1. Filter coalition members based on directed collaboration scores
        2. Apply directed weights (asymmetric) instead of sample-based weights
        3. Create a personalized aggregated model for client i
        
        The aggregated model for client i is:
            model_i = self_weight * model_i + (1-self_weight) * sum(w_ij * model_j)
        
        where w_ij is based on directed collaboration score D[i][j].
        """
        self.coalition_models = {}
        self.client_personalized_models = {}
        
        threshold = getattr(self.config, 'directed_threshold', 0.0)
        temperature = getattr(self.config, 'directed_temperature', 1.0)
        self_weight = getattr(self.config, 'directed_self_weight', 0.5)
        
        # Build reverse mapping: client_id -> union_id
        client_to_union = {}
        for union_id, union in enumerate(self.unions):
            for client_id in union:
                client_to_union[client_id] = union_id
        
        for union_id, union in enumerate(self.unions):
            for client_i in union:
                # Create personalized model for client i
                personalized_model = copy.deepcopy(self.model)
                self._zero_model_parameters(personalized_model)
                
                # Get directed collaboration scores for client i
                # Only consider clients in the same coalition
                trusted_clients = []
                raw_scores = []
                
                for client_j in union:
                    if client_j == client_i:
                        continue  # Self handled separately
                    
                    # Get directed score: how much i trusts j
                    score = self.directed_matrix[client_i][client_j]
                    
                    if score > threshold:
                        trusted_clients.append(client_j)
                        raw_scores.append(score)
                
                # If no trusted clients, just use own model
                if not trusted_clients:
                    for p_pers, p_self in zip(personalized_model.parameters(), 
                                               self.clients[client_i].model.parameters()):
                        p_pers.data.copy_(p_self.data)
                    self.client_personalized_models[client_i] = personalized_model
                    continue
                
                # Compute softmax weights from directed scores
                raw_scores = np.array(raw_scores) / temperature
                # Avoid numerical overflow
                raw_scores = raw_scores - np.max(raw_scores)
                exp_scores = np.exp(raw_scores)
                
                # Also consider sample sizes
                sample_weights = np.array([self.clients[c].train_samples for c in trusted_clients])
                sample_weights = sample_weights / sample_weights.sum()
                
                # Combined weights: directed score * sample proportion
                combined_scores = exp_scores * sample_weights
                weights = combined_scores / combined_scores.sum()
                
                # Aggregate from trusted clients
                aggregated_from_others = copy.deepcopy(self.model)
                self._zero_model_parameters(aggregated_from_others)
                
                for j, client_j in enumerate(trusted_clients):
                    w_j = weights[j]
                    for agg_p, client_p in zip(aggregated_from_others.parameters(),
                                                self.clients[client_j].model.parameters()):
                        agg_p.data += client_p.data * w_j
                
                # Final personalized model: blend self with aggregated
                for p_pers, p_self, p_agg in zip(personalized_model.parameters(),
                                                  self.clients[client_i].model.parameters(),
                                                  aggregated_from_others.parameters()):
                    p_pers.data = self_weight * p_self.data + (1 - self_weight) * p_agg.data
                
                self.client_personalized_models[client_i] = personalized_model
        
        # Also create standard coalition models for compatibility
        self._aggregate_coalitions_standard()
    
    def _distribute_coalition_models(self):
        """Distribute coalition models to clients.
        
        If directed collaboration is enabled, distribute personalized models.
        Otherwise, distribute standard coalition models.
        """
        use_directed = getattr(self.config, 'directed_collaboration', False)
        
        if use_directed and hasattr(self, 'client_personalized_models'):
            # Distribute personalized models
            for client_id, model in self.client_personalized_models.items():
                self.clients[client_id].set_parameters(model)
        else:
            # Standard distribution
            for union_id, union in enumerate(self.unions):
                for client_id in union:
                    self.clients[client_id].set_parameters(self.coalition_models[union_id])
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _zero_model_parameters(self, model):
        """Set all model parameters to zero."""
        for param in model.parameters():
            param.data.zero_()
    
    def _add_parameters(self, source_model, ratio: float):
        """Add weighted parameters from source to global model."""
        for glob_param, src_param in zip(self.model.parameters(), source_model.parameters()):
            glob_param.data += src_param.data * ratio
    
    def _broadcast_parameters(self):
        """Broadcast global model parameters to all clients."""
        for client in self.clients:
            client.set_parameters(self.model)
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def _evaluate(self) -> Tuple[Dict, float, Dict]:
        """Evaluate all clients on per-task accuracy."""
        accs = {}
        samples = {}
        total_correct = 0
        total_samples = 0
        
        for client in self.clients:
            task_correct, _, task_samples = client.test_per_task()
            
            accs[client.id] = [c / s if s > 0 else 0 
                              for c, s in zip(task_correct, task_samples)]
            samples[client.id] = task_samples
            
            total_correct += sum(task_correct)
            total_samples += sum(task_samples)
        
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return accs, avg_acc, samples
    
    def _compute_forgetting(self) -> float:
        """Compute forgetting rate."""
        if len(self.all_accs) < 2:
            return 0.0
        
        total_forgetting = 0.0
        total_samples = 0
        
        for client in self.clients:
            cid = client.id
            current_accs = self.all_accs[-1].get(cid, [])
            
            for task in range(len(current_accs) - 1):
                # Best accuracy on this task so far
                best_acc = max(
                    self.all_accs[t].get(cid, [0] * (task + 1))[task]
                    for t in range(task, len(self.all_accs) - 1)
                )
                
                # Current accuracy
                current_acc = current_accs[task] if task < len(current_accs) else 0
                
                # Forgetting
                forgetting = max(0, best_acc - current_acc)
                total_forgetting += forgetting
                total_samples += 1
        
        return total_forgetting / total_samples if total_samples > 0 else 0.0
    
    def _compute_per_task_accuracy(self, accs: Dict) -> List[float]:
        """Compute average accuracy per task across all clients."""
        if not accs:
            return []
        num_tasks_so_far = max(len(v) for v in accs.values())
        per_task = []
        for t in range(num_tasks_so_far):
            task_accs = [accs[cid][t] for cid in accs if t < len(accs[cid])]
            per_task.append(sum(task_accs) / len(task_accs) if task_accs else 0.0)
        return per_task
    
    def _compute_detailed_metrics(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Compute per-task accuracy and per-task forgetting after each task phase.
        
        Returns:
            per_task_acc[phase][task]: accuracy on task after training phase
            per_task_forget[phase][task]: forgetting on task after training phase
        """
        per_task_acc = []  # per_task_acc[phase] = [acc_task0, acc_task1, ...]
        per_task_forget = []  # per_task_forget[phase] = [forget_task0, ...]
        
        for phase_idx, accs in enumerate(self.all_accs):
            # Aggregate per-task accuracy across clients
            phase_acc = self._compute_per_task_accuracy(accs)
            per_task_acc.append(phase_acc)
            
            # Compute per-task forgetting (for tasks before current)
            phase_forget = []
            for t in range(len(phase_acc)):
                if t < phase_idx:  # Only past tasks can be forgotten
                    # Best accuracy on task t across phases [t .. phase_idx-1]
                    best = 0.0
                    for prev_phase in range(t, phase_idx):
                        prev_acc = self._compute_per_task_accuracy(self.all_accs[prev_phase])
                        if t < len(prev_acc):
                            best = max(best, prev_acc[t])
                    current = phase_acc[t] if t < len(phase_acc) else 0.0
                    phase_forget.append(max(0.0, best - current))
                else:
                    phase_forget.append(0.0)  # Current task: no forgetting
            per_task_forget.append(phase_forget)
        
        return per_task_acc, per_task_forget
    
    # =========================================================================
    # Emergence Evaluation
    # =========================================================================
    
    def evaluate_emergence(self) -> Dict:
        """
        Evaluate emergence phenomenon across all clients.
        
        Emergence is the phenomenon where clients can correctly predict classes
        they have NEVER seen in their local training data. This knowledge must
        have been transferred from other clients through federated learning.
        
        Returns:
            Dictionary containing:
            - 'global_emergence_rate': Overall accuracy on unseen classes
            - 'per_client_emergence': Dict of per-client emergence metrics
            - 'total_emergence_samples': Total number of correct predictions on unseen classes
            - 'emergence_by_class': Per-class emergence breakdown
            - 'knowledge_transfer_matrix': Matrix showing knowledge flow between clients
        """
        import numpy as np
        
        # Collect all test data and labels per task (global view)
        all_test_data = []
        all_labels_per_task = []
        
        # Build global test data from any client (they all share the same test structure)
        if self.clients:
            reference_client = self.clients[0]
            all_test_data = reference_client.test_data_per_task
            
            # Get labels for each task
            for task_data in all_test_data:
                task_labels = set()
                for _, y in DataLoader(task_data, batch_size=100):
                    task_labels.update(y.tolist())
                all_labels_per_task.append(list(task_labels))
        
        # Evaluate emergence for each client
        per_client_emergence = {}
        all_emergence_samples = []
        global_unseen_correct = 0
        global_unseen_total = 0
        global_seen_correct = 0
        global_seen_total = 0
        emergence_by_class = {}
        
        for client in self.clients:
            emergence_result = client.evaluate_emergence(all_test_data, all_labels_per_task)
            per_client_emergence[client.id] = emergence_result
            
            global_unseen_correct += emergence_result['unseen_correct']
            global_unseen_total += emergence_result['unseen_total']
            global_seen_correct += emergence_result['seen_correct']
            global_seen_total += emergence_result['seen_total']
            
            # Collect emergence samples
            all_emergence_samples.extend(emergence_result['emergence_samples'])
            
            # Aggregate per-class emergence
            for cls, stats in emergence_result['per_class_emergence'].items():
                if cls not in emergence_by_class:
                    emergence_by_class[cls] = {'correct': 0, 'total': 0, 'clients': []}
                emergence_by_class[cls]['correct'] += stats['correct']
                emergence_by_class[cls]['total'] += stats['total']
                if stats['correct'] > 0:
                    emergence_by_class[cls]['clients'].append(client.id)
        
        # Compute knowledge transfer matrix
        # transfer_matrix[i][j] = classes that client i learned from client j's tasks
        n = len(self.clients)
        transfer_matrix = np.zeros((n, n))
        
        for i, client in enumerate(self.clients):
            client_seen = set(client.classes_so_far)
            for j, other in enumerate(self.clients):
                if i != j:
                    other_unique = set(other.classes_so_far) - client_seen
                    # Check if client i can predict classes unique to client j
                    transfer_count = 0
                    for cls in other_unique:
                        if cls in per_client_emergence[client.id]['per_class_emergence']:
                            cls_stats = per_client_emergence[client.id]['per_class_emergence'][cls]
                            if cls_stats['correct'] > 0:
                                transfer_count += 1
                    transfer_matrix[i][j] = transfer_count
        
        # Compute emergence rate
        global_emergence_rate = global_unseen_correct / global_unseen_total if global_unseen_total > 0 else 0.0
        seen_accuracy = global_seen_correct / global_seen_total if global_seen_total > 0 else 0.0
        
        return {
            'global_emergence_rate': global_emergence_rate,
            'seen_accuracy': seen_accuracy,
            'global_unseen_correct': global_unseen_correct,
            'global_unseen_total': global_unseen_total,
            'global_seen_correct': global_seen_correct,
            'global_seen_total': global_seen_total,
            'per_client_emergence': per_client_emergence,
            'total_emergence_samples': len(all_emergence_samples),
            'emergence_samples': all_emergence_samples,
            'emergence_by_class': emergence_by_class,
            'knowledge_transfer_matrix': transfer_matrix.tolist(),
            'all_labels_per_task': all_labels_per_task,
        }
    
    def save_emergence_data(self, emergence_results: Dict, output_dir: str):
        """
        Save emergence samples and metadata for later analysis.
        
        Args:
            emergence_results: Results from evaluate_emergence()
            output_dir: Directory to save data
        """
        import os
        import json
        import pickle
        
        emergence_dir = os.path.join(output_dir, 'emergence_analysis')
        os.makedirs(emergence_dir, exist_ok=True)
        
        # Save metadata (JSON)
        metadata = {
            'global_emergence_rate': emergence_results['global_emergence_rate'],
            'seen_accuracy': emergence_results['seen_accuracy'],
            'global_unseen_correct': emergence_results['global_unseen_correct'],
            'global_unseen_total': emergence_results['global_unseen_total'],
            'global_seen_correct': emergence_results['global_seen_correct'],
            'global_seen_total': emergence_results['global_seen_total'],
            'total_emergence_samples': emergence_results['total_emergence_samples'],
            'knowledge_transfer_matrix': emergence_results['knowledge_transfer_matrix'],
            'all_labels_per_task': emergence_results['all_labels_per_task'],
            'emergence_by_class': {
                str(k): v for k, v in emergence_results['emergence_by_class'].items()
            },
            'per_client_summary': {},
        }
        
        # Per-client summary
        for cid, result in emergence_results['per_client_emergence'].items():
            metadata['per_client_summary'][str(cid)] = {
                'unseen_accuracy': result['unseen_accuracy'],
                'seen_accuracy': result['seen_accuracy'],
                'unseen_total': result['unseen_total'],
                'unseen_correct': result['unseen_correct'],
                'local_seen_classes': result['local_seen_classes'],
                'num_emergence_samples': len(result['emergence_samples']),
            }
        
        with open(os.path.join(emergence_dir, 'emergence_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save emergence samples (pickle for numpy arrays)
        samples_to_save = []
        for sample in emergence_results['emergence_samples']:
            samples_to_save.append({
                'sample': sample['sample'],  # numpy array
                'true_label': sample['true_label'],
                'pred_label': sample['pred_label'],
                'confidence': sample['confidence'],
                'task_idx': sample['task_idx'],
                'client_id': sample['client_id'],
                'local_seen_classes': sample['local_seen_classes'],
            })
        
        with open(os.path.join(emergence_dir, 'emergence_samples.pkl'), 'wb') as f:
            pickle.dump(samples_to_save, f)
        
        logger.info(f"Emergence data saved to {emergence_dir}")
        logger.info(f"  - Metadata: emergence_metadata.json")
        logger.info(f"  - Samples: emergence_samples.pkl ({len(samples_to_save)} samples)")
        
        return emergence_dir
