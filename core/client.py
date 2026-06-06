#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/client.py - backward-compatibility shim.

All federated-learning client implementations have moved to the
top-level FL_model/ package. Each algorithm lives in its own file
and subclasses FL_model.base_client.BaseClient.

This file re-exports Client / create_client so existing imports keep
working unchanged.
"""

# Re-exports for backward compatibility.
from FL_model.base_client import BaseClient as Client  # noqa: F401
from FL_model import create_client                     # noqa: F401

__all__ = ['Client', 'create_client']

import copy
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any

from .models import create_model
from .optimizers import ScaffoldOptimizer, PerAvgOptimizer, pFedMeOptimizer
from .replay_buffer import ReplayBuffer

logger = logging.getLogger('DCFCL.Client')


class Client:
    """
    Federated Learning Client for DCFCL.
    
    Implements local training with various FL algorithms including:
    - FedAvg, FedProx, FedLwF (Federated Learning without Forgetting)
    - SCAFFOLD (for variance reduction)
    - PerAvg, pFedMe (personalization)
    - DCFCL (our proposed method with coalition formation)
    
    Attributes:
        id: Client identifier
        config: Configuration object
        model: Local model
        train_data: Training dataset
        test_data: Test dataset
        classes_so_far: All classes seen so far
        current_labels: Labels for current task
    """
    
    def __init__(self, client_id: int, config, model, train_data, test_data, 
                 label_info: Dict, unique_labels: int):
        """
        Initialize client.
        
        Args:
            client_id: Unique client identifier
            config: Configuration object
            model: Model to copy for this client
            train_data: Training dataset
            test_data: Test dataset
            label_info: Dictionary with 'labels' and 'counts' keys
            unique_labels: Total number of unique labels
        """
        self.id = client_id
        self.config = config

        device_name = getattr(config, 'device', 'auto')
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device_name == 'cuda' and not torch.cuda.is_available():
            device_name = 'cpu'
        self.device = torch.device(device_name)
        
        # Create local model (deep copy)
        self.model = create_model(config)
        self.model.to(self.device)
        
        # Data attributes
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.unique_labels = unique_labels
        
        # Create data loaders
        self._setup_dataloaders()
        
        # Continual learning attributes
        self.classes_so_far: List[int] = list(label_info.get('labels', []))
        self.current_labels: List[int] = list(label_info.get('labels', []))
        self.classes_past_task: List[int] = []
        self.available_labels: List[int] = []
        self.available_labels_current: List[int] = []
        self.current_task: int = 0
        
        # Label tracking
        self.label_counts: Dict[int, int] = {i: 0 for i in range(unique_labels)}
        
        # Knowledge distillation (for continual learning)
        self.last_copy: Optional[nn.Module] = None
        self.if_last_copy: bool = False
        
        # Setup algorithm-specific components
        self._setup_optimizer()
        
        # For prototype-based methods (DCFCL)
        self.prototype = {"global": {}, "local": {}}
        self.radius = {"global": 0, "local": 0}
        self.feature_size = config.feature_size
        self.num_sample_class: Optional[Dict] = None
        
        # For gradient tracking
        self.param_vector = self._get_param_vector()
        self.differ = np.zeros_like(self.param_vector)
        self.l2_norm: Optional[float] = None
        
        # Test data tracking for continual learning
        self.test_data_so_far = list(test_data)
        self.test_data_per_task = [test_data]
        
        # Losses
        ls = getattr(config, 'label_smoothing', 0.0)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=ls)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # pFedMe specific
        self.local_model = copy.deepcopy(self.model) if config.algorithm == 'pFedMe' else None
        
        # Replay buffer for DynDFCL (DER/DER++)
        self.replay_buffer = None
        if config.algorithm == 'DynDFCL':
            self.replay_buffer = ReplayBuffer(config.buffer_size, self.device)
    
    def _setup_dataloaders(self):
        """Setup data loaders for training and testing."""
        self.trainloader = DataLoader(
            self.train_data, 
            batch_size=self.config.batch_size,
            shuffle=True, 
            drop_last=True
        )
        self.testloader = DataLoader(
            self.test_data,
            batch_size=self.config.batch_size,
            drop_last=False
        )
        self.iter_trainloader = iter(self.trainloader)
        
    def _setup_optimizer(self):
        """Setup optimizer based on algorithm."""
        if self.config.algorithm == 'SCAFFOLD':
            self.optimizer = ScaffoldOptimizer(
                self.model.parameters(),
                lr=self.config.scaffold_lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.algorithm == 'PerAvg':
            self.optimizer = PerAvgOptimizer(
                self.model.parameters(),
                lr=self.config.lr
            )
        elif self.config.algorithm == 'pFedMe':
            self.optimizer = pFedMeOptimizer(
                self.model.parameters(),
                lr=self.config.personal_lr,
                lamda=self.config.lamda
            )
        else:
            # Default: use model's built-in optimizer
            self.optimizer = self.model.classifier_optimizer
    
    def next_task(self, train_data, test_data, label_info: Dict):
        """
        Prepare client for next task in continual learning.
        
        Args:
            train_data: New training data
            test_data: New test data
            label_info: Label information for new task
        """
        # Save current model for knowledge distillation
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True
        
        # Update data
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        
        # Update data loaders
        self._setup_dataloaders()
        
        # Update label tracking
        self.classes_past_task = list(self.classes_so_far)
        self.classes_so_far.extend(label_info.get('labels', []))
        self.current_labels = list(label_info.get('labels', []))
        
        # Update test data for evaluation
        self.test_data_so_far.extend(test_data)
        self.test_data_per_task.append(test_data)
        
        self.current_task += 1
    
    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        Perform local training.
        
        Args:
            glob_iter: Global iteration number
            task: Current task index
            **kwargs: Algorithm-specific arguments
            
        Returns:
            Dictionary with training results
        """
        algorithm = self.config.algorithm
        
        if algorithm in ['FedAvg', 'Local']:
            return self._train_fedavg()
        elif algorithm == 'FedProx':
            return self._train_fedprox()
        elif algorithm == 'FedLwF':
            return self._train_lwf()
        elif algorithm == 'SCAFFOLD':
            return self._train_scaffold(kwargs.get('server_control'), kwargs.get('client_control'))
        elif algorithm == 'PerAvg':
            return self._train_peravg()
        elif algorithm == 'pFedMe':
            return self._train_pfedme()
        elif algorithm == 'DCFCL':
            return self._train_dcfcl(kwargs.get('proto_queue'))
        elif algorithm == 'DynDFCL':
            return self._train_dyndfcl(kwargs.get('proto_queue'))
        else:
            return self._train_fedavg()  # Default
    
    def _get_next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next training batch."""
        try:
            X, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            X, y = next(self.iter_trainloader)
        return X.to(self.device), y.to(self.device)
    
    # =========================================================================
    # Training Methods for Different Algorithms
    # =========================================================================
    
    def _train_fedavg(self) -> Dict[str, Any]:
        """Standard FedAvg local training."""
        initial_params = self.param_vector.copy()
        self.model.train()
        total_loss = 0.0
        
        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            loss = self._train_step_basic(x, y)
            total_loss += loss
        
        # Update differ for coalition-based algorithms (ClusterFL)
        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params
            
        return {'loss': total_loss / self.config.local_epochs}
    
    def _train_fedprox(self) -> Dict[str, Any]:
        """FedProx training with proximal term."""
        # Save global model for proximal term
        global_model = copy.deepcopy(self.model)
        self.model.train()
        total_loss = 0.0
        
        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            
            # Forward pass
            _, _, logits = self.model(x)
            loss = self.ce_loss(logits, y.long())
            
            # Add proximal term (matches original: L2 norm, not squared)
            proximal_term = 0.0
            for w, w_g in zip(self.model.parameters(), global_model.parameters()):
                proximal_term += (w - w_g).norm(2)
            loss += (self.config.mu / 2) * proximal_term
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return {'loss': total_loss / self.config.local_epochs}
    
    def _train_lwf(self) -> Dict[str, Any]:
        """Learning without Forgetting (LwF) training."""
        self.model.train()
        total_loss = 0.0
        
        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            
            # Current model output
            output, _, logits = self.model(x)
            class_loss = self.ce_loss(logits, y.long())
            
            # Knowledge distillation from previous model
            if self.if_last_copy and self.current_task > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                
                # KD loss (matches original implementation)
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)
                
                loss = class_loss + self.config.alpha * kd_loss
            else:
                loss = class_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return {'loss': total_loss / self.config.local_epochs}
    
    def _train_scaffold(self, server_control, client_control) -> Dict[str, Any]:
        """SCAFFOLD training with variance reduction."""
        # client_control is {client_id: control_dict}; extract this client's control
        my_control = client_control[self.id]
        
        global_model = copy.deepcopy(self.model)
        self.model.train()
        total_loss = 0.0
        
        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            
            _, _, logits = self.model(x)
            loss = self.ce_loss(logits, y.long())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(server_control, my_control)
            
            total_loss += loss.item()
        
        # Compute delta model and control
        self.delta_model = self._compute_delta_model(global_model, self.model)
        self.client_control, self.delta_control = self._update_local_control(
            self.delta_model, server_control, my_control
        )
        
        return {'loss': total_loss / self.config.local_epochs}
    
    def _train_peravg(self) -> Dict[str, Any]:
        """Per-FedAvg training."""
        self.model.train()
        total_loss = 0.0
        
        for _ in range(self.config.local_epochs):
            # First step: standard training
            temp_model = copy.deepcopy(self.model)
            x, y = self._get_next_batch()
            self._train_step_basic(x, y)
            
            # Second step: personalized update
            x, y = self._get_next_batch()
            _, _, logits = self.model(x)
            loss = self.ce_loss(logits, y.long())
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Restore and apply personalized step
            for old_p, new_p in zip(self.model.parameters(), temp_model.parameters()):
                old_p.data = new_p.data.clone()
            self.optimizer.step(beta=self.config.beta)
            
            total_loss += loss.item()
            
        return {'loss': total_loss / self.config.local_epochs}
    
    def _train_pfedme(self) -> Dict[str, Any]:
        """pFedMe personalized training."""
        self.model.train()
        total_loss = 0.0
        
        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            
            # K personalization steps
            for _ in range(self.config.K):
                _, _, logits = self.model(x)
                loss = self.ce_loss(logits, y.long())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.local_model.parameters())
            
            # Update local model
            for local_w, w in zip(self.local_model.parameters(), self.model.parameters()):
                local_w.data = local_w.data - self.config.lamda * self.config.lr * (local_w.data - w.data)
            
            total_loss += loss.item()
        
        # Copy local model to model for aggregation
        self.set_parameters(self.local_model)
        
        return {'loss': total_loss / self.config.local_epochs}
    
    def _train_dcfcl(self, proto_queue=None) -> Dict[str, Any]:
        """
        DCFCL training with prototype augmentation and coalition formation.
        
        This is the main training method for the proposed algorithm.
        DCFCL training per paper Section 4.1:
        L_k = L_class + lambda * L_dis + lambda_proto_aug * L_proto
        
        Paper Eq.6, Table 3: optimal lambda_kd = 0.2
        Official code omits KD, but paper ablation (Table 2) shows KD helps.
        """
        self.model.train()
        total_loss = 0.0
        num_sample_class = {k: 0 for k in range(self.config.num_classes)}
        
        # Save teacher model for KD — updated EVERY round, matching official code
        # (paper Eq.4: teacher is model from round τ-1, not just from task transition)
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True
        
        # Record initial params — differ will be total model change (official code accumulates all steps)
        initial_params = self._get_param_vector().copy()
        
        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            
            # Track samples per class
            for label in y.tolist():
                num_sample_class[label] += 1
            
            # Forward pass
            output, features, logits = self.model(x)
            
            # Classification loss (use logits, not softmax output)
            ce_loss = self.ce_loss(logits, y.long())
            
            # Knowledge distillation loss (paper Eq.4, Section 4.1)
            # Maintains classifier feature space consistency to identify cooperators
            # Official code applies KD every round (including task 0); we follow the
            # paper interpretation that KD prevents drift when learning NEW tasks.
            kd_loss = 0.0
            if self.if_last_copy and self.current_task > 0 and self.config.lambda_kd > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)
            
            # Prototype augmentation loss
            proto_loss = 0.0
            if proto_queue is not None and self.config.lambda_proto_aug > 0:
                proto_loss = self._compute_proto_aug_loss(proto_queue)
            
            # Total loss: paper Eq.6 with lambda_kd (NOT alpha)
            loss = ce_loss + self.config.lambda_kd * kd_loss + self.config.lambda_proto_aug * proto_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Track total model change across ALL training steps for coalition similarity.
        # Official code accumulates (param_before - param_after) each step, which is
        # equivalent to (initial_params - final_params). We use the positive direction.
        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params
        
        return {
            'loss': total_loss / self.config.local_epochs,
            'num_sample_class': num_sample_class
        }
    
    def _train_dyndfcl(self, proto_queue=None) -> Dict[str, Any]:
        """
        DynDFCL: DCFCL + Dark Experience Replay (DER++).
        
        Combines the original DCFCL losses (CE + KD + proto_aug) with
        DER++ replay losses:
          - alpha_der * MSE(model(x_buf), stored_logits)  (dark experience)
          - beta_der  * CE(model(x_buf), stored_labels)   (experience replay)
        
        The replay buffer persists across rounds and tasks so that replayed
        samples come from earlier tasks, directly combating forgetting.
        """
        self.model.train()
        total_loss = 0.0
        num_sample_class = {k: 0 for k in range(self.config.num_classes)}
        
        # Save teacher for KD (same as DCFCL)
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True
        
        initial_params = self._get_param_vector().copy()
        
        der_alpha = self.config.der_alpha
        der_beta = self.config.der_beta
        
        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            
            for label in y.tolist():
                num_sample_class[label] += 1
            
            # ---- Forward on current data ----
            output, features, logits = self.model(x)
            ce_loss = self.ce_loss(logits, y.long())
            
            # ---- KD loss (same as DCFCL) ----
            kd_loss = 0.0
            if self.if_last_copy and self.current_task > 0 and self.config.lambda_kd > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)
            
            # ---- Prototype augmentation loss (same as DCFCL) ----
            proto_loss = 0.0
            if proto_queue is not None and self.config.lambda_proto_aug > 0:
                proto_loss = self._compute_proto_aug_loss(proto_queue)
            
            # ---- DER++ replay losses ----
            der_logit_loss = 0.0
            der_ce_loss = 0.0
            if not self.replay_buffer.is_empty():
                buf_x, buf_y, buf_stored_logits = self.replay_buffer.get_data(
                    self.config.batch_size)
                _, _, buf_logits = self.model(buf_x)
                
                # DER: match current logits to stored logits
                if der_alpha > 0:
                    der_logit_loss = F.mse_loss(buf_logits, buf_stored_logits)
                
                # DER++: CE on replay samples with true labels
                if der_beta > 0:
                    der_ce_loss = self.ce_loss(buf_logits, buf_y)
            
            # ---- Total loss ----
            loss = (ce_loss
                    + self.config.lambda_kd * kd_loss
                    + self.config.lambda_proto_aug * proto_loss
                    + der_alpha * der_logit_loss
                    + der_beta * der_ce_loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # ---- Fill replay buffer with current data ----
            with torch.no_grad():
                _, _, store_logits = self.model(x)
            self.replay_buffer.add_data(x.detach(), y.detach(), store_logits.detach())
            
            total_loss += loss.item()
        
        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params
        
        return {
            'loss': total_loss / self.config.local_epochs,
            'num_sample_class': num_sample_class
        }
    
    def _train_step_basic(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Basic training step."""
        _, _, logits = self.model(x)
        loss = self.ce_loss(logits, y.long())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _cross_entropy_distill(self, outputs, targets, exp=1.0, eps=1e-5):
        """Cross-entropy with temperature scaling for knowledge distillation.
        
        Matches the original implementation's distillation loss.
        """
        out = F.softmax(outputs, dim=1)
        tar = F.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        return ce.mean()

    def _compute_proto_aug_loss(self, proto_queue) -> torch.Tensor:
        """Compute prototype augmentation loss."""
        prototype = self.prototype.get("global", {}) or self.prototype.get("local", {})
        radius = self.radius.get("global", 0) or self.radius.get("local", 0)
        
        if not prototype or not radius:
            return torch.tensor(0.0).to(self.device)
        
        # Generate augmented prototypes
        proto_aug = []
        proto_aug_label = []
        
        valid_indices = [k for k, v in prototype.items() 
                        if np.sum(v) != 0 and k not in self.current_labels]
        
        if not valid_indices:
            return torch.tensor(0.0).to(self.device)
        
        for _ in range(self.config.batch_size):
            idx = np.random.choice(valid_indices)
            aug = prototype[idx] + np.random.normal(0, 1, self.feature_size) * radius
            proto_aug.append(aug)
            proto_aug_label.append(idx)
        
        proto_aug = torch.from_numpy(np.array(proto_aug)).float().to(self.device)
        proto_aug_label = torch.from_numpy(np.array(proto_aug_label)).long().to(self.device)
        
        # Compute loss
        output = self.model.fc(proto_aug)
        return self.ce_loss(output, proto_aug_label)
    
    # =========================================================================
    # Prototype Methods
    # =========================================================================
    
    def compute_prototypes(self) -> Tuple[float, Dict[int, np.ndarray], List[int]]:
        """
        Compute class prototypes from local data.
        
        Returns:
            Tuple of (radius, prototypes dict, class labels)
        """
        self.model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for _ in range(self.config.local_epochs):
                x, y = self._get_next_batch()
                feature = self.model.feature(x)
                features_list.append(feature.cpu().numpy())
                labels_list.append(y.cpu().numpy())
        
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        feature_dim = features.shape[1]
        
        prototype = {}
        radius_dict = {}
        class_labels = []
        
        for cls in self.current_labels:
            idx = np.where(labels == cls)[0]
            class_labels.append(cls)
            
            if len(idx) == 0:
                prototype[cls] = np.zeros(self.feature_size)
            else:
                prototype[cls] = np.mean(features[idx], axis=0)
            
            # Compute radius if not already computed
            if not self.prototype["local"]:
                if len(idx) > 1:
                    cov = np.cov(features[idx].T)
                    trace = np.trace(cov)
                    if not math.isnan(trace):
                        radius_dict[cls] = trace / feature_dim
                    else:
                        radius_dict[cls] = 0
                else:
                    radius_dict[cls] = 0
        
        # Compute average radius
        if self.radius["local"]:
            radius = self.radius["local"]
        else:
            radius = np.sqrt(np.mean(list(radius_dict.values()))) if radius_dict else 0
        
        self.model.train()
        return radius, prototype, class_labels
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def set_parameters(self, model, beta: float = 1.0):
        """Set model parameters from another model and refresh cached state."""
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            new_data = new_param.data.to(old_param.device)
            old_param.data = beta * new_data.clone() + (1 - beta) * old_param.data.clone()

        # Keep the cached parameter vector in sync with the newly received model.
        # Otherwise coalition similarity is computed from stale pre-sync weights.
        self.param_vector = self._get_param_vector()
        self.differ = np.zeros_like(self.param_vector)
    
    def _get_param_vector(self) -> np.ndarray:
        """Get flattened parameter vector."""
        params = []
        for param in self.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def compute_l2_norm(self):
        """Compute L2 norm of parameter difference."""
        self.l2_norm = np.linalg.norm(self.differ, ord=2)
    
    def _compute_delta_model(self, model0, model1) -> Dict[str, torch.Tensor]:
        """Compute parameter difference between two models."""
        delta = {}
        for name, param0 in model0.state_dict().items():
            param1 = model1.state_dict()[name]
            delta[name] = param0.detach() - param1.detach()
        return delta
    
    def _update_local_control(self, delta_model, server_control, my_control):
        """Update local control variate for SCAFFOLD.
        
        Args:
            delta_model: Difference between old and new model parameters
            server_control: Server control variate (keyed by param name)
            my_control: This client's control variate (keyed by param name)
        """
        new_control = copy.deepcopy(my_control)
        delta_control = copy.deepcopy(my_control)
        
        for name in delta_model.keys():
            c = server_control[name].to(self.device)
            ci = my_control[name].to(self.device)
            delta = delta_model[name].to(self.device)
            
            new_ci = ci.data - c.data + delta.data / (self.config.local_epochs * self.config.scaffold_lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci
        
        return new_control, delta_control
    
    def get_sample_number(self) -> int:
        """Get number of training samples."""
        return self.train_samples
    
    # =========================================================================
    # Evaluation Methods
    # =========================================================================
    
    def test(self) -> Tuple[float, float, int]:
        """
        Test model on current test data.
        
        Returns:
            Tuple of (correct samples, loss, total samples)
        """
        self.model.eval()
        correct = 0
        total_loss = 0.0
        total = 0
        
        with torch.no_grad():
            for x, y in DataLoader(self.test_data, batch_size=self.config.batch_size):
                x, y = x.to(self.device), y.to(self.device)
                output, _, logits = self.model(x)
                total_loss += self.ce_loss(logits, y.long()).item()
                correct += (output.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        
        return correct, total_loss, total
    
    def test_per_task(self) -> Tuple[List[int], List[float], List[int]]:
        """
        Test model on each past task.
        
        Returns:
            Tuple of (correct per task, loss per task, samples per task)
        """
        self.model.eval()
        accs = []
        losses = []
        samples = []
        
        with torch.no_grad():
            for test_data in self.test_data_per_task:
                loader = DataLoader(test_data, batch_size=20)
                correct = 0
                loss = 0.0
                total = 0
                
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output, _, logits = self.model(x)
                    loss += self.ce_loss(logits, y.long()).item()
                    correct += (output.argmax(dim=1) == y).sum().item()
                    total += y.size(0)
                
                accs.append(correct)
                losses.append(loss)
                samples.append(total)
        
        return accs, losses, samples
    
    def evaluate_emergence(self, all_test_data: List, all_labels_per_task: List[List[int]]) -> Dict:
        """
        Evaluate emergence phenomenon for this client.
        
        Emergence is defined as the ability to correctly predict classes that
        the client has NEVER seen in its local training data. This knowledge
        must have been transferred from other clients through federation.
        
        Args:
            all_test_data: List of test datasets for all tasks (global)
            all_labels_per_task: List of label sets for each task (global)
            
        Returns:
            Dictionary containing:
            - 'unseen_correct': Correct predictions on unseen classes
            - 'unseen_total': Total samples from unseen classes
            - 'unseen_accuracy': Accuracy on unseen classes (emergence rate)
            - 'seen_correct': Correct predictions on seen classes
            - 'seen_total': Total samples from seen classes
            - 'seen_accuracy': Accuracy on seen classes
            - 'emergence_samples': List of (x, y, pred, confidence) for emerged samples
            - 'per_class_emergence': Dict mapping unseen class -> (correct, total)
        """
        self.model.eval()
        
        # Get classes this client has seen in its local data
        local_seen_classes = set(self.classes_so_far)
        
        # Track metrics
        unseen_correct = 0
        unseen_total = 0
        seen_correct = 0
        seen_total = 0
        
        emergence_samples = []  # Samples showing emergence
        per_class_emergence = {}  # Per unseen class accuracy
        
        with torch.no_grad():
            for task_idx, test_data in enumerate(all_test_data):
                if len(test_data) == 0:
                    continue
                    
                loader = DataLoader(test_data, batch_size=20)
                
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output, features, logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)
                    preds = output.argmax(dim=1)
                    confidences = probs.max(dim=1).values
                    
                    for i in range(len(y)):
                        true_label = y[i].item()
                        pred_label = preds[i].item()
                        confidence = confidences[i].item()
                        
                        if true_label in local_seen_classes:
                            # This is a class the client has seen locally
                            seen_total += 1
                            if pred_label == true_label:
                                seen_correct += 1
                        else:
                            # This is an UNSEEN class - emergence potential!
                            unseen_total += 1
                            
                            if true_label not in per_class_emergence:
                                per_class_emergence[true_label] = {'correct': 0, 'total': 0}
                            per_class_emergence[true_label]['total'] += 1
                            
                            if pred_label == true_label:
                                unseen_correct += 1
                                per_class_emergence[true_label]['correct'] += 1
                                
                                # Record emergence sample
                                emergence_samples.append({
                                    'sample': x[i].cpu().numpy(),
                                    'true_label': true_label,
                                    'pred_label': pred_label,
                                    'confidence': confidence,
                                    'task_idx': task_idx,
                                    'client_id': self.id,
                                    'local_seen_classes': list(local_seen_classes),
                                })
        
        return {
            'unseen_correct': unseen_correct,
            'unseen_total': unseen_total,
            'unseen_accuracy': unseen_correct / unseen_total if unseen_total > 0 else 0.0,
            'seen_correct': seen_correct,
            'seen_total': seen_total,
            'seen_accuracy': seen_correct / seen_total if seen_total > 0 else 0.0,
            'emergence_samples': emergence_samples,
            'per_class_emergence': per_class_emergence,
            'local_seen_classes': list(local_seen_classes),
        }
    
    def clean_up_counts(self):
        """Reset label counts."""
        self.label_counts = {i: 0 for i in range(self.unique_labels)}
