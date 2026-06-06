#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BaseClient: base class for every federated-learning client.

Provides the shared infrastructure (data loading, model management,
prototype computation, evaluation) that all algorithms reuse. Algorithm
subclasses only need to implement train().
"""

import copy
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any

from core.optimizers import ScaffoldOptimizer, PerAvgOptimizer, pFedMeOptimizer

logger = logging.getLogger('DCFCL.Client')


class BaseClient:
    """
    Base class for federated-learning clients.

    Shared functionality:
    - Data loading and management
    - Model initialization
    - Optimizer configuration
    - Prototype computation (for continual learning)
    - Parameter sync utilities
    - Evaluation methods

    Subclasses implement train() to define algorithm-specific local
    training.
    """

    def __init__(self, client_id: int, config, model, train_data, test_data,
                 label_info: Dict, unique_labels: int):
        """
        Initialize the client.

        Args:
            client_id:    unique client identifier
            config:       configuration object
            model:        the global model (deep-copied into the local model)
            train_data:   training dataset
            test_data:    test dataset
            label_info:   label-info dict (with 'labels' and 'counts')
            unique_labels: total number of classes in the federation
        """
        self.id = client_id
        self.config = config

        # Device selection.
        device_name = getattr(config, 'device', 'auto')
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device_name == 'cuda' and not torch.cuda.is_available():
            device_name = 'cpu'
        self.device = torch.device(device_name)

        # Build the local model (deep copy of the global one).
        # Imported lazily to avoid a circular import.
        from core.models import create_model
        self.model = create_model(config)
        self.model.to(self.device)

        # Data attributes.
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.unique_labels = unique_labels

        # Build the DataLoaders.
        self._setup_dataloaders()

        # Continual-learning label bookkeeping.
        self.classes_so_far: List[int] = list(label_info.get('labels', []))
        self.current_labels: List[int] = list(label_info.get('labels', []))
        self.classes_past_task: List[int] = []
        self.available_labels: List[int] = []
        self.available_labels_current: List[int] = []
        self.current_task: int = 0

        # Per-class sample counts.
        self.label_counts: Dict[int, int] = {i: 0 for i in range(unique_labels)}

        # Knowledge-distillation snapshot (continual learning).
        self.last_copy: Optional[nn.Module] = None
        self.if_last_copy: bool = False

        # Optimizer.
        self._setup_optimizer()

        # Prototype state (DCFCL-family).
        self.prototype = {"global": {}, "local": {}}
        self.radius = {"global": 0, "local": 0}
        self.feature_size = config.feature_size
        self.num_sample_class: Optional[Dict] = None

        # Gradient tracking (used for coalition similarity scoring).
        self.param_vector = self._get_param_vector()
        self.differ = np.zeros_like(self.param_vector)
        self.l2_norm: Optional[float] = None

        # Test-data bookkeeping (continual cross-task evaluation).
        self.test_data_so_far = list(test_data)
        self.test_data_per_task = [test_data]

        # Loss functions.
        ls = getattr(config, 'label_smoothing', 0.0)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=ls)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    # =========================================================================
    # Initialization helpers
    # =========================================================================

    def _setup_dataloaders(self):
        """Configure the train and test DataLoaders."""
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
        """Configure the optimizer based on the algorithm."""
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
            self.optimizer = self.model.classifier_optimizer

    # =========================================================================
    # Task management
    # =========================================================================

    def next_task(self, train_data, test_data, label_info: Dict):
        """
        Update the client state for the next continual-learning task.

        Args:
            train_data: training data for the new task
            test_data:  test data for the new task
            label_info: label info for the new task
        """
        # Snapshot the current model for knowledge distillation.
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True

        # Refresh data.
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self._setup_dataloaders()

        # Update label bookkeeping.
        self.classes_past_task = list(self.classes_so_far)
        self.classes_so_far.extend(label_info.get('labels', []))
        self.current_labels = list(label_info.get('labels', []))

        # Update test bookkeeping.
        self.test_data_so_far.extend(test_data)
        self.test_data_per_task.append(test_data)

        self.current_task += 1

    # =========================================================================
    # Training interface (subclasses must implement)
    # =========================================================================

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        Run local training.

        Args:
            glob_iter: global communication round index
            task:      current continual-learning task index
            **kwargs:  algorithm-specific parameters

        Returns:
            dict of training results (must include at least 'loss').
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement train()."
        )

    # =========================================================================
    # Shared training utilities
    # =========================================================================

    def _get_next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch the next training batch; reset the iterator at end-of-epoch."""
        try:
            X, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            X, y = next(self.iter_trainloader)
        return X.to(self.device), y.to(self.device)

    def _train_step_basic(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """One basic training step (forward + backward + update)."""
        _, _, logits = self.model(x)
        loss = self.ce_loss(logits, y.long())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _cross_entropy_distill(self, outputs, targets, exp: float = 1.0, eps: float = 1e-5):
        """
        Cross-entropy distillation with temperature scaling.

        Matches the reference implementation.

        Args:
            outputs: current model outputs (softmax probabilities)
            targets: teacher outputs (softmax probabilities)
            exp:     temperature exponent
            eps:     numerical-stability smoothing constant

        Returns:
            scalar distillation loss.
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
        """
        Prototype-augmentation loss.

        Generates augmented samples by adding Gaussian noise around the
        existing class prototypes, which prevents the model from
        forgetting feature representations of past tasks.

        Args:
            proto_queue: global prototype-queue object

        Returns:
            scalar prototype-augmentation loss.
        """
        prototype = self.prototype.get("global", {}) or self.prototype.get("local", {})
        radius = self.radius.get("global", 0) or self.radius.get("local", 0)

        if not prototype or not radius:
            return torch.tensor(0.0).to(self.device)

        # Only augment classes that are not in the current task.
        valid_indices = [k for k, v in prototype.items()
                         if np.sum(v) != 0 and k not in self.current_labels]

        if not valid_indices:
            return torch.tensor(0.0).to(self.device)

        proto_aug = []
        proto_aug_label = []
        for _ in range(self.config.batch_size):
            idx = np.random.choice(valid_indices)
            aug = prototype[idx] + np.random.normal(0, 1, self.feature_size) * radius
            proto_aug.append(aug)
            proto_aug_label.append(idx)

        proto_aug = torch.from_numpy(np.array(proto_aug)).float().to(self.device)
        proto_aug_label = torch.from_numpy(np.array(proto_aug_label)).long().to(self.device)

        output = self.model.fc(proto_aug)
        return self.ce_loss(output, proto_aug_label)

    # =========================================================================
    # Prototype computation
    # =========================================================================

    def compute_prototypes(self) -> Tuple[float, Dict[int, np.ndarray], List[int]]:
        """
        Compute per-class prototypes from the local data.

        Returns:
            Tuple of (radius, prototype_dict, class_labels)
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

        if self.radius["local"]:
            radius = self.radius["local"]
        else:
            radius = np.sqrt(np.mean(list(radius_dict.values()))) if radius_dict else 0

        self.model.train()
        return radius, prototype, class_labels

    # =========================================================================
    # Parameter sync + utilities
    # =========================================================================

    def set_parameters(self, model, beta: float = 1.0):
        """
        Sync parameters from another model and refresh cached state.

        Args:
            model: source model
            beta:  EMA mixing coefficient (1.0 = full replacement)
        """
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            new_data = new_param.data.to(old_param.device)
            old_param.data = beta * new_data.clone() + (1 - beta) * old_param.data.clone()
        # Refresh the cached parameter vector so coalition similarity
        # never uses stale weights.
        self.param_vector = self._get_param_vector()
        self.differ = np.zeros_like(self.param_vector)

    def _get_param_vector(self) -> np.ndarray:
        """Return the flattened parameter vector."""
        params = []
        for param in self.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def compute_l2_norm(self):
        """Compute the L2 norm of the parameter delta."""
        self.l2_norm = np.linalg.norm(self.differ, ord=2)

    def _compute_delta_model(self, model0, model1) -> Dict[str, torch.Tensor]:
        """Compute the parameter delta between two models."""
        delta = {}
        for name, param0 in model0.state_dict().items():
            param1 = model1.state_dict()[name]
            delta[name] = param0.detach() - param1.detach()
        return delta

    def _update_local_control(self, delta_model, server_control, my_control):
        """
        Update the SCAFFOLD local control variate.

        Args:
            delta_model:    model parameter delta (old - new)
            server_control: server-side control variate (keyed by param name)
            my_control:     this client's control variate (keyed by param name)

        Returns:
            Tuple of (new_control, delta_control).
        """
        new_control = copy.deepcopy(my_control)
        delta_control = copy.deepcopy(my_control)

        for name in delta_model.keys():
            c = server_control[name].to(self.device)
            ci = my_control[name].to(self.device)
            delta = delta_model[name].to(self.device)

            new_ci = ci.data - c.data + delta.data / (
                self.config.local_epochs * self.config.scaffold_lr
            )
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci

        return new_control, delta_control

    def get_sample_number(self) -> int:
        """Return the number of training samples."""
        return self.train_samples

    def clean_up_counts(self):
        """Reset the per-class label counts."""
        self.label_counts = {i: 0 for i in range(self.unique_labels)}

    # =========================================================================
    # Evaluation
    # =========================================================================

    def test(self) -> Tuple[float, float, int]:
        """
        Evaluate the model on the current-task test set.

        Returns:
            Tuple of (correct_count, total_loss, total_samples)
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
        Per-task evaluation (used to measure continual-learning forgetting).

        Returns:
            Tuple of (correct_per_task, loss_per_task, samples_per_task).
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
        Evaluate the collective-intelligence emergence on this client.

        Emergence: the client correctly predicts classes it has never
        observed locally. This capability comes from peer knowledge
        transferred through federated coalition aggregation.

        Args:
            all_test_data:       per-task test datasets (global view)
            all_labels_per_task: per-task label sets (global view)

        Returns:
            dict with:
            - 'unseen_correct':    correct predictions on unseen classes
            - 'unseen_total':      total samples of unseen classes
            - 'unseen_accuracy':   accuracy on unseen classes (emergence rate)
            - 'seen_correct':      correct predictions on seen classes
            - 'seen_total':        total samples of seen classes
            - 'seen_accuracy':     accuracy on seen classes
            - 'emergence_samples': list of (x, y, pred, confidence)
            - 'per_class_emergence': {class -> (correct, total)}
        """
        self.model.eval()

        # Classes this client has seen locally.
        local_seen_classes = set(self.classes_so_far)

        # Tallies.
        unseen_correct = 0
        unseen_total = 0
        seen_correct = 0
        seen_total = 0

        emergence_samples = []   # samples where emergence happened
        per_class_emergence = {} # accuracy on each unseen class

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
                            # Class seen locally.
                            seen_total += 1
                            if pred_label == true_label:
                                seen_correct += 1
                        else:
                            # Unseen class -- candidate for emergence.
                            unseen_total += 1

                            if true_label not in per_class_emergence:
                                per_class_emergence[true_label] = {'correct': 0, 'total': 0}
                            per_class_emergence[true_label]['total'] += 1

                            if pred_label == true_label:
                                unseen_correct += 1
                                per_class_emergence[true_label]['correct'] += 1

                                # Log the emergence sample.
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
