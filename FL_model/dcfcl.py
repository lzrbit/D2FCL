#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCFCL client implementation.

The core method from the Decentralized Continual Federated Learning
(DCFCL) paper. Local training combines three losses:
  - L_CE:        classification cross-entropy
  - L_KD:        knowledge distillation (preserves the feature space
                 learned on previous tasks)
  - L_proto_aug: prototype augmentation (synthesizes virtual samples
                 from previously seen class prototypes)

Full loss (paper Eq. 6):
  L = L_CE + lambda_kd * L_KD + lambda_proto_aug * L_proto_aug

At the end of training, the per-client parameter delta (``differ``)
is computed for coalition similarity scoring.
"""

import copy
import torch
from typing import Dict, Any
from .base_client import BaseClient


class DCFCLClient(BaseClient):
    """
    DCFCL client.

    Implements the local training loop from the paper (Section 4.1):
    1. Snapshot a teacher model each round (matches the official code's
       round-level KD, not just at task boundaries).
    2. Jointly optimize the three losses.
    3. Record the parameter delta for coalition formation.
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        DCFCL local training.

        Args:
            glob_iter:   global communication round index
            task:        current continual-learning task index
            proto_queue: global prototype queue (used by the prototype-aug loss)

        Returns:
            {
                'loss':             mean training loss over local epochs,
                'num_sample_class': per-class sample counts (dict)
            }
        """
        proto_queue = kwargs.get('proto_queue')
        self.model.train()
        total_loss = 0.0
        num_sample_class = {k: 0 for k in range(self.config.num_classes)}

        # Snapshot the teacher model for KD.
        # Paper Eq. 4: the teacher is the previous-round model (not only
        # at task boundaries).
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True

        # Record the initial parameters; `differ` is the cumulative delta
        # over all local steps.
        initial_params = self._get_param_vector().copy()

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            # Track per-class sample counts (used by prototype-weighted aggregation).
            for label in y.tolist():
                num_sample_class[label] += 1

            output, features, logits = self.model(x)

            # Classification loss (uses logits, not softmax outputs).
            ce_loss = self.ce_loss(logits, y.long())

            # Knowledge-distillation loss (paper Section 4.1, Eq. 4).
            kd_loss = 0.0
            if self.if_last_copy and self.current_task > 0 and self.config.lambda_kd > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)

            # Prototype-augmentation loss.
            proto_loss = 0.0
            if proto_queue is not None and self.config.lambda_proto_aug > 0:
                proto_loss = self._compute_proto_aug_loss(proto_queue)

            # Composite loss (paper Eq. 6).
            loss = (ce_loss
                    + self.config.lambda_kd * kd_loss
                    + self.config.lambda_proto_aug * proto_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Cumulative parameter delta (used by coalition similarity scoring).
        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params

        return {
            'loss': total_loss / self.config.local_epochs,
            'num_sample_class': num_sample_class,
        }
