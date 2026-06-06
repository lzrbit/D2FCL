#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-FedAvg client implementation.

Personalized FL via MAML (Model-Agnostic Meta-Learning): trains a
global initialization that each client can adapt to its own task with
only a few gradient steps.

Reference: Personalized Federated Learning with Theoretical Guarantees:
           A Model-Agnostic Meta-Learning Approach (Fallah et al., 2020)
"""

import copy
from typing import Dict, Any
from .base_client import BaseClient


class PerAvgClient(BaseClient):
    """
    Per-FedAvg client.

    Each local step is a two-stage update:
    1. Inner loop: standard forward-backward update.
    2. Outer loop: personalization step using the second-batch gradient
       (step size = beta).
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        Per-FedAvg local training.

        Args:
            glob_iter: global communication round index
            task:      current continual-learning task index

        Returns:
            {'loss': mean training loss over local epochs}
        """
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            # Step 1: standard update; snapshot the pre-step model.
            temp_model = copy.deepcopy(self.model)
            x, y = self._get_next_batch()
            self._train_step_basic(x, y)

            # Step 2: personalization update.
            x, y = self._get_next_batch()
            _, _, logits = self.model(x)
            loss = self.ce_loss(logits, y.long())

            self.optimizer.zero_grad()
            loss.backward()

            # Restore the pre-step parameters, then apply the personalization step.
            for old_p, new_p in zip(self.model.parameters(), temp_model.parameters()):
                old_p.data = new_p.data.clone()
            self.optimizer.step(beta=self.config.beta)

            total_loss += loss.item()

        return {'loss': total_loss / self.config.local_epochs}
