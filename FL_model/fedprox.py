#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedProx client implementation.

Adds a proximal regularizer on top of FedAvg to keep the local model
from drifting too far from the global one, improving convergence
stability on non-IID data.

Reference: Federated Optimization in Heterogeneous Networks
           (Li et al., 2020)
"""

import copy
from typing import Dict, Any
from .base_client import BaseClient


class FedProxClient(BaseClient):
    """
    FedProx client.

    Loss:  L = L_CE + (mu/2) * ||w - w_global||_2
    The proximal term uses the (non-squared) L2 norm, matching the
    official reference implementation.
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        FedProx local training.

        Args:
            glob_iter: global communication round index
            task:      current continual-learning task index

        Returns:
            {'loss': mean training loss over local epochs}
        """
        # Snapshot the global model for the proximal term
        global_model = copy.deepcopy(self.model)
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            _, _, logits = self.model(x)
            loss = self.ce_loss(logits, y.long())

            # Proximal term: L2 norm (matches the official implementation,
            # not the squared form).
            proximal_term = 0.0
            for w, w_g in zip(self.model.parameters(), global_model.parameters()):
                proximal_term += (w - w_g).norm(2)
            loss += (self.config.mu / 2) * proximal_term

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {'loss': total_loss / self.config.local_epochs}
