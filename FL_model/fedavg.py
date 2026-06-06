#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedAvg client implementation.

Also covers the Local algorithm: the per-client training logic is
identical to FedAvg; the server simply skips aggregation in Local mode.

Reference: Communication-Efficient Learning of Deep Networks
           from Decentralized Data (McMahan et al., 2017)
"""

from typing import Dict, Any
from .base_client import BaseClient


class FedAvgClient(BaseClient):
    """
    FedAvg / Local client.

    Runs standard local SGD each round, then ships parameters to the
    server for aggregation. In Local mode the server does not aggregate,
    but the client training loop is unchanged.
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        FedAvg local training.

        Runs `local_epochs` rounds of standard cross-entropy training
        and updates `differ` (parameter delta) for coalition-based
        algorithms such as ClusterFL.

        Args:
            glob_iter: global communication round index
            task:      current continual-learning task index

        Returns:
            {'loss': mean training loss over local epochs}
        """
        initial_params = self.param_vector.copy()
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            loss = self._train_step_basic(x, y)
            total_loss += loss

        # Update `differ` (used by coalition similarity scoring)
        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params

        return {'loss': total_loss / self.config.local_epochs}
