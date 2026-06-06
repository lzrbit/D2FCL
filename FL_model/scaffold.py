#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCAFFOLD client implementation.

Uses control variates to correct client-side gradient bias and
mitigate client drift under non-IID data.

Reference: SCAFFOLD: Stochastic Controlled Averaging for Federated
           Learning (Karimireddy et al., 2020)
"""

import copy
from typing import Dict, Any
from .base_client import BaseClient


class ScaffoldClient(BaseClient):
    """
    SCAFFOLD client.

    After each round of local training, computes:
    - delta_model:    parameter delta (sent to the server for aggregation)
    - client_control: updated local control variate
    - delta_control:  control-variate delta (sent to the server)
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        SCAFFOLD local training.

        Args:
            glob_iter:      global communication round index
            task:           current continual-learning task index
            server_control: server-side control variate (required)
            client_control: dict of all clients' control variates (required)

        Returns:
            {'loss': mean training loss over local epochs}
        """
        server_control = kwargs.get('server_control')
        client_control = kwargs.get('client_control')
        # Pull out this client's control variate
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
            # The SCAFFOLD optimizer applies the control-variate correction.
            self.optimizer.step(server_control, my_control)

            total_loss += loss.item()

        # Compute parameter delta and the updated control variate
        self.delta_model = self._compute_delta_model(global_model, self.model)
        self.client_control, self.delta_control = self._update_local_control(
            self.delta_model, server_control, my_control
        )

        return {'loss': total_loss / self.config.local_epochs}
