#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pFedMe client implementation.

Personalized FL via Moreau-envelope regularization: each client keeps
an independent personalized local model alongside the global one.

Reference: Personalized Federated Learning with Moreau Envelopes
           (Dinh et al., 2020)
"""

import copy
from typing import Dict, Any
from .base_client import BaseClient


class pFedMeClient(BaseClient):
    """
    pFedMe client.

    Maintains two models:
    - self.model:       the global model (participates in aggregation)
    - self.local_model: the personalized local model (not aggregated)

    Each round first runs K proximal steps on the personalized model,
    then updates the global model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pFedMe-specific: the personalized local model.
        self.local_model = copy.deepcopy(self.model)

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        pFedMe local training.

        Args:
            glob_iter: global communication round index
            task:      current continual-learning task index

        Returns:
            {'loss': mean training loss over local epochs}
        """
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            # K proximal personalization steps.
            for _ in range(self.config.K):
                _, _, logits = self.model(x)
                loss = self.ce_loss(logits, y.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.local_model.parameters())

            # Moreau-envelope proximal update of the personalized model.
            for local_w, w in zip(self.local_model.parameters(), self.model.parameters()):
                local_w.data = (
                    local_w.data
                    - self.config.lamda * self.config.lr * (local_w.data - w.data)
                )

            total_loss += loss.item()

        # Upload the personalized parameters (used when computing the
        # aggregation contribution).
        self.set_parameters(self.local_model)

        return {'loss': total_loss / self.config.local_epochs}
