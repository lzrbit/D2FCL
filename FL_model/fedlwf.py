#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedLwF client implementation.

Combines federated learning with Learning without Forgetting (LwF):
knowledge distillation against the previous-task model mitigates
catastrophic forgetting in the continual-learning setting.

Reference: Learning without Forgetting (Li & Hoiem, 2017),
           applied to the federated continual-learning setting.
"""

import torch
from typing import Dict, Any
from .base_client import BaseClient


class FedLwFClient(BaseClient):
    """
    FedLwF client.

    Loss:  L = L_CE + alpha * L_KD
    where L_KD is the distillation loss between the current model and
    the previous-task snapshot.
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        FedLwF local training.

        When a previous-task snapshot exists (if_last_copy=True) and we
        are past the first task, a distillation loss is added on top of
        the cross-entropy term.

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

            output, _, logits = self.model(x)
            class_loss = self.ce_loss(logits, y.long())

            if self.if_last_copy and self.current_task > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                # Distillation loss (temperature exponent exp=0.5, matching
                # the original implementation).
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)
                loss = class_loss + self.config.alpha * kd_loss
            else:
                loss = class_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {'loss': total_loss / self.config.local_epochs}
