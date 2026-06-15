#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2FCL client implementation.

Adds a Dark Experience Replay++ (DER++) mechanism on top of DCFCL:
a per-client replay buffer stores historical samples together with
their logits and ground-truth labels, and rehearses them alongside
the current task during training.

Full loss:
  L = L_CE + lambda_kd * L_KD + lambda_proto_aug * L_proto
    + alpha_der * L_DER    (dark experience replay: MSE on stored logits)
    + beta_der  * L_ER     (experience replay: CE on stored labels)

Reference: Dark Experience for General Continual Learning: a Strong,
           Simple Baseline (Buzzega et al., 2020)
"""

import copy
import torch
import torch.nn.functional as F
from typing import Dict, Any
from .base_client import BaseClient
from core.replay_buffer import ReplayBuffer


class D2FCLClient(BaseClient):
    """
    D2FCL client.

    Stacks the DER++ replay loss on top of the three DCFCL losses for
    stronger resistance to catastrophic forgetting.

    The replay buffer persists across both rounds and tasks; it is
    refreshed at the end of every training step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # D2FCL-specific: the DER++ replay buffer.
        self.replay_buffer = ReplayBuffer(self.config.buffer_size, self.device)

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        D2FCL local training (DCFCL + DER++).

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

        der_alpha = self.config.der_alpha
        der_beta = self.config.der_beta

        # Refresh the teacher model (matching DCFCL: snapshot every round).
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True

        initial_params = self._get_param_vector().copy()

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            for label in y.tolist():
                num_sample_class[label] += 1

            # ---- Forward pass on the current batch ----
            output, features, logits = self.model(x)
            ce_loss = self.ce_loss(logits, y.long())

            # ---- Knowledge-distillation loss (same as DCFCL) ----
            kd_loss = 0.0
            if self.if_last_copy and self.current_task > 0 and self.config.lambda_kd > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)

            # ---- Prototype-augmentation loss (same as DCFCL) ----
            proto_loss = 0.0
            if proto_queue is not None and self.config.lambda_proto_aug > 0:
                proto_loss = self._compute_proto_aug_loss(proto_queue)

            # ---- DER++ replay loss ----
            der_logit_loss = 0.0
            der_ce_loss = 0.0
            if self.config.use_der and not self.replay_buffer.is_empty():
                buf_x, buf_y, buf_stored_logits = self.replay_buffer.get_data(
                    self.config.batch_size
                )
                _, _, buf_logits = self.model(buf_x)

                # DER: match current logits to the stored historical logits
                # (the dark-experience term).
                if der_alpha > 0:
                    der_logit_loss = F.mse_loss(buf_logits, buf_stored_logits)

                # DER++: CE on the stored ground-truth labels.
                if der_beta > 0:
                    der_ce_loss = self.ce_loss(buf_logits, buf_y)

            # ---- Composite loss ----
            loss = (ce_loss
                    + self.config.lambda_kd * kd_loss
                    + self.config.lambda_proto_aug * proto_loss
                    + der_alpha * der_logit_loss
                    + der_beta * der_ce_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---- Store the current batch into the replay buffer ----
            if self.config.use_der:
                with torch.no_grad():
                    _, _, store_logits = self.model(x)
                self.replay_buffer.add_data(x.detach(), y.detach(), store_logits.detach())

            total_loss += loss.item()

        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params

        return {
            'loss': total_loss / self.config.local_epochs,
            'num_sample_class': num_sample_class,
        }
