#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replay Buffer for DCFCL-DER (Dark Experience Replay in Federated Continual Learning).

Implements a reservoir-sampling-based replay buffer that stores past examples along
with their "dark experience" (model logits at the time of storage). Inspired by:
    "Dark Experience for General Continual Learning: A Strong, Simple Baseline"
    (Buzzega et al., NeurIPS 2020)

Each client maintains its own buffer. Raw data never leaves the client.
"""

import numpy as np
import torch
from typing import Tuple, Optional


class ReplayBuffer:
    """
    Reservoir-sampling replay buffer storing (example, label, logits).

    Attributes:
        buffer_size: maximum number of samples to store
        device: torch device
        num_seen: total samples seen (for reservoir sampling)
        examples: stored input tensors
        labels: stored ground-truth labels
        logits: stored model logits at time of insertion ("dark experience")
    """

    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen = 0
        self.examples: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return min(self.num_seen, self.buffer_size)

    def is_empty(self) -> bool:
        return self.num_seen == 0

    # ------------------------------------------------------------------
    # Reservoir sampling
    # ------------------------------------------------------------------
    def _reservoir_index(self) -> int:
        """Return index to place new sample, or -1 if rejected."""
        if self.num_seen < self.buffer_size:
            return self.num_seen
        rand = np.random.randint(0, self.num_seen + 1)
        return rand if rand < self.buffer_size else -1

    # ------------------------------------------------------------------
    # Add data
    # ------------------------------------------------------------------
    def add_data(self, examples: torch.Tensor, labels: torch.Tensor,
                 logits: torch.Tensor):
        """
        Add a batch of (example, label, logit) tuples via reservoir sampling.

        Args:
            examples: (B, *) input tensor
            labels: (B,) label tensor
            logits: (B, C) logit tensor (detached)
        """
        if self.examples is None:
            # Lazy init storage
            self.examples = torch.zeros(
                (self.buffer_size, *examples.shape[1:]),
                dtype=torch.float32, device=self.device)
            self.labels = torch.zeros(
                self.buffer_size, dtype=torch.long, device=self.device)
            self.logits = torch.zeros(
                (self.buffer_size, logits.shape[1]),
                dtype=torch.float32, device=self.device)

        for i in range(examples.shape[0]):
            idx = self._reservoir_index()
            self.num_seen += 1
            if idx >= 0:
                self.examples[idx] = examples[i].to(self.device)
                self.labels[idx] = labels[i].to(self.device)
                self.logits[idx] = logits[i].to(self.device)

    # ------------------------------------------------------------------
    # Sample data
    # ------------------------------------------------------------------
    def get_data(self, size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch from the buffer.

        Args:
            size: number of samples requested

        Returns:
            (examples, labels, logits) tuple, all on self.device
        """
        n = len(self)
        if size > n:
            size = n
        choice = np.random.choice(n, size=size, replace=False)
        return (self.examples[choice],
                self.labels[choice],
                self.logits[choice])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def to(self, device: torch.device) -> 'ReplayBuffer':
        self.device = device
        if self.examples is not None:
            self.examples = self.examples.to(device)
            self.labels = self.labels.to(device)
            self.logits = self.logits.to(device)
        return self
