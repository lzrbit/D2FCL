#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom optimizers for DCFCL.

Implements specialized optimizers for different FL algorithms:
- ScaffoldOptimizer: For SCAFFOLD variance reduction
- PerAvgOptimizer: For Per-FedAvg personalization
- pFedMeOptimizer: For pFedMe personalization
"""

import torch
from torch.optim import Optimizer


class ScaffoldOptimizer(Optimizer):
    """
    SCAFFOLD optimizer with control variates for variance reduction.
    
    Reference: "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    """
    
    def __init__(self, params, lr: float, weight_decay: float = 0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, server_control, client_control, closure=None):
        """
        Perform optimization step with control variate correction.
        
        Args:
            server_control: Server control variate
            client_control: Client control variate
            closure: Optional closure for computing loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Get parameter names (excluding batch norm running stats)
        names = [name for name in server_control.keys() 
                 if 'running' not in name and 'num_batch' not in name]
        
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    idx += 1
                    continue
                
                if idx < len(names):
                    name = names[idx]
                    c = server_control[name].to(p.device)
                    ci = client_control[name].to(p.device)
                    
                    # SCAFFOLD update: gradient + c - ci
                    d_p = p.grad.data + c.data - ci.data
                    p.data = p.data - group['lr'] * d_p
                
                idx += 1
        
        return loss


class PerAvgOptimizer(Optimizer):
    """
    Per-FedAvg optimizer for personalized federated learning.
    
    Reference: "Personalized Federated Learning with Moreau Envelopes"
    """
    
    def __init__(self, params, lr: float):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    
    def step(self, beta: float = 0.0, closure=None):
        """
        Perform optimization step.
        
        Args:
            beta: Personalization learning rate (if 0, uses default lr)
            closure: Optional closure for computing loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = beta if beta != 0 else group['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(p.grad.data, alpha=-lr)
        
        return loss


class pFedMeOptimizer(Optimizer):
    """
    pFedMe optimizer for personalized federated learning.
    
    Reference: "Personalized Federated Learning with Moreau Envelopes"
    """
    
    def __init__(self, params, lr: float = 0.01, lamda: float = 0.1, mu: float = 0.001):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super().__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        """
        Perform optimization step with personalization.
        
        Args:
            local_weight_updated: Local model weights for regularization
            closure: Optional closure for computing loss
            
        Returns:
            Updated parameters and loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p, local_w in zip(group['params'], local_weight_updated):
                if p.grad is not None:
                    # pFedMe update rule
                    update = (p.grad.data + 
                              group['lamda'] * (p.data - local_w.data) + 
                              group['mu'] * p.data)
                    p.data = p.data - group['lr'] * update
        
        return list(self.param_groups[0]['params']), loss
    
    def update_param(self, local_weight_updated):
        """Copy local weights to model parameters."""
        for group in self.param_groups:
            for p, local_w in zip(group['params'], local_weight_updated):
                p.data = local_w.data.clone()
        return list(self.param_groups[0]['params'])
