#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FL_model: modular federated learning algorithm implementations.

Each algorithm lives in its own file and subclasses BaseClient.
Use the create_client() factory to instantiate the correct client
from config.algorithm.
"""

from .base_client import BaseClient
from .fedavg import FedAvgClient
from .fedprox import FedProxClient
from .fedlwf import FedLwFClient
from .scaffold import ScaffoldClient
from .peravg import PerAvgClient
from .pfedme import pFedMeClient
from .dcfcl import DCFCLClient
from .dyndfcl import DynDFCLClient

__all__ = [
    'BaseClient',
    'FedAvgClient',
    'FedProxClient',
    'FedLwFClient',
    'ScaffoldClient',
    'PerAvgClient',
    'pFedMeClient',
    'DCFCLClient',
    'DynDFCLClient',
    'create_client',
]

# Algorithm name -> client class mapping.
_ALGORITHM_MAP = {
    'FedAvg':    FedAvgClient,
    'Local':     FedAvgClient,    # Local uses the same client loop as FedAvg.
    'FedProx':   FedProxClient,
    'FedLwF':    FedLwFClient,
    'SCAFFOLD':  ScaffoldClient,
    'PerAvg':    PerAvgClient,
    'pFedMe':    pFedMeClient,
    'ClusterFL': FedAvgClient,    # ClusterFL: same client loop; coalition formed server-side.
    'DCFCL':     DCFCLClient,
    'DynDFCL':   DynDFCLClient,
}


def create_client(config, client_id: int, model, train_data, test_data,
                  label_info: dict, unique_labels: int) -> BaseClient:
    """
    Factory: instantiate the federated-learning client corresponding to
    ``config.algorithm``.

    Args:
        config:        configuration object (must include `algorithm`)
        client_id:     unique client identifier
        model:         the (global) model used to initialize the local model
        train_data:    training dataset
        test_data:     test dataset
        label_info:    label-info dict (with 'labels' and 'counts')
        unique_labels: total number of classes in the federation

    Returns:
        an instance of the BaseClient subclass for the chosen algorithm.

    Raises:
        ValueError: if config.algorithm is not in the supported set.
    """
    algorithm = getattr(config, 'algorithm', None)
    cls = _ALGORITHM_MAP.get(algorithm)
    if cls is None:
        supported = list(_ALGORITHM_MAP.keys())
        raise ValueError(
            f"Unsupported algorithm: '{algorithm}'. "
            f"Supported: {supported}"
        )
    return cls(
        client_id=client_id,
        config=config,
        model=model,
        train_data=train_data,
        test_data=test_data,
        label_info=label_info,
        unique_labels=unique_labels,
    )
