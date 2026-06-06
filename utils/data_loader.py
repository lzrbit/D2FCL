#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading utilities for DCFCL.

Handles:
- Dataset loading (EMNIST, CIFAR100, MNIST-SVHN-FASHION)
- Data splitting for federated learning
- Per-client, per-task data organization
"""

import os
import pickle
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class TransformDataset(Dataset):
    """Dataset wrapper that applies transforms to data."""
    
    def __init__(self, X: List, Y: List, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        x = self.X[index]
        y = self.Y[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self) -> int:
        return len(self.X)


def get_dataset(config) -> Dict:
    """
    Load dataset and split into federated format.
    
    Args:
        config: Configuration object with dataset settings
        
    Returns:
        Dictionary with:
        - client_names: List of client identifiers
        - train_data: Training data per client
        - test_data: Test data per client
        - unique_labels: Number of unique labels
    """
    dataset_name = config.dataset
    datadir = config.datadir
    data_split_file = config.data_split_file
    
    if dataset_name in ['EMNIST-Letters', 'EMNIST-Letters-shuffle']:
        return _load_emnist(datadir, data_split_file, shuffle='shuffle' in dataset_name)
    
    elif dataset_name == 'CIFAR100':
        return _load_cifar100(datadir, data_split_file)
    
    elif dataset_name == 'MNIST-SVHN-FASHION':
        return _load_mixed_dataset(datadir, data_split_file)
    
    elif dataset_name == 'TEST-noniid':
        return _load_test_noniid(datadir, config.num_users)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_emnist(datadir: str, split_file: str, shuffle: bool = False) -> Dict:
    """Load EMNIST-Letters dataset."""
    unique_labels = 26
    
    # Load data
    data_train = datasets.EMNIST(
        datadir, 'letters', 
        download=True, train=True,
        transform=transforms.ToTensor(),
        target_transform=lambda x: x - 1  # Labels 1-26 -> 0-25
    )
    data_test = datasets.EMNIST(
        datadir, 'letters',
        download=True, train=False,
        transform=transforms.ToTensor(),
        target_transform=lambda x: x - 1
    )
    
    # Load split indices
    split_path = os.path.join(datadir, split_file)
    with open(split_path, 'rb') as f:
        split_data = pickle.load(f)
    
    # Organize data by client and task
    data_train_reshape = _split_data_from_indices(data_train, split_data['train_inds'])
    data_test_reshape = _split_data_from_indices(data_test, split_data['test_inds'])
    
    return {
        'client_names': list(data_train_reshape.keys()),
        'train_data': data_train_reshape,
        'test_data': data_test_reshape,
        'unique_labels': unique_labels
    }


def _load_cifar100(datadir: str, split_file: str) -> Dict:
    """Load CIFAR-100 dataset."""
    unique_labels = 100
    
    data_train = datasets.CIFAR100(datadir, download=True, train=True)
    data_test = datasets.CIFAR100(datadir, download=True, train=False)
    
    split_path = os.path.join(datadir, split_file)
    with open(split_path, 'rb') as f:
        split_data = pickle.load(f)
    
    data_train_reshape = _split_data_from_indices(data_train, split_data['train_inds'])
    data_test_reshape = _split_data_from_indices(data_test, split_data['test_inds'])
    
    return {
        'client_names': list(data_train_reshape.keys()),
        'train_data': data_train_reshape,
        'test_data': data_test_reshape,
        'unique_labels': unique_labels
    }


def _load_mixed_dataset(datadir: str, split_file: str) -> Dict:
    """Load mixed MNIST-SVHN-FashionMNIST dataset."""
    unique_labels = 20
    
    # MNIST
    repeat_transform = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    mnist_transform = transforms.Compose([
        transforms.Pad(padding=2, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1,), (0.2752,)),
        repeat_transform
    ])
    
    mnist_train = datasets.MNIST(datadir, train=True, download=True, transform=mnist_transform)
    mnist_test = datasets.MNIST(datadir, train=False, download=True, transform=mnist_transform)
    
    # SVHN
    svhn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4377, 0.4438, 0.4728], [0.198, 0.201, 0.197])
    ])
    
    svhn_train = datasets.SVHN(datadir, split='train', download=True, transform=svhn_transform)
    svhn_test = datasets.SVHN(datadir, split='test', download=True, transform=svhn_transform)
    
    # FashionMNIST (labels + 10)
    fashion_transform = transforms.Compose([
        transforms.Pad(padding=2, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.2190,), (0.3318,)),
        repeat_transform
    ])
    
    fashion_train = datasets.FashionMNIST(
        datadir, train=True, download=True, transform=fashion_transform,
        target_transform=lambda x: x + 10
    )
    fashion_test = datasets.FashionMNIST(
        datadir, train=False, download=True, transform=fashion_transform,
        target_transform=lambda x: x + 10
    )
    
    # Combine datasets
    data_train = []
    data_test = []
    
    for dataset in [mnist_train, svhn_train, fashion_train]:
        data_train.extend([(dataset[i][0], dataset[i][1]) for i in range(len(dataset))])
    
    for dataset in [mnist_test, svhn_test, fashion_test]:
        data_test.extend([(dataset[i][0], dataset[i][1]) for i in range(len(dataset))])
    
    # Load split
    split_path = os.path.join(datadir, split_file)
    with open(split_path, 'rb') as f:
        split_data = pickle.load(f)
    
    data_train_reshape = _split_data_from_indices(data_train, split_data['train_inds'])
    data_test_reshape = _split_data_from_indices(data_test, split_data['test_inds'])
    
    return {
        'client_names': list(data_train_reshape.keys()),
        'train_data': data_train_reshape,
        'test_data': data_test_reshape,
        'unique_labels': unique_labels
    }


def _load_test_noniid(datadir: str, num_users: int) -> Dict:
    """Load test non-IID MNIST dataset."""
    import random
    
    unique_labels = 10
    
    mean, std = (0.1,), (0.2752,)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    data_train = datasets.MNIST(datadir, train=True, download=True, transform=transform)
    data_test = datasets.MNIST(datadir, train=False, download=True, transform=transform)
    
    # Create non-IID split: each client gets 2 classes
    train_labels = np.array([data_train[i][1] for i in range(len(data_train))])
    test_labels = np.array([data_test[i][1] for i in range(len(data_test))])
    
    class_indices_train = {c: np.where(train_labels == c)[0].tolist() for c in range(10)}
    class_indices_test = {c: np.where(test_labels == c)[0].tolist() for c in range(10)}
    
    data_train_reshape = {}
    data_test_reshape = {}
    
    for i in range(num_users):
        # Client i gets classes i and (i+1) % 10
        c1, c2 = i % 10, (i + 1) % 10
        
        train_idx = random.sample(class_indices_train[c1], 200) + random.sample(class_indices_train[c2], 200)
        test_idx = random.sample(class_indices_test[c1], 400) + random.sample(class_indices_test[c2], 400)
        
        data_train_reshape[f'client_{i}'] = {
            'x': [[data_train[j][0] for j in train_idx]],
            'y': [[data_train[j][1] for j in train_idx]]
        }
        data_test_reshape[f'client_{i}'] = {
            'x': [[data_test[j][0] for j in test_idx]],
            'y': [[data_test[j][1] for j in test_idx]]
        }
    
    return {
        'client_names': list(data_train_reshape.keys()),
        'train_data': data_train_reshape,
        'test_data': data_test_reshape,
        'unique_labels': unique_labels
    }


def _split_data_from_indices(data, indices: List) -> Dict:
    """
    Split dataset according to client/task indices.
    
    Args:
        data: Original dataset
        indices: List of lists of lists - [client][task][sample_indices]
        
    Returns:
        Dictionary mapping client names to their data
    """
    data_reshape = {}
    
    for client_idx in range(len(indices)):
        x_client = []
        y_client = []
        
        for task_idx in range(len(indices[client_idx])):
            task_indices = indices[client_idx][task_idx]
            
            x_task = [data[i][0] for i in task_indices]
            y_task = [data[i][1] for i in task_indices]
            
            x_client.append(x_task)
            y_client.append(y_task)
        
        data_reshape[f'client_{client_idx}'] = {'x': x_client, 'y': y_client}
    
    return data_reshape


def read_user_data(index: int, data: Dict, dataset: str, task: int = 0) -> Tuple:
    """
    Read data for a specific user and task.
    
    Args:
        index: Client index
        data: Data dictionary from get_dataset
        dataset: Dataset name
        task: Task index
        
    Returns:
        Tuple of (client_id, train_data, test_data, label_info)
    """
    client_name = data['client_names'][index]
    train_data = data['train_data'][client_name]
    test_data = data['test_data'][client_name]
    
    X_train = train_data['x'][task]
    y_train = train_data['y'][task]
    X_test = test_data['x'][task]
    y_test = test_data['y'][task]
    
    # Convert to tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets based on dataset type
    if 'EMNIST' in dataset or dataset == 'MNIST-SVHN-FASHION':
        train_dataset = [(x, y) for x, y in zip(X_train, y_train_tensor)]
        test_dataset = [(x, y) for x, y in zip(X_test, y_test_tensor)]
    
    elif dataset == 'CIFAR100':
        img_size = 32
        train_transform = transforms.Compose([
            transforms.RandomCrop((img_size, img_size), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = TransformDataset(X_train, y_train_tensor, train_transform)
        test_dataset = TransformDataset(X_test, y_test_tensor, test_transform)
    
    else:
        train_dataset = [(x, y) for x, y in zip(X_train, y_train_tensor)]
        test_dataset = [(x, y) for x, y in zip(X_test, y_test_tensor)]
    
    # Compute label info
    unique_labels, counts = torch.unique(y_train_tensor, return_counts=True)
    label_info = {
        'labels': unique_labels.numpy().tolist(),
        'counts': counts.numpy().tolist()
    }
    
    return client_name, train_dataset, test_dataset, label_info
