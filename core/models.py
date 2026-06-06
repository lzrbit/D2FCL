#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural network models for DCFCL.

This module contains the model architectures used in the paper:
- SimpleCNN: A simple convolutional network for EMNIST/MNIST
- ResNet18: ResNet-18 with CBAM attention for CIFAR100
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# =============================================================================
# Simple CNN for EMNIST/MNIST
# =============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN classifier for EMNIST-Letters and similar datasets.
    
    Architecture:
        Conv(1/3 -> 64) -> Conv(64 -> 128) -> Conv(128 -> 256) -> FC(feature_dim) -> FC(num_classes)
    """
    
    def __init__(self, image_size: int, in_channels: int, num_classes: int, 
                 feature_dim: int = 512, channel_size: int = 64):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.channel_size = channel_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, channel_size, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        
        # Calculate flattened size
        self.flat_size = (image_size // 8) ** 2 * channel_size * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier."""
        # No ReLU between conv layers (matches original S_ConvNet architecture)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Returns:
            tuple: (probabilities, features, logits)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        return probs, features, logits
    
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (alias for extract_features)."""
        return self.extract_features(x)
    
    def fc(self, x: torch.Tensor) -> torch.Tensor:
        """Apply classifier to features (returns logits)."""
        return self.classifier(x)

class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, in_planes: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class BasicBlockCBAM(nn.Module):
    """Basic ResNet block with CBAM attention."""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # CBAM attention
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return self.relu(out)


class ResNet18CBAM(nn.Module):
    """
    ResNet-18 with CBAM attention for CIFAR100.
    
    Modified for smaller input sizes (32x32).
    """
    
    def __init__(self, num_classes: int = 100, feature_dim: int = 512):
        super().__init__()
        self.in_planes = 64
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Initial conv layer (modified for CIFAR: kernel=3, stride=1, no maxpool)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, planes: int, num_blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            
        layers = [BasicBlockCBAM(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlockCBAM(planes, planes))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Returns:
            tuple: (probabilities, features, logits)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        return probs, features, logits
    
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (alias)."""
        return self.extract_features(x)
    
    def fc(self, x: torch.Tensor) -> torch.Tensor:
        """Apply classifier to features (returns logits)."""
        return self.classifier(x)


# =============================================================================
# Model Wrapper for DCFCL
# =============================================================================

class DCFCLModel(nn.Module):
    """
    Wrapper model for DCFCL that provides unified interface.
    
    Attributes:
        classifier: The underlying neural network
        classifier_optimizer: Adam optimizer for training
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create backbone based on dataset
        self.classifier = self._create_backbone()
        
        # Create optimizer
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
        
    def _create_backbone(self) -> nn.Module:
        """Create backbone network based on config."""
        dataset = self.config.dataset
        num_classes = self.config.num_classes
        feature_dim = self.config.feature_size
        
        if dataset in ['EMNIST-Letters', 'EMNIST-Letters-shuffle']:
            return SimpleCNN(
                image_size=28,
                in_channels=1,
                num_classes=num_classes,
                feature_dim=feature_dim
            )
        elif dataset == 'CIFAR100':
            return ResNet18CBAM(
                num_classes=num_classes,
                feature_dim=feature_dim
            )
        elif dataset == 'MNIST-SVHN-FASHION':
            return SimpleCNN(
                image_size=32,
                in_channels=3,
                num_classes=num_classes,
                feature_dim=feature_dim
            )
        elif dataset == 'TEST-noniid':
            # Simple MLP for testing
            return SimpleMLP(
                input_dim=784,
                hidden_dim=20,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through classifier."""
        return self.classifier(x)
    
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        return self.classifier.feature(x)
    
    def fc(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final classifier."""
        return self.classifier.fc(x)
    
    def to(self, device):
        """Move model to device."""
        self.classifier = self.classifier.to(device)
        return self
    
    def parameters(self):
        """Get model parameters."""
        return self.classifier.parameters()
    
    def named_parameters(self):
        """Get named parameters."""
        for name, param in self.classifier.named_parameters():
            yield f'classifier.{name}', param


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))
    
    def forward(self, x: torch.Tensor) -> tuple:
        features = self.extract_features(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        return probs, features, logits
    
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)
    
    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.classifier(x), dim=1)


def create_model(config) -> DCFCLModel:
    """Factory function to create DCFCL model."""
    return DCFCLModel(config)
