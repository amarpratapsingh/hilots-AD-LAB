from __future__ import annotations

import numpy as np
import torch


class PointCloudTransform:
    """Base class for point cloud transformations."""
    
    def __call__(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        raise NotImplementedError


class RandomRotation(PointCloudTransform):
    """Randomly rotate point cloud."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        if np.random.rand() < self.prob:
            # Random rotation matrix
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Rotate around z-axis
            rotation = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            # Apply rotation to xyz and normals
            coords = features[:, :3].numpy()
            normals = features[:, 6:9].numpy()
            
            coords = coords @ rotation.T
            normals = normals @ rotation.T
            
            features = features.clone()
            features[:, :3] = torch.from_numpy(coords).float()
            features[:, 6:9] = torch.from_numpy(normals).float()
        
        return features, labels


class RandomJitter(PointCloudTransform):
    """Add random noise to point coordinates."""
    
    def __init__(self, std: float = 0.01, prob: float = 0.5):
        self.std = std
        self.prob = prob
    
    def __call__(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        if np.random.rand() < self.prob:
            features = features.clone()
            noise = torch.randn_like(features[:, :3]) * self.std
            features[:, :3] += noise
        
        return features, labels


class RandomScale(PointCloudTransform):
    """Randomly scale point cloud."""
    
    def __init__(self, scale_range: tuple = (0.8, 1.2), prob: float = 0.5):
        self.scale_range = scale_range
        self.prob = prob
    
    def __call__(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        if np.random.rand() < self.prob:
            scale = np.random.uniform(*self.scale_range)
            features = features.clone()
            features[:, :3] *= scale
            # Scale intensity and curvature too
            features[:, 9] *= scale  # curvature
            features[:, 10] *= scale  # intensity
        
        return features, labels


class RandomDropout(PointCloudTransform):
    """Randomly drop some points."""
    
    def __init__(self, drop_rate: float = 0.1, prob: float = 0.5):
        self.drop_rate = drop_rate
        self.prob = prob
    
    def __call__(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        if np.random.rand() < self.prob:
            num_points = len(features)
            num_keep = int(num_points * (1 - self.drop_rate))
            indices = np.random.choice(num_points, num_keep, replace=False)
            
            features = features[indices]
            labels = labels[indices]
            
            # Pad back to original length if needed
            if len(features) < num_points:
                pad_idx = np.random.choice(len(features), num_points - len(features))
                features = torch.cat([features, features[pad_idx]], dim=0)
                labels = torch.cat([labels, labels[pad_idx]], dim=0)
        
        return features, labels


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        for transform in self.transforms:
            features, labels = transform(features, labels)
        return features, labels
