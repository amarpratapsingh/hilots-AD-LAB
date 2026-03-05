from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    num_points: int = 1024
    num_classes: int = 5
    noise_std: float = 0.02
    seed: int = 7


class SyntheticPointSegDataset(Dataset):
    """
    Synthetic dataset with 5 classes and 11-dimensional features.
    Classes: 0-4 (5 total classes)
    Features: x,y,z,r,g,b,nx,ny,nz,curvature,intensity (11 dims)
    """
    
    def __init__(
        self,
        num_samples: int,
        config: SyntheticConfig,
    ) -> None:
        self.num_samples = num_samples
        self.config = config

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        """Generate a random point cloud sample with 5 classes."""
        rng = np.random.default_rng(self.config.seed + idx)
        num_points = self.config.num_points
        
        # Generate random labels (5 classes with equal distribution)
        labels = rng.integers(0, self.config.num_classes, size=num_points, dtype=np.int64)
        
        # Generate random features (11 channels): x, y, z, r, g, b, nx, ny, nz, curvature, intensity
        coords = rng.standard_normal((num_points, 3)).astype(np.float32) * 0.5
        rgb = rng.uniform(0, 1, size=(num_points, 3)).astype(np.float32)
        
        # Normals: unit vectors
        normals = rng.standard_normal((num_points, 3)).astype(np.float32)
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        # Curvature: scalar value
        curvature = rng.uniform(0, 1, size=(num_points, 1)).astype(np.float32)
        
        # Intensity: distance from origin
        intensity = np.linalg.norm(coords, axis=1, keepdims=True).astype(np.float32)
        
        # Stack all features
        features = np.concatenate([coords, rgb, normals, curvature, intensity], axis=1).astype(np.float32)
        
        return torch.from_numpy(features).float(), torch.from_numpy(labels).long()
