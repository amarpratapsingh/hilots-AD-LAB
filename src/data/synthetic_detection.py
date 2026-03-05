from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DetectionConfig:
    num_points: int = 1024
    num_classes: int = 3  # car, pedestrian, cyclist
    max_objects: int = 10  # Maximum objects per scene
    noise_std: float = 0.02
    seed: int = 7


class SyntheticDetectionDataset(Dataset):
    """
    Synthetic 3D object detection dataset.
    
    Returns:
        features: [num_points, 11] - point cloud features (x,y,z,r,g,b,nx,ny,nz,curvature,intensity)
        boxes: [num_objects, 7] - 3D boxes (x,y,z,w,h,d,angle)
        labels: [num_objects] - class labels (0 to num_classes-1)
    """
    
    def __init__(
        self,
        num_samples: int,
        config: DetectionConfig,
    ) -> None:
        self.num_samples = num_samples
        self.config = config

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        """Generate a random point cloud with 3D bounding boxes."""
        rng = np.random.default_rng(self.config.seed + idx)
        num_points = self.config.num_points
        
        # Generate random number of objects (1 to max_objects)
        num_objects = rng.integers(1, self.config.max_objects + 1)
        
        # Generate bounding boxes [x, y, z, w, h, d, angle]
        boxes = np.zeros((num_objects, 7), dtype=np.float32)
        
        # Random centers in [-10, 10] range
        boxes[:, :3] = rng.uniform(-10, 10, size=(num_objects, 3))
        
        # Random sizes (width, height, depth) in [0.5, 3.0] range
        boxes[:, 3:6] = rng.uniform(0.5, 3.0, size=(num_objects, 3))
        
        # Random rotation angle in [0, 2π]
        boxes[:, 6] = rng.uniform(0, 2 * np.pi, size=num_objects)
        
        # Generate class labels
        labels = rng.integers(0, self.config.num_classes, size=num_objects, dtype=np.int64)
        
        # Generate point cloud (sample points from boxes + background)
        points_per_object = num_points // (num_objects + 1)
        background_points = num_points - (points_per_object * num_objects)
        
        all_points = []
        
        # Generate points inside each box
        for i in range(num_objects):
            cx, cy, cz, w, h, d, angle = boxes[i]
            
            # Generate points in local box coordinates
            local_points = rng.uniform(-0.5, 0.5, size=(points_per_object, 3)).astype(np.float32)
            local_points[:, 0] *= w  # scale by width
            local_points[:, 1] *= h  # scale by height
            local_points[:, 2] *= d  # scale by depth
            
            # Rotate around z-axis
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            rotated_points = local_points @ rotation.T
            
            # Translate to center
            rotated_points += np.array([cx, cy, cz], dtype=np.float32)
            
            all_points.append(rotated_points)
        
        # Generate background points
        background = rng.uniform(-15, 15, size=(background_points, 3)).astype(np.float32)
        all_points.append(background)
        
        # Concatenate all points
        coords = np.concatenate(all_points, axis=0).astype(np.float32)
        
        # Shuffle points
        shuffle_idx = rng.permutation(num_points)
        coords = coords[shuffle_idx]
        
        # Generate RGB colors (class-dependent for visualization)
        rgb = np.zeros((num_points, 3), dtype=np.float32)
        rgb[:points_per_object * num_objects] = rng.uniform(0.3, 1.0, size=(points_per_object * num_objects, 3))
        rgb[points_per_object * num_objects:] = rng.uniform(0, 0.3, size=(background_points, 3))
        rgb = rgb[shuffle_idx]
        
        # Generate normals (random unit vectors)
        normals = rng.standard_normal((num_points, 3)).astype(np.float32)
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        # Generate curvature (random)
        curvature = rng.uniform(0, 1, size=(num_points, 1)).astype(np.float32)
        
        # Generate intensity (distance from origin)
        intensity = np.linalg.norm(coords, axis=1, keepdims=True).astype(np.float32)
        
        # Stack all features [num_points, 11]
        features = np.concatenate([coords, rgb, normals, curvature, intensity], axis=1).astype(np.float32)
        
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(boxes).float(),
            torch.from_numpy(labels).long()
        )
