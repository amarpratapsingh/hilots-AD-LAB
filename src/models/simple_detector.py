from __future__ import annotations

import torch
from torch import nn


class SimpleDetector(nn.Module):
    """
    Simple 3D object detector.
    
    Uses max pooling for global feature extraction, then predicts:
    - Bounding boxes: (x, y, z, w, h, d, angle)
    - Class logits
    - Objectness scores
    
    Input: [batch_size, num_points, in_channels]
    Output:
        - boxes: [batch_size, max_objects, 7]
        - class_logits: [batch_size, max_objects, num_classes]
        - objectness: [batch_size, max_objects]
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        max_objects: int = 10,
        hidden_dims: list | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        # Point-wise feature extraction
        layers = []
        prev_dim = in_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = hidden_dim
        
        self.point_encoder = nn.Sequential(*layers)
        
        # Global feature aggregation (max pooling)
        self.global_dim = hidden_dims[-1]
        
        # Detection heads
        # Box regression head: 7 values per box (x,y,z,w,h,d,angle)
        self.box_head = nn.Sequential(
            nn.Linear(self.global_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, max_objects * 7),
        )
        
        # Classification head: num_classes logits per object
        self.class_head = nn.Sequential(
            nn.Linear(self.global_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, max_objects * num_classes),
        )
        
        # Objectness head: 1 score per object (is there an object?)
        self.objectness_head = nn.Sequential(
            nn.Linear(self.global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, max_objects),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, num_points, in_channels]
        
        Returns:
            boxes: [batch_size, max_objects, 7]
            class_logits: [batch_size, max_objects, num_classes]
            objectness: [batch_size, max_objects]
        """
        batch_size, num_points, in_channels = x.shape
        
        # Extract point-wise features
        x = x.view(-1, in_channels)  # [batch_size * num_points, in_channels]
        x = self.point_encoder(x)  # [batch_size * num_points, global_dim]
        x = x.view(batch_size, num_points, -1)  # [batch_size, num_points, global_dim]
        
        # Global max pooling
        global_features = torch.max(x, dim=1)[0]  # [batch_size, global_dim]
        
        # Predict boxes
        boxes = self.box_head(global_features)  # [batch_size, max_objects * 7]
        boxes = boxes.view(batch_size, self.max_objects, 7)
        
        # Predict class logits
        class_logits = self.class_head(global_features)  # [batch_size, max_objects * num_classes]
        class_logits = class_logits.view(batch_size, self.max_objects, self.num_classes)
        
        # Predict objectness
        objectness = self.objectness_head(global_features)  # [batch_size, max_objects]
        
        return boxes, class_logits, objectness
