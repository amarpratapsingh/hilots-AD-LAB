from __future__ import annotations

import torch
from torch import nn


class PointMLP(nn.Module):
    """
    Multi-class point cloud semantic segmentation using MLP.
    
    Input: [batch_size, num_points, in_channels]
    - in_channels: 11 (x,y,z,r,g,b,nx,ny,nz,curvature,intensity)
    
    Output: [batch_size, num_points, num_classes]
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dims: list | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        
        self.num_classes = num_classes
        
        # Build MLP backbone
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
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_points, in_channels]
        
        Returns:
            logits: [batch_size, num_points, num_classes]
        """
        batch_size, num_points, in_channels = x.shape
        
        # Reshape to [batch_size * num_points, in_channels]
        x = x.view(-1, in_channels)
        
        # Forward pass
        logits = self.net(x)
        
        # Reshape back to [batch_size, num_points, num_classes]
        logits = logits.view(batch_size, num_points, self.num_classes)
        
        return logits
