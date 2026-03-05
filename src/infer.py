from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models.point_mlp import PointMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point cloud semantic segmentation inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--input", type=str, help="Input point cloud (xyz.npy or all_features.npy)")
    parser.add_argument("--output", type=str, default="predictions.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def infer_single_cloud(
    features: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Segment a single point cloud.
    
    Args:
        features: [num_points, num_features] numpy array
        model: trained model
        device: torch device
        batch_size: batch size for inference
    
    Returns:
        predictions: [num_points] class labels
    """
    num_points = len(features)
    predictions = np.zeros(num_points, dtype=np.int64)
    
    # If features has only 3 channels (xyz), pad with zeros
    if features.shape[1] == 3:
        features = np.pad(features, ((0, 0), (0, 8)), mode='constant')
        print(f"Warning: Input has only 3 channels, padding to 11 channels")
    
    model.eval()
    with torch.no_grad():
        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            batch_features = features[start_idx:end_idx]
            
            # Add batch dimension and move to device
            batch_features = torch.from_numpy(batch_features).float().unsqueeze(0)
            batch_features = batch_features.to(device)
            
            # Forward pass
            logits = model(batch_features)
            batch_preds = logits.argmax(dim=-1).squeeze(0)
            
            # Store predictions
            predictions[start_idx:end_idx] = batch_preds.cpu().numpy()
    
    return predictions


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    
    if config is None:
        print("Error: Config not found in checkpoint")
        return
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Model config: num_classes={config.model.num_classes}")
    
    # Create and load model
    model = PointMLP(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    
    # Create synthetic test data if no input provided
    if args.input is None:
        print("No input provided. Creating synthetic test data...")
        np.random.seed(42)
        num_points = 1024
        # Simple synthetic data: random points with 11 features
        features = np.random.randn(num_points, 11).astype(np.float32)
        # Normalize first 3 channels (xyz) to [-1, 1]
        features[:, :3] = np.tanh(features[:, :3])
    else:
        print(f"Loading point cloud from {args.input}")
        features = np.load(args.input)
        if not isinstance(features, np.ndarray):
            print(f"Error: Expected numpy array, got {type(features)}")
            return
        if features.dtype not in [np.float32, np.float64]:
            features = features.astype(np.float32)
    
    print(f"Input shape: {features.shape}")
    
    # Run inference
    print("Running inference...")
    predictions = infer_single_cloud(
        features,
        model,
        device,
        batch_size=args.batch_size,
    )
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions)
    
    print(f"Saved predictions to {output_path}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique classes: {np.unique(predictions)}")
    print(f"Class distribution: {np.bincount(predictions)}")


if __name__ == "__main__":
    main()
