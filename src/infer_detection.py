from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models.simple_detector import SimpleDetector
from utils.detection_metrics import filter_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D object detection inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_detection/model.pt")
    parser.add_argument("--input", type=str, help="Input point cloud (xyz.npy or all_features.npy)")
    parser.add_argument("--output-boxes", type=str, default="predicted_boxes.npy")
    parser.add_argument("--output-labels", type=str, default="predicted_labels.npy")
    parser.add_argument("--output-scores", type=str, default="predicted_scores.npy")
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def infer_single_cloud(
    features: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    objectness_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect objects in a single point cloud.
    
    Args:
        features: [num_points, num_features] numpy array
        model: trained model
        device: torch device
        objectness_threshold: threshold for objectness score
        score_threshold: threshold for final detection score
    
    Returns:
        boxes: [num_detections, 7] - (x,y,z,w,h,d,angle)
        labels: [num_detections] - class labels
        scores: [num_detections] - detection scores
    """
    # If features has only 3 channels (xyz), pad with zeros
    if features.shape[1] == 3:
        features = np.pad(features, ((0, 0), (0, 8)), mode='constant')
        print(f"Warning: Input has only 3 channels, padding to 11 channels")
    
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        batch_features = torch.from_numpy(features).float().unsqueeze(0).to(device)
        
        # Forward pass
        pred_boxes, pred_class_logits, pred_objectness = model(batch_features)
        
        # Filter predictions
        filtered_boxes, filtered_scores, filtered_labels = filter_predictions(
            pred_boxes.squeeze(0),
            pred_class_logits.squeeze(0),
            pred_objectness.squeeze(0),
            objectness_threshold=objectness_threshold,
            score_threshold=score_threshold,
        )
        
        # Convert to numpy
        boxes = filtered_boxes.cpu().numpy()
        labels = filtered_labels.cpu().numpy()
        scores = filtered_scores.cpu().numpy()
    
    return boxes, labels, scores


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
    print(f"Model config: num_classes={config.model.num_classes}, max_objects={config.model.max_objects}")
    
    # Create and load model
    model = SimpleDetector(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        max_objects=config.model.max_objects,
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
        # Normalize first 3 channels (xyz) to [-10, 10] range
        features[:, :3] = features[:, :3] * 5
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
    boxes, labels, scores = infer_single_cloud(
        features,
        model,
        device,
        objectness_threshold=args.objectness_threshold,
        score_threshold=args.score_threshold,
    )
    
    # Save predictions
    output_dir = Path(args.output_boxes).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(args.output_boxes, boxes)
    np.save(args.output_labels, labels)
    np.save(args.output_scores, scores)
    
    print(f"\nDetection results:")
    print(f"  Number of detections: {len(boxes)}")
    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Unique classes: {np.unique(labels) if len(labels) > 0 else []}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]" if len(scores) > 0 else "  No detections")
    
    print(f"\nSaved predictions:")
    print(f"  Boxes: {args.output_boxes}")
    print(f"  Labels: {args.output_labels}")
    print(f"  Scores: {args.output_scores}")
    
    # Print detailed detection results
    if len(boxes) > 0:
        print(f"\nDetailed results:")
        print("-" * 70)
        print(f"{'ID':<5} {'Class':<8} {'Score':<10} {'Center (x,y,z)':<30} {'Size (w,h,d)'}")
        print("-" * 70)
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x, y, z, w, h, d, angle = box
            print(f"{i:<5} {label:<8} {score:<10.4f} ({x:6.2f}, {y:6.2f}, {z:6.2f})  ({w:5.2f}, {h:5.2f}, {d:5.2f})")
        print("-" * 70)


if __name__ == "__main__":
    main()
