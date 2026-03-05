from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.synthetic import SyntheticConfig, SyntheticPointSegDataset
from models.point_mlp import PointMLP
from utils.metrics import confusion_matrix, compute_accuracy, compute_iou
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced point segmentation evaluator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def print_per_class_metrics(cm: torch.Tensor, num_classes: int) -> None:
    """Print per-class precision, recall, and F1 score."""
    print("\nPer-class metrics:")
    print("-" * 70)
    print(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 70)
    
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        print(f"Class {c:<5} {precision:<15.4f} {recall:<15.4f} {f1:<15.4f}")
    
    print("-" * 70)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    
    if config is None:
        print("Warning: Config not found in checkpoint. Using default config.")
        config = SyntheticConfig()
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Model config: num_classes={config.model.num_classes}, "
          f"num_points={config.dataset.num_points}")

    # Set seed for reproducibility
    set_seed(config.dataset.seed)

    # Create validation dataset
    synthetic_config = SyntheticConfig(
        num_points=config.dataset.num_points,
        num_classes=config.dataset.num_classes,
        seed=config.dataset.seed,
    )
    
    val_ds = SyntheticPointSegDataset(args.val_samples, synthetic_config)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create and load model
    model = PointMLP(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Run inference
    cm = torch.zeros(
        (config.model.num_classes, config.model.num_classes),
        dtype=torch.int64
    )
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(val_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            preds = logits.argmax(dim=-1)
            
            cm += confusion_matrix(
                preds.cpu(),
                labels.cpu(),
                num_classes=config.model.num_classes
            )
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size} samples")

    # Compute metrics
    iou = compute_iou(cm)
    miou = iou.mean().item()
    acc = compute_accuracy(cm)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Mean mIoU:        {miou:.4f}")
    print(f"Per-class IoU:    {[f'{x:.4f}' for x in iou.tolist()]}")
    
    # Print per-class metrics
    print_per_class_metrics(cm, config.model.num_classes)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm.numpy())
    print("=" * 70)


if __name__ == "__main__":
    main()
