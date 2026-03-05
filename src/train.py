from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from data.synthetic import SyntheticConfig, SyntheticPointSegDataset
from data.transforms import Compose, RandomRotation, RandomJitter, RandomScale
from models.point_mlp import PointMLP
from utils.metrics import confusion_matrix, compute_accuracy, compute_iou
from utils.seed import set_seed
from utils.config import Config, DatasetConfig, ModelConfig, TrainConfig, EvalConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced point segmentation trainer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=200)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--use-class-weights", action="store_true", default=True)
    return parser.parse_args()


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """Compute class weights (using uniform weights for speed)."""
    return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Create config
    config = Config(
        dataset=DatasetConfig(
            num_points=args.num_points,
            num_classes=args.num_classes,
            num_samples_train=args.train_samples,
            num_samples_val=args.val_samples,
            seed=args.seed,
        ),
        model=ModelConfig(
            in_channels=11,  # x,y,z,r,g,b,nx,ny,nz,curvature,intensity
            num_classes=args.num_classes,
            hidden_dims=[128, 256, 128],
            dropout=0.1,
        ),
        train=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lr_schedule=args.lr_schedule,
            use_class_weights=args.use_class_weights,
            device="cuda" if not args.cpu else "cpu",
            seed=args.seed,
        ),
    )

    # Create datasets
    synthetic_config = SyntheticConfig(
        num_points=config.dataset.num_points,
        num_classes=config.dataset.num_classes,
        seed=config.dataset.seed,
    )
    
    train_ds = SyntheticPointSegDataset(args.train_samples, synthetic_config)
    val_ds = SyntheticPointSegDataset(args.val_samples, synthetic_config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = PointMLP(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Compute class weights
    class_weights = None
    if config.train.use_class_weights:
        class_weights = compute_class_weights(train_ds, config.model.num_classes)
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"Class weights: {class_weights.tolist()}")

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=1e-5,
    )

    # Learning rate scheduler
    if config.train.lr_schedule == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            # Reshape for augmentation
            batch_size, num_points, num_channels = features.shape
            
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(features)
            
            # Reshape for loss computation
            loss = criterion(
                logits.view(-1, config.model.num_classes),
                labels.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Validation
        model.eval()
        cm = torch.zeros((config.model.num_classes, config.model.num_classes), dtype=torch.int64)
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                logits = model(features)
                preds = logits.argmax(dim=-1)
                
                cm += confusion_matrix(
                    preds.cpu(),
                    labels.cpu(),
                    num_classes=config.model.num_classes
                )

        iou = compute_iou(cm)
        miou = iou.mean().item()
        acc = compute_accuracy(cm)

        # Save best model
        if miou > best_miou:
            best_miou = miou
            checkpoint_path = out_dir / "model_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "miou": miou,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:03d} | loss {avg_loss:.4f} | mIoU {miou:.4f} | "
            f"acc {acc:.4f} | lr {current_lr:.2e}"
        )

    # Save final model
    checkpoint_path = out_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": args.epochs,
        },
        checkpoint_path,
    )
    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
