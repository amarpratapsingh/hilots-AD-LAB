from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data.synthetic_detection import DetectionConfig as SyntheticDetectionConfig, SyntheticDetectionDataset
from models.simple_detector import SimpleDetector
from models.detection_loss import DetectionLoss
from utils.detection_metrics import compute_detection_metrics, filter_predictions
from utils.seed import set_seed
from utils.config import DetectionConfig, DetectionDatasetConfig, DetectionModelConfig, DetectionTrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D object detection trainer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=200)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=str, default="checkpoints_detection")
    return parser.parse_args()


def collate_fn(batch):
    """Custom collate function to handle variable number of objects."""
    features_list = []
    boxes_list = []
    labels_list = []
    
    max_objects = 0
    for features, boxes, labels in batch:
        features_list.append(features)
        boxes_list.append(boxes)
        labels_list.append(labels)
        max_objects = max(max_objects, len(boxes))
    
    # Stack features
    features_batch = torch.stack(features_list, dim=0)
    
    # Pad boxes and labels to same length
    batch_size = len(batch)
    boxes_batch = torch.zeros((batch_size, max_objects, 7), dtype=torch.float32)
    labels_batch = torch.zeros((batch_size, max_objects), dtype=torch.int64)
    
    for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        num_obj = len(boxes)
        boxes_batch[i, :num_obj] = boxes
        labels_batch[i, :num_obj] = labels
    
    return features_batch, boxes_batch, labels_batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Create config
    config = DetectionConfig(
        dataset=DetectionDatasetConfig(
            num_points=args.num_points,
            num_classes=args.num_classes,
            max_objects=args.max_objects,
            num_samples_train=args.train_samples,
            num_samples_val=args.val_samples,
            seed=args.seed,
        ),
        model=DetectionModelConfig(
            in_channels=11,
            num_classes=args.num_classes,
            max_objects=args.max_objects,
            hidden_dims=[128, 256, 512],
            dropout=0.1,
        ),
        train=DetectionTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device="cuda" if not args.cpu else "cpu",
            seed=args.seed,
        ),
    )

    # Create datasets
    synthetic_config = SyntheticDetectionConfig(
        num_points=config.dataset.num_points,
        num_classes=config.dataset.num_classes,
        max_objects=config.dataset.max_objects,
        seed=config.dataset.seed,
    )
    
    train_ds = SyntheticDetectionDataset(args.train_samples, synthetic_config)
    val_ds = SyntheticDetectionDataset(args.val_samples, synthetic_config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Create model
    model = SimpleDetector(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        max_objects=config.model.max_objects,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss and optimizer
    criterion = DetectionLoss(
        num_classes=config.model.num_classes,
        box_loss_weight=config.train.box_loss_weight,
        class_loss_weight=config.train.class_loss_weight,
        objectness_loss_weight=config.train.objectness_loss_weight,
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    # Learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_box_loss = 0.0
        running_class_loss = 0.0
        running_obj_loss = 0.0

        for features, gt_boxes, gt_labels in train_loader:
            features = features.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_labels = gt_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_boxes, pred_class_logits, pred_objectness = model(features)

            # Compute loss
            loss_dict = criterion(pred_boxes, pred_class_logits, pred_objectness, gt_boxes, gt_labels)
            loss = loss_dict["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_box_loss += loss_dict["box_loss"].item()
            running_class_loss += loss_dict["class_loss"].item()
            running_obj_loss += loss_dict["objectness_loss"].item()

        avg_loss = running_loss / max(len(train_loader), 1)
        avg_box_loss = running_box_loss / max(len(train_loader), 1)
        avg_class_loss = running_class_loss / max(len(train_loader), 1)
        avg_obj_loss = running_obj_loss / max(len(train_loader), 1)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Validation
        model.eval()
        pred_boxes_list = []
        pred_scores_list = []
        pred_labels_list = []
        gt_boxes_list = []
        gt_labels_list = []
        
        with torch.no_grad():
            for features, gt_boxes, gt_labels in val_loader:
                features = features.to(device)
                
                pred_boxes, pred_class_logits, pred_objectness = model(features)
                
                # Process each sample in batch
                for i in range(len(features)):
                    # Filter predictions
                    filtered_boxes, filtered_scores, filtered_labels = filter_predictions(
                        pred_boxes[i],
                        pred_class_logits[i],
                        pred_objectness[i],
                        objectness_threshold=0.3,
                        score_threshold=0.3,
                    )
                    
                    pred_boxes_list.append(filtered_boxes.cpu())
                    pred_scores_list.append(filtered_scores.cpu())
                    pred_labels_list.append(filtered_labels.cpu())
                    
                    # Get ground truth (filter out padding)
                    valid_mask = (gt_boxes[i].sum(dim=-1) != 0)
                    gt_boxes_list.append(gt_boxes[i][valid_mask].cpu())
                    gt_labels_list.append(gt_labels[i][valid_mask].cpu())

        # Compute metrics
        metrics = compute_detection_metrics(
            pred_boxes_list,
            pred_scores_list,
            pred_labels_list,
            gt_boxes_list,
            gt_labels_list,
            num_classes=config.model.num_classes,
            iou_threshold=0.5,
        )
        
        mAP = metrics["mAP"]

        # Save best model
        if mAP > best_map:
            best_map = mAP
            checkpoint_path = out_dir / "model_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "mAP": mAP,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:03d} | loss {avg_loss:.4f} (box {avg_box_loss:.4f} "
            f"cls {avg_class_loss:.4f} obj {avg_obj_loss:.4f}) | "
            f"mAP {mAP:.4f} | lr {current_lr:.2e}"
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
    print(f"Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
