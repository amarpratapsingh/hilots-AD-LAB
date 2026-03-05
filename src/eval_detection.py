from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.synthetic_detection import DetectionConfig as SyntheticDetectionConfig, SyntheticDetectionDataset
from models.simple_detector import SimpleDetector
from utils.detection_metrics import compute_detection_metrics, filter_predictions
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D object detection evaluator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_detection/model.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
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
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    
    if config is None:
        print("Warning: Config not found in checkpoint.")
        return
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Model config: num_classes={config.model.num_classes}, "
          f"max_objects={config.model.max_objects}")

    # Set seed for reproducibility
    set_seed(config.dataset.seed)

    # Create validation dataset
    synthetic_config = SyntheticDetectionConfig(
        num_points=config.dataset.num_points,
        num_classes=config.dataset.num_classes,
        max_objects=config.dataset.max_objects,
        seed=config.dataset.seed,
    )
    
    val_ds = SyntheticDetectionDataset(args.val_samples, synthetic_config)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

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
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Run inference
    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, (features, gt_boxes, gt_labels) in enumerate(val_loader):
            features = features.to(device)
            
            pred_boxes, pred_class_logits, pred_objectness = model(features)
            
            # Process each sample in batch
            for i in range(len(features)):
                # Filter predictions
                filtered_boxes, filtered_scores, filtered_labels = filter_predictions(
                    pred_boxes[i],
                    pred_class_logits[i],
                    pred_objectness[i],
                    objectness_threshold=args.objectness_threshold,
                    score_threshold=args.score_threshold,
                )
                
                pred_boxes_list.append(filtered_boxes.cpu())
                pred_scores_list.append(filtered_scores.cpu())
                pred_labels_list.append(filtered_labels.cpu())
                
                # Get ground truth (filter out padding)
                valid_mask = (gt_boxes[i].sum(dim=-1) != 0)
                gt_boxes_list.append(gt_boxes[i][valid_mask].cpu())
                gt_labels_list.append(gt_labels[i][valid_mask].cpu())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size} samples")

    # Compute metrics
    metrics = compute_detection_metrics(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        gt_boxes_list,
        gt_labels_list,
        num_classes=config.model.num_classes,
        iou_threshold=args.iou_threshold,
    )

    print("\n" + "=" * 70)
    print("DETECTION EVALUATION RESULTS")
    print("=" * 70)
    print(f"mAP@{args.iou_threshold}: {metrics['mAP']:.4f}")
    print(f"\nPer-class Average Precision:")
    print("-" * 70)
    for key, value in metrics.items():
        if key.startswith("AP_class_"):
            class_id = key.split("_")[-1]
            print(f"  Class {class_id}: {value:.4f}")
    
    # Statistics
    total_predictions = sum(len(boxes) for boxes in pred_boxes_list)
    total_ground_truth = sum(len(boxes) for boxes in gt_boxes_list)
    avg_predictions = total_predictions / len(pred_boxes_list)
    avg_ground_truth = total_ground_truth / len(gt_boxes_list)
    
    print("-" * 70)
    print(f"Total predictions: {total_predictions}")
    print(f"Total ground truth: {total_ground_truth}")
    print(f"Avg predictions per sample: {avg_predictions:.2f}")
    print(f"Avg ground truth per sample: {avg_ground_truth:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
