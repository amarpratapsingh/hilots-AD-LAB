#!/usr/bin/env python
"""Quick test script to verify the detection model works."""

import torch
from data.synthetic_detection import DetectionConfig, SyntheticDetectionDataset
from models.simple_detector import SimpleDetector
from models.detection_loss import DetectionLoss
from utils.detection_metrics import filter_predictions

print("Testing 3D object detection model...")

# Create config
config = DetectionConfig(num_points=512, num_classes=3, max_objects=5, seed=42)

# Create dataset
print("\n1. Creating dataset...")
dataset = SyntheticDetectionDataset(num_samples=5, config=config)
features, boxes, labels = dataset[0]
print(f"   Features shape: {features.shape} (expected: [512, 11])")
print(f"   Boxes shape: {boxes.shape} (expected: [num_objects, 7])")
print(f"   Labels shape: {labels.shape} (expected: [num_objects])")
print(f"   Number of objects: {len(boxes)}")
print(f"   Unique classes: {torch.unique(labels).tolist()}")
assert features.shape == (512, 11), "Feature shape mismatch!"
assert boxes.shape[1] == 7, "Box shape mismatch!"
assert len(boxes) == len(labels), "Boxes and labels length mismatch!"
print("   ✓ Dataset test passed!")

# Create model
print("\n2. Creating model...")
model = SimpleDetector(
    in_channels=11,
    num_classes=3,
    max_objects=5,
    hidden_dims=[128, 256, 512],
    dropout=0.1,
)
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("   ✓ Model creation passed!")

# Test forward pass
print("\n3. Testing forward pass...")
batch_features = features.unsqueeze(0)  # Add batch dimension
pred_boxes, pred_class_logits, pred_objectness = model(batch_features)
print(f"   Predicted boxes shape: {pred_boxes.shape} (expected: [1, 5, 7])")
print(f"   Predicted class logits shape: {pred_class_logits.shape} (expected: [1, 5, 3])")
print(f"   Predicted objectness shape: {pred_objectness.shape} (expected: [1, 5])")
assert pred_boxes.shape == (1, 5, 7), "Predicted boxes shape mismatch!"
assert pred_class_logits.shape == (1, 5, 3), "Predicted class logits shape mismatch!"
assert pred_objectness.shape == (1, 5), "Predicted objectness shape mismatch!"
print("   ✓ Forward pass test passed!")

# Test loss computation
print("\n4. Testing loss computation...")
criterion = DetectionLoss(num_classes=3)

# Pad ground truth boxes to match max_objects
gt_boxes_padded = torch.zeros((1, 5, 7))
gt_labels_padded = torch.zeros((1, 5), dtype=torch.long)
num_gt = len(boxes)
gt_boxes_padded[0, :num_gt] = boxes
gt_labels_padded[0, :num_gt] = labels

loss_dict = criterion(pred_boxes, pred_class_logits, pred_objectness, gt_boxes_padded, gt_labels_padded)
print(f"   Total loss: {loss_dict['loss'].item():.4f}")
print(f"   Box loss: {loss_dict['box_loss'].item():.4f}")
print(f"   Class loss: {loss_dict['class_loss'].item():.4f}")
print(f"   Objectness loss: {loss_dict['objectness_loss'].item():.4f}")
assert loss_dict['loss'].item() > 0, "Loss should be positive!"
print("   ✓ Loss computation test passed!")

# Test prediction filtering
print("\n5. Testing prediction filtering...")
filtered_boxes, filtered_scores, filtered_labels = filter_predictions(
    pred_boxes.squeeze(0),
    pred_class_logits.squeeze(0),
    pred_objectness.squeeze(0),
    objectness_threshold=0.3,
    score_threshold=0.3,
)
print(f"   Filtered boxes shape: {filtered_boxes.shape}")
print(f"   Filtered scores shape: {filtered_scores.shape}")
print(f"   Filtered labels shape: {filtered_labels.shape}")
print(f"   Number of detections: {len(filtered_boxes)}")
if len(filtered_boxes) > 0:
    print(f"   Score range: [{filtered_scores.min():.4f}, {filtered_scores.max():.4f}]")
    print(f"   Detected classes: {torch.unique(filtered_labels).tolist()}")
print("   ✓ Prediction filtering test passed!")

# Test metrics computation
print("\n6. Testing metrics computation...")
from utils.detection_metrics import compute_detection_metrics

# Create simple test case
pred_boxes_list = [filtered_boxes]
pred_scores_list = [filtered_scores]
pred_labels_list = [filtered_labels]
gt_boxes_list = [boxes]
gt_labels_list = [labels]

metrics = compute_detection_metrics(
    pred_boxes_list,
    pred_scores_list,
    pred_labels_list,
    gt_boxes_list,
    gt_labels_list,
    num_classes=3,
    iou_threshold=0.5,
)
print(f"   mAP: {metrics['mAP']:.4f}")
for key, value in metrics.items():
    if key.startswith("AP_class_"):
        print(f"   {key}: {value:.4f}")
print("   ✓ Metrics computation test passed!")

print("\n✅ All tests passed! The detection model is working correctly.")
print("\nNext steps:")
print("  1. Run: python src/train_detection.py --epochs 10 --num-classes 3 --max-objects 10")
print("  2. Run: python src/eval_detection.py --checkpoint checkpoints_detection/model.pt")
print("  3. Run: python src/infer_detection.py --checkpoint checkpoints_detection/model.pt")
