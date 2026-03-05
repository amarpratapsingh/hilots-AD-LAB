#!/usr/bin/env python
"""Quick test script to verify the model works."""

import torch
from data.synthetic import SyntheticConfig, SyntheticPointSegDataset
from models.point_mlp import PointMLP

print("Testing enhanced semantic segmentation model...")

# Create config
config = SyntheticConfig(num_points=512, num_classes=5, seed=42)

# Create dataset
print("\n1. Creating dataset...")
dataset = SyntheticPointSegDataset(num_samples=5, config=config)
features, labels = dataset[0]
print(f"   Features shape: {features.shape} (expected: [512, 11])")
print(f"   Labels shape: {labels.shape} (expected: [512])")
print(f"   Unique classes: {torch.unique(labels).tolist()}")
assert features.shape == (512, 11), "Feature shape mismatch!"
assert labels.shape == (512,), "Label shape mismatch!"
print("   ✓ Dataset test passed!")

# Create model
print("\n2. Creating model...")
model = PointMLP(
    in_channels=11,
    num_classes=5,
    hidden_dims=[128, 256, 128],
    dropout=0.1,
)
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("   ✓ Model creation passed!")

# Test forward pass
print("\n3. Testing forward pass...")
batch_features = features.unsqueeze(0)  # Add batch dimension
logits = model(batch_features)
print(f"   Logits shape: {logits.shape} (expected: [1, 512, 5])")
assert logits.shape == (1, 512, 5), "Logits shape mismatch!"
print("   ✓ Forward pass test passed!")

# Test loss computation
print("\n4. Testing loss computation...")
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(logits.view(-1, 5), labels.view(-1))
print(f"   Loss: {loss.item():.4f}")
assert loss.item() > 0, "Loss should be positive!"
print("   ✓ Loss computation test passed!")

# Test evaluation metrics
print("\n5. Computing evaluation metrics...")
from utils.metrics import confusion_matrix, compute_iou, compute_accuracy
preds = logits.argmax(dim=-1).squeeze(0)
cm = confusion_matrix(preds, labels, num_classes=5)
iou = compute_iou(cm)
acc = compute_accuracy(cm)
print(f"   Confusion matrix shape: {cm.shape}")
print(f"   Per-class mIoU: {iou.tolist()}")
print(f"   Overall accuracy: {acc:.4f}")
print("   ✓ Metrics computation test passed!")

print("\n✅ All tests passed! The model is working correctly.")
print("\nNext steps:")
print("  1. Run: python src/train.py --epochs 10 --num-classes 5")
print("  2. Run: python src/eval.py --checkpoint checkpoints/model.pt")
print("  3. Run: python src/infer.py --checkpoint checkpoints/model.pt")
