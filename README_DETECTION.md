# 3D Object Detection for Point Clouds

This directory now contains both **semantic segmentation** and **3D object detection** implementations for point cloud data.

## Overview

### Semantic Segmentation (Original)
- **Task**: Per-point classification
- **Files**: `train.py`, `eval.py`, `infer.py`, `test_model.py`
- **Model**: PointMLP (69K parameters)
- **Output**: Class label for each point
- **Metrics**: mIoU, accuracy, per-class IoU

### 3D Object Detection (New)
- **Task**: Detect 3D bounding boxes and classify objects
- **Files**: `train_detection.py`, `eval_detection.py`, `infer_detection.py`, `test_detection.py`
- **Model**: SimpleDetector (852K parameters)
- **Output**: 3D boxes (x,y,z,w,h,d,angle), class labels, confidence scores
- **Metrics**: mAP (mean Average Precision), per-class AP

## Quick Start

### Testing

```bash
# Test segmentation model
python src/test_model.py

# Test detection model
python src/test_detection.py
```

### Training

```bash
# Train segmentation model (5 classes)
python src/train.py --epochs 20 --num-classes 5 --batch-size 4

# Train detection model (3 classes: car, pedestrian, cyclist)
python src/train_detection.py --epochs 30 --num-classes 3 --max-objects 10 --batch-size 4
```

### Evaluation

```bash
# Evaluate segmentation
python src/eval.py --checkpoint checkpoints/model.pt

# Evaluate detection
python src/eval_detection.py --checkpoint checkpoints_detection/model.pt
```

### Inference

```bash
# Segmentation inference
python src/infer.py --checkpoint checkpoints/model.pt --input your_pointcloud.npy

# Detection inference
python src/infer_detection.py --checkpoint checkpoints_detection/model.pt --input your_pointcloud.npy
```

## Architecture Comparison

| Component | Segmentation | Detection |
|-----------|-------------|-----------|
| **Dataset** | `SyntheticPointSegDataset` | `SyntheticDetectionDataset` |
| **Input** | [N, 11] point features | [N, 11] point features |
| **Output** | [N] class labels | Boxes [M, 7], Labels [M], Scores [M] |
| **Model** | PointMLP (per-point) | SimpleDetector (global features) |
| **Parameters** | 69,125 | 851,511 |
| **Loss** | CrossEntropyLoss | DetectionLoss (box + class + objectness) |
| **Metrics** | IoU, accuracy | mAP, AP per class |
| **Training** | ~20 epochs | ~30 epochs |

## Key Files

### Detection Implementation

1. **Dataset**: [`src/data/synthetic_detection.py`](src/data/synthetic_detection.py)
   - Generates synthetic 3D bounding boxes with class labels
   - Returns: point cloud features, boxes [num_objects, 7], labels

2. **Model**: [`src/models/simple_detector.py`](src/models/simple_detector.py)
   - Point-wise feature extraction → global max pooling
   - Three prediction heads: boxes, classification, objectness
   - 851K parameters (vs 69K for segmentation)

3. **Loss**: [`src/models/detection_loss.py`](src/models/detection_loss.py)
   - Box regression loss (Smooth L1)
   - Classification loss (Cross Entropy)
   - Objectness loss (Binary Cross Entropy)
   - Simple assignment strategy (first-N matching)

4. **Metrics**: [`src/utils/detection_metrics.py`](src/utils/detection_metrics.py)
   - mAP calculation with 11-point interpolation
   - 3D IoU computation (simplified)
   - Prediction filtering by confidence thresholds

5. **Config**: [`src/utils/config.py`](src/utils/config.py)
   - `DetectionConfig`, `DetectionDatasetConfig`, etc.
   - Separate configs for segmentation and detection

## Feature Comparison

### What Changed?

1. **Data Format**
   - **Before**: Point features → Point labels
   - **After**: Point features → 3D boxes + Object classes

2. **Model Architecture**
   - **Before**: Per-point MLP (processes each point independently)
   - **After**: Global feature extraction + detection heads

3. **Training Objectives**
   - **Before**: Classify each point correctly
   - **After**: Localize objects + classify them + determine objectness

4. **Evaluation**
   - **Before**: Per-point IoU and accuracy
   - **After**: mAP based on box IoU matching

### Common Elements

Both implementations share:
- 11-channel input features (x,y,z,RGB,normals,curvature,intensity)
- PyTorch training pipeline
- Cosine learning rate scheduling
- Configuration management via dataclasses
- Comprehensive test suites

## Example Usage

### Training Detection Model

```python
from data.synthetic_detection import DetectionConfig, SyntheticDetectionDataset
from models.simple_detector import SimpleDetector
from models.detection_loss import DetectionLoss

# Create dataset
config = DetectionConfig(num_points=1024, num_classes=3, max_objects=10)
dataset = SyntheticDetectionDataset(num_samples=200, config=config)

# Create model
model = SimpleDetector(in_channels=11, num_classes=3, max_objects=10)

# Train (see train_detection.py for full example)
```

### Running Inference

```python
from models.simple_detector import SimpleDetector
from utils.detection_metrics import filter_predictions
import torch
import numpy as np

# Load model
checkpoint = torch.load("checkpoints_detection/model.pt")
model = SimpleDetector(...)
model.load_state_dict(checkpoint["model_state"])

# Run inference
features = np.load("pointcloud.npy")  # [N, 11]
features = torch.from_numpy(features).float().unsqueeze(0)

pred_boxes, pred_logits, pred_obj = model(features)

# Filter predictions
boxes, scores, labels = filter_predictions(
    pred_boxes[0], pred_logits[0], pred_obj[0],
    objectness_threshold=0.5, score_threshold=0.5
)

print(f"Detected {len(boxes)} objects")
```

## Performance Notes

- **Training time**: Detection takes ~2-3x longer than segmentation (more parameters)
- **Memory usage**: Detection requires more GPU memory (~1.5GB vs ~500MB)
- **Inference speed**: Detection is faster per-scene (global features) but slower per-point
- **Accuracy**: mAP depends heavily on synthetic data quality and matching strategy

## Future Improvements

For production use, consider:
1. **Better assignment**: Use Hungarian matching or IoU-based assignment instead of first-N
2. **Real 3D IoU**: Implement proper 3D IoU with rotation handling
3. **Anchor-based detection**: Add anchor boxes for better localization
4. **NMS**: Implement non-maximum suppression for duplicate removal
5. **Real data**: Train on actual 3D detection datasets (KITTI, nuScenes, Waymo)
6. **Advanced architectures**: Replace SimpleDetector with PointNet++, VoxelNet, or PointPillars

## References

- Original segmentation implementation: See [README_ENHANCED.md](README_ENHANCED.md)
- Point cloud processing: [NumPy](https://numpy.org/) and [PyTorch](https://pytorch.org/)
- 3D detection theory: KITTI dataset paper, PointNet, VoxelNet

---

**Note**: This is a simplified educational implementation. For production 3D object detection, use established libraries like [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) or [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
