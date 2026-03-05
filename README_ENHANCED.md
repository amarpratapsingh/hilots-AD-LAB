# Enhanced Point Cloud Semantic Segmentation - Prototype

A simple yet functional point cloud semantic segmentation model built as a baseline for the HiLoTs project.

## ✨ Features

### Model Architecture
- **PointMLP**: Multi-layer perceptron for point-wise semantic segmentation
- **Multi-class support**: 5 semantic classes (expandable)
- **Rich features**: 11-dimensional input (x, y, z, r, g, b, nx, ny, nz, curvature, intensity)
- **69,125 parameters**: Lightweight and efficient

### Dataset
- **Synthetic point cloud generation**: For rapid prototyping and testing
- **Configurable**: Number of points, classes, and random seed
- **11-channel features**: Coordinates, RGB colors, normals, curvature, and intensity

### Training Pipeline
- **Cosine annealing LR scheduling**: Smooth learning rate decay
- **Per-class metrics**: IoU, precision, recall, F1-score
- **Best model checkpointing**: Automatic saving of best performing model
- **Config-based**: All hyperparameters configurable via dataclasses

### Data Augmentation (Implemented but not yet integrated)
- Random rotation around z-axis
- Random jitter (Gaussian noise)
- Random scaling
- Random point dropout

## 📁 Project Structure

```
prototype/src/
├── data/
│   ├── synthetic.py      # Synthetic dataset generation
│   └── transforms.py     # Data augmentation transforms
├── models/
│   └── point_mlp.py      # PointMLP model architecture
├── utils/
│   ├── config.py         # Configuration dataclasses
│   ├── metrics.py        # Evaluation metrics (IoU, accuracy)
│   └── seed.py           # Reproducibility utilities
├── train.py              # Training script
├── eval.py               # Evaluation script
├── infer.py              # Inference script
└── test_model.py         # Unit tests for model components
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy
```

### 2. Train the Model

```bash
python src/train.py --epochs 10 --num-classes 5 --batch-size 8 --train-samples 100 --val-samples 30
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 20)
- `--num-classes`: Number of semantic classes (default: 5)
- `--batch-size`: Batch size for training (default: 4)
- `--train-samples`: Number of training samples (default: 200)
- `--val-samples`: Number of validation samples (default: 50)
- `--num-points`: Points per sample (default: 1024)
- `--lr`: Learning rate (default: 1e-3)
- `--cpu`: Force CPU usage
- `--out-dir`: Output directory for checkpoints (default: checkpoints)

### 3. Evaluate the Model

```bash
python src/eval.py --checkpoint checkpoints/model.pt --val-samples 50
```

**Output:**
- Overall accuracy
- Mean IoU (mIoU)
- Per-class IoU, precision, recall, F1-score
- Confusion matrix

### 4. Run Inference

```bash
python src/infer.py --checkpoint checkpoints/model.pt --input your_pointcloud.npy --output predictions.npy
```

If no input is provided, synthetic test data will be generated automatically.

### 5. Run Tests

```bash
python src/test_model.py
```

## 📊 Example Results

**Training (5 epochs, 40 samples):**
```
Epoch 001 | loss 1.6598 | mIoU 0.0756 | acc 0.2048 | lr 9.05e-04
Epoch 002 | loss 1.6283 | mIoU 0.0914 | acc 0.2106 | lr 6.55e-04
Epoch 003 | loss 1.6224 | mIoU 0.0918 | acc 0.2132 | lr 3.45e-04
Epoch 004 | loss 1.6184 | mIoU 0.0993 | acc 0.2180 | lr 9.55e-05
Epoch 005 | loss 1.6154 | mIoU 0.1064 | acc 0.2219 | lr 0.00e+00
```

**Evaluation:**
```
Overall Accuracy: 0.2219
Mean mIoU:        0.1064
Per-class IoU:    [0.1276, 0.1868, 0.0429, 0.0935, 0.0810]
```

## 🔧 Configuration System

The model uses dataclasses for clean, type-safe configuration:

```python
from utils.config import Config, DatasetConfig, ModelConfig, TrainConfig

config = Config(
    dataset=DatasetConfig(
        num_points=1024,
        num_classes=5,
        seed=7
    ),
    model=ModelConfig(
        in_channels=11,
        num_classes=5,
        hidden_dims=[128, 256, 128],
        dropout=0.1
    ),
    train=TrainConfig(
        epochs=20,
        batch_size=4,
        learning_rate=1e-3,
        lr_schedule="cosine"
    )
)
```

## 🎯 Model Architecture Details

**Input:** `[batch_size, num_points, 11]`
- 11 channels: x, y, z, r, g, b, nx, ny, nz, curvature, intensity

**Architecture:**
```
Linear(11 → 128) → BatchNorm → ReLU → Dropout(0.1)
Linear(128 → 256) → BatchNorm → ReLU → Dropout(0.1)
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.1)
Linear(128 → 5)
```

**Output:** `[batch_size, num_points, num_classes]`
- Logits for each point and class

## 📈 Next Steps to Scale to Full HiLoTs

To evolve this prototype into a production-ready model:

### 1. **Add Real Dataset Loaders**
- Implement KITTI dataset loader
- Implement SemanticKITTI dataset loader
- Implement nuScenes dataset loader
- Add proper data preprocessing pipelines

### 2. **Upgrade Model Architecture**
- PointNet++ with hierarchical feature learning
- PointPillars for efficient voxelization
- Sparse 3D convolutions (MinkUNet, Cylinder3D)
- Attention mechanisms

### 3. **Enhance Training**
- Distributed training (DDP)
- Mixed precision training (AMP)
- Advanced augmentation strategies (MixUp, CutMix for 3D)
- Learning rate warmup and better schedulers

### 4. **Add Advanced Features**
- Multi-task learning (detection + segmentation)
- Multi-modal fusion (LiDAR + camera)
- Test-time augmentation
- Model ensembling

### 5. **Production Infrastructure**
- TensorBoard logging
- Weights & Biases integration
- YAML-based configuration files
- Automated hyperparameter tuning
- Docker containerization

### 6. **Evaluation & Analysis**
- Real-world dataset benchmarking
- Visualization tools for predictions
- Error analysis utilities
- Speed/accuracy trade-off analysis

## 🧪 Key Differences from Original Prototype

| Feature | Original | Enhanced |
|---------|----------|----------|
| Classes | 2 (binary) | 5 (multi-class) |
| Input Channels | 4 (x,y,z,intensity) | 11 (coords,RGB,normals,curvature,intensity) |
| Model | Simple MLP | MLP with BatchNorm and Dropout |
| Config | Hardcoded | Dataclass-based system |
| Metrics | Basic mIoU | Per-class IoU, precision, recall, F1 |
| Scheduler | None | Cosine annealing |
| Checkpointing | Final only | Best + final models |

## 🤝 Contributing

This is a prototype implementation. To integrate with the main HiLoTs repository:

1. Move components to appropriate `mmdet3d/` directories
2. Register models/datasets with MMDetection3D registry
3. Add proper configuration files to `configs/hilots/`
4. Implement proper data converters in `tools/dataset_converters/`

## 📝 License

Follows the same license as the parent HiLoTs project.

## 🙏 Acknowledgments

Built on top of:
- PyTorch for deep learning framework
- NumPy for numerical operations
- MMDetection3D architecture inspiration (parent project)
