from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for dataset."""
    num_points: int = 1024
    num_classes: int = 5
    num_samples_train: int = 200
    num_samples_val: int = 50
    seed: int = 7


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    in_channels: int = 11  # x,y,z,r,g,b,nx,ny,nz,curvature,intensity
    num_classes: int = 5
    hidden_dims: list = field(default_factory=lambda: [128, 256, 128])
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Configuration for training."""
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_schedule: str = "cosine"  # "constant" or "cosine"
    use_class_weights: bool = True
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 7


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    batch_size: int = 4
    device: str = "cuda"


@dataclass
class Config:
    """Complete configuration."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"


# Detection-specific configurations
@dataclass
class DetectionDatasetConfig:
    """Configuration for detection dataset."""
    num_points: int = 1024
    num_classes: int = 3  # car, pedestrian, cyclist
    max_objects: int = 10
    num_samples_train: int = 200
    num_samples_val: int = 50
    seed: int = 7


@dataclass
class DetectionModelConfig:
    """Configuration for detection model architecture."""
    in_channels: int = 11  # x,y,z,r,g,b,nx,ny,nz,curvature,intensity
    num_classes: int = 3
    max_objects: int = 10
    hidden_dims: list = field(default_factory=lambda: [128, 256, 512])
    dropout: float = 0.1


@dataclass
class DetectionTrainConfig:
    """Configuration for detection training."""
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_schedule: str = "cosine"
    device: str = "cuda"
    seed: int = 7
    # Loss weights
    box_loss_weight: float = 5.0
    class_loss_weight: float = 1.0
    objectness_loss_weight: float = 2.0


@dataclass
class DetectionEvalConfig:
    """Configuration for detection evaluation."""
    batch_size: int = 4
    device: str = "cuda"
    iou_threshold: float = 0.5
    objectness_threshold: float = 0.5
    score_threshold: float = 0.5


@dataclass
class DetectionConfig:
    """Complete detection configuration."""
    dataset: DetectionDatasetConfig = field(default_factory=DetectionDatasetConfig)
    model: DetectionModelConfig = field(default_factory=DetectionModelConfig)
    train: DetectionTrainConfig = field(default_factory=DetectionTrainConfig)
    eval: DetectionEvalConfig = field(default_factory=DetectionEvalConfig)
    checkpoint_dir: str = "checkpoints_detection"
    output_dir: str = "outputs_detection"
