from __future__ import annotations

import torch


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)

    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def compute_iou(cm: torch.Tensor) -> torch.Tensor:
    intersection = torch.diag(cm)
    union = cm.sum(dim=0) + cm.sum(dim=1) - intersection
    return intersection.float() / union.clamp_min(1).float()


def compute_accuracy(cm: torch.Tensor) -> float:
    correct = torch.diag(cm).sum().item()
    total = cm.sum().item()
    return float(correct / max(total, 1))
