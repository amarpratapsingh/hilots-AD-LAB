from __future__ import annotations

import torch
import numpy as np


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision (AP) from recall-precision curve.
    
    Uses 11-point interpolation method.
    """
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def compute_iou_3d_boxes(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute simplified 3D IoU between boxes.
    
    Args:
        boxes1: [N, 7] (x,y,z,w,h,d,angle)
        boxes2: [M, 7] (x,y,z,w,h,d,angle)
    
    Returns:
        iou: [N, M]
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=boxes1.device)
    
    # Extract centers and sizes
    centers1 = boxes1[:, :3].unsqueeze(1)  # [N, 1, 3]
    centers2 = boxes2[:, :3].unsqueeze(0)  # [1, M, 3]
    sizes1 = boxes1[:, 3:6].unsqueeze(1)  # [N, 1, 3]
    sizes2 = boxes2[:, 3:6].unsqueeze(0)  # [1, M, 3]
    
    # Compute distance between centers
    dist = torch.norm(centers1 - centers2, dim=-1)  # [N, M]
    
    # Compute overlap in each dimension (simplified)
    half_sizes1 = sizes1 / 2
    half_sizes2 = sizes2 / 2
    
    # Min of (half_size1 + half_size2 - distance) for each dimension
    overlap = torch.clamp(half_sizes1 + half_sizes2 - torch.abs(centers1 - centers2), min=0)
    intersection = overlap.prod(dim=-1)  # [N, M]
    
    # Compute volumes
    volume1 = sizes1.prod(dim=-1)  # [N, 1]
    volume2 = sizes2.prod(dim=-1)  # [1, M]
    
    # IoU
    union = volume1 + volume2 - intersection
    iou = intersection / (union + 1e-8)
    
    return iou


def compute_detection_metrics(
    pred_boxes_list: list[torch.Tensor],  # List of [num_pred, 7] per sample
    pred_scores_list: list[torch.Tensor],  # List of [num_pred] per sample
    pred_labels_list: list[torch.Tensor],  # List of [num_pred] per sample
    gt_boxes_list: list[torch.Tensor],  # List of [num_gt, 7] per sample
    gt_labels_list: list[torch.Tensor],  # List of [num_gt] per sample
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute detection metrics (mAP, precision, recall).
    
    Returns:
        Dictionary with 'mAP', 'mAP50', per-class AP, etc.
    """
    all_aps = []
    per_class_ap = {}
    
    for class_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        sample_ids_pred = []
        sample_ids_gt = []
        
        for sample_id, (pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels) in enumerate(
            zip(pred_boxes_list, pred_scores_list, pred_labels_list, gt_boxes_list, gt_labels_list)
        ):
            # Filter predictions for this class
            class_mask = pred_labels == class_id
            if class_mask.any():
                all_pred_boxes.append(pred_boxes[class_mask])
                all_pred_scores.append(pred_scores[class_mask])
                sample_ids_pred.extend([sample_id] * class_mask.sum().item())
            
            # Filter ground truths for this class
            gt_class_mask = gt_labels == class_id
            if gt_class_mask.any():
                all_gt_boxes.append(gt_boxes[gt_class_mask])
                sample_ids_gt.extend([sample_id] * gt_class_mask.sum().item())
        
        if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
            per_class_ap[f"AP_class_{class_id}"] = 0.0
            continue
        
        # Concatenate
        all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
        all_pred_scores = torch.cat(all_pred_scores, dim=0)
        all_gt_boxes = torch.cat(all_gt_boxes, dim=0)
        sample_ids_pred = np.array(sample_ids_pred)
        sample_ids_gt = np.array(sample_ids_gt)
        
        # Sort predictions by score (descending)
        sorted_indices = torch.argsort(all_pred_scores, descending=True)
        all_pred_boxes = all_pred_boxes[sorted_indices]
        all_pred_scores = all_pred_scores[sorted_indices]
        sample_ids_pred = sample_ids_pred[sorted_indices.cpu().numpy()]
        
        # Track which ground truths have been matched
        num_gt = len(all_gt_boxes)
        gt_matched = np.zeros(num_gt, dtype=bool)
        
        # Track true positives and false positives
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        
        for i, (pred_box, pred_sample_id) in enumerate(zip(all_pred_boxes, sample_ids_pred)):
            # Find ground truths in the same sample
            gt_in_sample = sample_ids_gt == pred_sample_id
            if not gt_in_sample.any():
                fp[i] = 1
                continue
            
            gt_boxes_in_sample = all_gt_boxes[gt_in_sample]
            gt_indices_in_sample = np.where(gt_in_sample)[0]
            
            # Compute IoU with all ground truths in this sample
            ious = compute_iou_3d_boxes(
                pred_box.unsqueeze(0),
                gt_boxes_in_sample
            ).squeeze(0)
            
            # Find best matching ground truth
            max_iou, max_idx = ious.max(dim=0)
            
            if max_iou >= iou_threshold:
                gt_idx = gt_indices_in_sample[max_idx.item()]
                if not gt_matched[gt_idx]:
                    tp[i] = 1
                    gt_matched[gt_idx] = True
                else:
                    fp[i] = 1  # Already matched
            else:
                fp[i] = 1
        
        # Compute cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        per_class_ap[f"AP_class_{class_id}"] = ap
        all_aps.append(ap)
    
    # Compute mAP
    mAP = np.mean(all_aps) if all_aps else 0.0
    
    results = {
        "mAP": mAP,
        "mAP50": mAP,  # We use single IoU threshold
        **per_class_ap
    }
    
    return results


def filter_predictions(
    boxes: torch.Tensor,
    class_logits: torch.Tensor,
    objectness: torch.Tensor,
    objectness_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter predictions based on objectness and classification scores.
    
    Returns:
        filtered_boxes: [num_filtered, 7]
        filtered_scores: [num_filtered]
        filtered_labels: [num_filtered]
    """
    # Apply sigmoid to objectness
    objectness_scores = torch.sigmoid(objectness)
    
    # Apply softmax to class logits
    class_probs = torch.softmax(class_logits, dim=-1)
    class_scores, class_labels = class_probs.max(dim=-1)
    
    # Combine objectness and classification confidence
    final_scores = objectness_scores * class_scores
    
    # Filter by thresholds
    keep_mask = (objectness_scores > objectness_threshold) & (final_scores > score_threshold)
    
    filtered_boxes = boxes[keep_mask]
    filtered_scores = final_scores[keep_mask]
    filtered_labels = class_labels[keep_mask]
    
    return filtered_boxes, filtered_scores, filtered_labels
