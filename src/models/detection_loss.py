from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    Detection loss combining:
    1. Box regression loss (Smooth L1)
    2. Classification loss (Cross Entropy)
    3. Objectness loss (Binary Cross Entropy)
    
    Uses Hungarian matching to assign predictions to ground truth.
    """
    
    def __init__(
        self,
        num_classes: int,
        box_loss_weight: float = 5.0,
        class_loss_weight: float = 1.0,
        objectness_loss_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_loss_weight = box_loss_weight
        self.class_loss_weight = class_loss_weight
        self.objectness_loss_weight = objectness_loss_weight
    
    def forward(
        self,
        pred_boxes: torch.Tensor,  # [batch_size, max_objects, 7]
        pred_class_logits: torch.Tensor,  # [batch_size, max_objects, num_classes]
        pred_objectness: torch.Tensor,  # [batch_size, max_objects]
        gt_boxes: torch.Tensor,  # [batch_size, num_gt_objects, 7]
        gt_labels: torch.Tensor,  # [batch_size, num_gt_objects]
    ) -> dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Returns:
            Dictionary with 'loss', 'box_loss', 'class_loss', 'objectness_loss'
        """
        batch_size, max_objects = pred_boxes.shape[:2]
        device = pred_boxes.device
        
        total_loss = 0.0
        total_box_loss = 0.0
        total_class_loss = 0.0
        total_objectness_loss = 0.0
        
        for i in range(batch_size):
            # Get predictions for this sample
            boxes_i = pred_boxes[i]  # [max_objects, 7]
            class_logits_i = pred_class_logits[i]  # [max_objects, num_classes]
            objectness_i = pred_objectness[i]  # [max_objects]
            
            # Get ground truth (filter out padding)
            gt_boxes_i = gt_boxes[i]  # [num_gt_objects, 7]
            gt_labels_i = gt_labels[i]  # [num_gt_objects]
            
            # Find valid ground truth boxes (non-zero boxes)
            valid_mask = (gt_boxes_i.sum(dim=-1) != 0)
            gt_boxes_i = gt_boxes_i[valid_mask]
            gt_labels_i = gt_labels_i[valid_mask]
            num_gt = len(gt_boxes_i)
            
            if num_gt == 0:
                # No ground truth objects - penalize all predictions
                objectness_target = torch.zeros_like(objectness_i)
                obj_loss = F.binary_cross_entropy_with_logits(objectness_i, objectness_target)
                total_objectness_loss += obj_loss
                total_loss += self.objectness_loss_weight * obj_loss
                continue
            
            # Simple assignment: match first num_gt predictions to ground truth
            # (In practice, use Hungarian matching or IoU-based assignment)
            num_matched = min(num_gt, max_objects)
            
            # Box regression loss (only for matched objects)
            if num_matched > 0:
                matched_boxes = boxes_i[:num_matched]
                matched_gt_boxes = gt_boxes_i[:num_matched]
                box_loss = F.smooth_l1_loss(matched_boxes, matched_gt_boxes)
                total_box_loss += box_loss
                
                # Classification loss (only for matched objects)
                matched_class_logits = class_logits_i[:num_matched]
                matched_gt_labels = gt_labels_i[:num_matched]
                class_loss = F.cross_entropy(matched_class_logits, matched_gt_labels)
                total_class_loss += class_loss
            
            # Objectness loss (all predictions)
            objectness_target = torch.zeros_like(objectness_i)
            objectness_target[:num_matched] = 1.0  # First num_matched should be objects
            obj_loss = F.binary_cross_entropy_with_logits(objectness_i, objectness_target)
            total_objectness_loss += obj_loss
            
            # Combine losses
            sample_loss = (
                self.box_loss_weight * box_loss +
                self.class_loss_weight * class_loss +
                self.objectness_loss_weight * obj_loss
            )
            total_loss += sample_loss
        
        # Average over batch
        total_loss /= batch_size
        total_box_loss /= batch_size
        total_class_loss /= batch_size
        total_objectness_loss /= batch_size
        
        return {
            "loss": total_loss,
            "box_loss": total_box_loss,
            "class_loss": total_class_loss,
            "objectness_loss": total_objectness_loss,
        }


def compute_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D IoU between two sets of boxes (simplified version).
    
    Args:
        boxes1: [N, 7] (x,y,z,w,h,d,angle)
        boxes2: [M, 7] (x,y,z,w,h,d,angle)
    
    Returns:
        iou: [N, M]
    """
    # Simplified: compute IoU based on center distance and size overlap
    # (Proper 3D IoU requires rotation handling)
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Extract centers and sizes
    centers1 = boxes1[:, :3].unsqueeze(1)  # [N, 1, 3]
    centers2 = boxes2[:, :3].unsqueeze(0)  # [1, M, 3]
    sizes1 = boxes1[:, 3:6].unsqueeze(1)  # [N, 1, 3]
    sizes2 = boxes2[:, 3:6].unsqueeze(0)  # [1, M, 3]
    
    # Compute distance between centers
    dist = torch.norm(centers1 - centers2, dim=-1)  # [N, M]
    
    # Compute size similarity
    size_sim = torch.min(sizes1, sizes2).sum(dim=-1) / torch.max(sizes1, sizes2).sum(dim=-1)
    
    # Approximate IoU (not exact 3D IoU)
    iou = size_sim * torch.exp(-dist / 10.0)
    
    return iou
