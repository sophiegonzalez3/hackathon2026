"""
PointPillars + CenterHead — Loss Functions (with per-class weighting)
======================================================================
- Modified Focal Loss for heatmap (from CornerNet / CenterNet)
- SmoothL1 for regression (offset, z, dim, rotation) at GT center locations only
- Per-class weighting to handle imbalanced/noisy classes (e.g., Cable)
"""

import torch
import torch.nn.functional as F


def focal_loss(pred, target, alpha=2.0, beta=4.0, class_weights=None):
    """
    Modified focal loss from CornerNet / CenterNet with per-class weighting.

    For positive locations (target == 1):
        loss = -(1 - pred)^alpha * log(pred)

    For negative locations (target < 1):
        loss = -(1 - target)^beta * pred^alpha * log(1 - pred)

    Args:
        pred:   (B, C, H, W) — predicted heatmap (after sigmoid)
        target: (B, C, H, W) — GT Gaussian heatmap
        alpha:  focal loss alpha (default 2.0)
        beta:   focal loss beta (default 4.0)
        class_weights: (C,) tensor — per-class loss multipliers, or None

    Returns:
        scalar loss (normalized by number of positive locations)
    """
    # Clamp for numerical stability
    pred = pred.clamp(min=1e-4, max=1 - 1e-4)

    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()

    # Positive loss
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask

    # Negative loss
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask

    # Apply per-class weights if provided
    if class_weights is not None:
        # class_weights shape: (C,) -> reshape to (1, C, 1, 1) for broadcasting
        weights = class_weights.view(1, -1, 1, 1).to(pred.device)
        pos_loss = pos_loss * weights
        neg_loss = neg_loss * weights

    num_pos = pos_mask.sum().clamp(min=1)
    loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

    return loss


def reg_loss(pred, target, mask, weight=1.0):
    """
    Regression loss (SmoothL1) applied only at GT center locations.

    Args:
        pred:   (B, C, H, W)
        target: (B, C, H, W)
        mask:   (B, H, W) — 1 at GT centers, 0 elsewhere
        weight: scalar multiplier

    Returns:
        scalar loss
    """
    num_pos = mask.sum().clamp(min=1)

    # Expand mask to match channel dim
    mask_expanded = mask.unsqueeze(1).expand_as(pred)  # (B, C, H, W)

    loss = F.smooth_l1_loss(pred * mask_expanded, target * mask_expanded,
                            reduction='sum')
    return weight * loss / num_pos


def reg_loss_per_class(pred, target, class_indices, class_weights, weight=1.0):
    """
    Regression loss with per-class weighting.
    
    This version weights the regression loss for each GT object based on its class.
    Useful when you want to downweight regression for noisy classes (e.g., Cable).

    Args:
        pred:          (B, C, H, W) — predicted regression values
        target:        (B, C, H, W) — GT regression values
        class_indices: (B, H, W) — class ID at each GT location (-1 for background)
        class_weights: dict {class_id: weight} or tensor (num_classes,)
        weight:        scalar multiplier for entire loss

    Returns:
        scalar loss
    """
    B, C, H, W = pred.shape
    
    # Build weight map from class indices
    if isinstance(class_weights, dict):
        weight_map = torch.ones(B, H, W, device=pred.device)
        for cls_id, cls_weight in class_weights.items():
            weight_map[class_indices == cls_id] = cls_weight
    else:
        # Assume tensor of shape (num_classes,)
        # Map class indices to weights, treating -1 as 0 weight
        valid_mask = class_indices >= 0
        weight_map = torch.zeros(B, H, W, device=pred.device)
        weight_map[valid_mask] = class_weights[class_indices[valid_mask].long()]
    
    # Expand weight map to match pred shape
    weight_map = weight_map.unsqueeze(1).expand_as(pred)  # (B, C, H, W)
    
    # Only compute loss where we have GT (weight > 0)
    mask = (weight_map > 0).float()
    num_pos = (mask[:, 0, :, :].sum()).clamp(min=1)  # Count unique positions
    
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='none')
    weighted_loss = (loss * weight_map).sum()
    
    return weight * weighted_loss / num_pos


class CenterHeadLoss(torch.nn.Module):
    """Combined loss for CenterHead training with per-class weighting."""

    def __init__(self, cfg):
        super().__init__()
        self.hm_weight = cfg.heatmap_weight
        self.off_weight = cfg.offset_weight
        self.z_weight = cfg.z_weight
        self.dim_weight = cfg.dim_weight
        self.rot_weight = cfg.rot_weight
        
        # Per-class weights for focal loss (heatmap)
        # Default: all classes weighted equally at 1.0
        self.class_weights = None
        if hasattr(cfg, 'class_loss_weights') and cfg.class_loss_weights is not None:
            # Convert dict to tensor in class order
            # Assumes cfg.class_names = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']
            weights = []
            for cls_name in cfg.class_names:
                w = cfg.class_loss_weights.get(cls_name, 1.0)
                weights.append(w)
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            print(f"[CenterHeadLoss] Using per-class weights: {dict(zip(cfg.class_names, weights))}")

    def forward(self, preds, targets):
        """
        Args:
            preds: dict from CenterHead forward
            targets: dict from collate_fn with GT tensors

        Returns:
            total_loss, loss_dict
        """
        pred_hm = preds['heatmap']
        gt_hm = targets['heatmap']
        mask = targets['reg_mask']

        # Focal loss with per-class weighting
        loss_hm = self.hm_weight * focal_loss(
            pred_hm, gt_hm, 
            class_weights=self.class_weights
        )
        
        # Regression losses (standard, no per-class weighting)
        # You could extend these with reg_loss_per_class if needed
        loss_off = reg_loss(preds['offset'], targets['offset'], mask, self.off_weight)
        loss_z = reg_loss(preds['z'], targets['z_target'], mask, self.z_weight)
        loss_dim = reg_loss(preds['dim'], targets['dim_target'], mask, self.dim_weight)
        loss_rot = reg_loss(preds['rot'], targets['rot_target'], mask, self.rot_weight)

        total = loss_hm + loss_off + loss_z + loss_dim + loss_rot

        loss_dict = {
            'total': total.item(),
            'heatmap': loss_hm.item(),
            'offset': loss_off.item(),
            'z': loss_z.item(),
            'dim': loss_dim.item(),
            'rot': loss_rot.item(),
        }
        return total, loss_dict