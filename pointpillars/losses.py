"""
PointPillars + CenterHead — Loss Functions
============================================
- Modified Focal Loss for heatmap (from CornerNet / CenterNet)
- SmoothL1 for regression (offset, z, dim, rotation) at GT center locations only
"""

import torch
import torch.nn.functional as F


def focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    Modified focal loss from CornerNet / CenterNet.

    For positive locations (target == 1):
        loss = -(1 - pred)^alpha * log(pred)

    For negative locations (target < 1):
        loss = -(1 - target)^beta * pred^alpha * log(1 - pred)

    Args:
        pred:   (B, C, H, W) — predicted heatmap (after sigmoid)
        target: (B, C, H, W) — GT Gaussian heatmap

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


class CenterHeadLoss(torch.nn.Module):
    """Combined loss for CenterHead training."""

    def __init__(self, cfg):
        super().__init__()
        self.hm_weight = cfg.heatmap_weight
        self.off_weight = cfg.offset_weight
        self.z_weight = cfg.z_weight
        self.dim_weight = cfg.dim_weight
        self.rot_weight = cfg.rot_weight

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

        loss_hm = self.hm_weight * focal_loss(pred_hm, gt_hm)
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
