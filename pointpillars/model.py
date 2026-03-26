"""
PointPillars + CenterHead — Model
==================================
Pure PyTorch, no spconv dependency.

Architecture:
  PillarVFE  →  PointPillarScatter  →  Backbone2D  →  BEVNeck  →  CenterHead
     ↑                                                              ↓
  (N,9)→(P,C)    (P,C)→(C,H,W)       multi-scale              heatmaps + reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Pillar Feature Encoder  (PointNet-style per-pillar)
# ═══════════════════════════════════════════════════════════════════════════════

class PillarVFE(nn.Module):
    """
    Encodes raw point features within each pillar into a fixed-size vector.

    Input per point (9 features):
      [x, y, z, r, x_c, y_c, z_c, x_p, y_p]
      - (x,y,z,r): raw coordinates + reflectivity
      - (x_c, y_c, z_c): offset from pillar mean
      - (x_p, y_p): offset from pillar grid center

    Output: (num_pillars, out_channels) after max-pool over points in each pillar.
    """

    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, pillar_features, num_points_per_pillar):
        """
        Args:
            pillar_features: (P, max_pts, 9) — zero-padded point features
            num_points_per_pillar: (P,) — actual point count per pillar

        Returns:
            (P, C) — one feature vector per pillar
        """
        P, max_pts, D = pillar_features.shape

        # Flatten → apply linear → reshape
        x = pillar_features.reshape(P * max_pts, D)       # (P*max_pts, 9)
        x = self.net(x)                                    # (P*max_pts, C)
        x = x.reshape(P, max_pts, self.out_channels)       # (P, max_pts, C)

        # Mask padded points before max-pool (non-inplace for autograd safety)
        mask = torch.arange(max_pts, device=x.device).unsqueeze(0) < num_points_per_pillar.unsqueeze(1)
        x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        # Max pool over points in each pillar
        x = x.max(dim=1)[0]                                # (P, C)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Scatter to BEV pseudo-image
# ═══════════════════════════════════════════════════════════════════════════════

class PointPillarScatter(nn.Module):
    """Scatter pillar features onto a 2D BEV grid."""

    def __init__(self, in_channels, grid_x, grid_y):
        super().__init__()
        self.in_channels = in_channels
        self.grid_x = grid_x  # H
        self.grid_y = grid_y  # W

    def forward(self, pillar_features, pillar_coords, batch_size):
        """
        Args:
            pillar_features: (P_total, C) — all pillars in the batch
            pillar_coords: (P_total, 3) — [batch_idx, grid_x_idx, grid_y_idx]
            batch_size: int

        Returns:
            (B, C, H, W) — BEV pseudo-image
        """
        C = self.in_channels
        device = pillar_features.device

        bev = torch.zeros(batch_size, C, self.grid_x, self.grid_y,
                          dtype=pillar_features.dtype, device=device)

        batch_idx = pillar_coords[:, 0].long()
        x_idx = pillar_coords[:, 1].long()
        y_idx = pillar_coords[:, 2].long()

        bev[batch_idx, :, x_idx, y_idx] = pillar_features

        return bev


# ═══════════════════════════════════════════════════════════════════════════════
# 2D Backbone (multi-scale feature extraction on BEV)
# ═══════════════════════════════════════════════════════════════════════════════

class BackboneBlock(nn.Module):
    """A single block: first layer strides, rest are same-resolution."""

    def __init__(self, in_channels, out_channels, num_layers, stride):
        super().__init__()
        layers = []
        # First layer with stride
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        # Remaining layers at same resolution
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Backbone2D(nn.Module):
    """Multi-scale 2D backbone producing features at different strides."""

    def __init__(self, in_channels, block_configs):
        """
        Args:
            in_channels: input feature channels (from PillarVFE)
            block_configs: list of (out_channels, num_layers, stride)
        """
        super().__init__()
        self.blocks = nn.ModuleList()

        c_in = in_channels
        for out_c, n_layers, stride in block_configs:
            self.blocks.append(BackboneBlock(c_in, out_c, n_layers, stride))
            c_in = out_c

    def forward(self, x):
        """Returns list of feature maps at each scale."""
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs


# ═══════════════════════════════════════════════════════════════════════════════
# BEV Neck (FPN-like: upsample + concat)
# ═══════════════════════════════════════════════════════════════════════════════

class BEVNeck(nn.Module):
    """Upsample multi-scale features to a common resolution and concatenate."""

    def __init__(self, in_channels_list, neck_config):
        """
        Args:
            in_channels_list: [64, 128, 256] — channels from each backbone block
            neck_config: [(upsample_stride, out_channels), ...]
        """
        super().__init__()
        self.deblocks = nn.ModuleList()
        total_out = 0

        for (up_stride, out_c), in_c in zip(neck_config, in_channels_list):
            if up_stride > 1:
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_c, out_c, up_stride, stride=up_stride, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            self.deblocks.append(block)
            total_out += out_c

        self.out_channels = total_out

    def forward(self, multi_scale_features):
        ups = []
        for feat, deblock in zip(multi_scale_features, self.deblocks):
            ups.append(deblock(feat))

        # Handle potential size mismatches from rounding
        target_h = ups[0].shape[2]
        target_w = ups[0].shape[3]
        aligned = []
        for u in ups:
            if u.shape[2] != target_h or u.shape[3] != target_w:
                u = F.interpolate(u, size=(target_h, target_w), mode='bilinear',
                                  align_corners=False)
            aligned.append(u)

        return torch.cat(aligned, dim=1)  # (B, total_out, H/2, W/2)


# ═══════════════════════════════════════════════════════════════════════════════
# CenterHead (heatmap + regression)
# ═══════════════════════════════════════════════════════════════════════════════

class SeparateHead(nn.Module):
    """A single regression sub-head: Conv → BN → ReLU → Conv."""

    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, x):
        return self.head(x)


class CenterHead(nn.Module):
    """
    CenterNet-style detection head.

    Outputs:
      - heatmap:  (B, num_classes, H, W) — class-specific Gaussian peaks
      - offset:   (B, 2, H, W)          — sub-pixel xy offset
      - height_z: (B, 1, H, W)          — z center
      - dim:      (B, 3, H, W)          — log(w), log(l), log(h)
      - rot:      (B, 2, H, W)          — sin(yaw), cos(yaw)
    """

    def __init__(self, in_channels, num_classes, hidden=128):
        super().__init__()
        self.num_classes = num_classes

        # Shared feature refinement
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # Per-task heads
        self.heatmap_head = SeparateHead(hidden, hidden, num_classes)
        self.offset_head = SeparateHead(hidden, hidden, 2)    # xy offset
        self.z_head = SeparateHead(hidden, hidden, 1)          # z center
        self.dim_head = SeparateHead(hidden, hidden, 3)        # log dims
        self.rot_head = SeparateHead(hidden, hidden, 2)        # sin/cos yaw

        # Initialize heatmap bias to -2.19 (≈sigmoid → 0.1)
        # This prevents the heatmap from being all-zero at start, improving
        # gradient flow for the focal loss.
        nn.init.constant_(self.heatmap_head.head[-1].bias, -2.19)

    def forward(self, x):
        x = self.shared_conv(x)
        return {
            'heatmap': torch.sigmoid(self.heatmap_head(x)),
            'offset': self.offset_head(x),
            'z': self.z_head(x),
            'dim': self.dim_head(x),
            'rot': self.rot_head(x),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class PointPillarsCenterHead(nn.Module):
    """Complete PointPillars + CenterHead detector."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1. Pillar Feature Encoder
        #    Input features per point: x, y, z, r, x_c, y_c, z_c, x_p, y_p = 9
        self.vfe = PillarVFE(in_channels=9, out_channels=cfg.pillar_feat_channels)

        # 2. Scatter to BEV
        self.scatter = PointPillarScatter(
            in_channels=cfg.pillar_feat_channels,
            grid_x=cfg.grid_x,
            grid_y=cfg.grid_y,
        )

        # 3. 2D Backbone
        self.backbone = Backbone2D(
            in_channels=cfg.pillar_feat_channels,
            block_configs=cfg.backbone_blocks,
        )

        # 4. Neck
        backbone_out_channels = [c for c, _, _ in cfg.backbone_blocks]
        self.neck = BEVNeck(backbone_out_channels, cfg.neck_config)

        # 5. CenterHead
        self.head = CenterHead(
            in_channels=self.neck.out_channels,
            num_classes=cfg.num_classes,
            hidden=cfg.head_channels,
        )

    def forward(self, pillar_features, num_points_per_pillar, pillar_coords, batch_size):
        """
        Args:
            pillar_features: (P, max_pts, 9)
            num_points_per_pillar: (P,)
            pillar_coords: (P, 3)  — [batch_idx, x_idx, y_idx]
            batch_size: int

        Returns:
            dict with 'heatmap', 'offset', 'z', 'dim', 'rot'
        """
        # Encode pillars
        pillar_feat = self.vfe(pillar_features, num_points_per_pillar)  # (P, C)

        # Scatter to BEV
        bev = self.scatter(pillar_feat, pillar_coords, batch_size)     # (B, C, H, W)

        # Backbone + Neck
        multi_scale = self.backbone(bev)
        bev_feat = self.neck(multi_scale)                              # (B, 384, H/2, W/2)

        # Detection head
        preds = self.head(bev_feat)

        return preds

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable