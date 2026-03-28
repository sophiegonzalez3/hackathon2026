"""
PointPillars + CenterHead — Configuration
==========================================
All tuneable parameters in one place.
Adjust POINT_CLOUD_RANGE and PILLAR_SIZE based on GPU memory.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────
    processed_dir: str = "processed/"
    gt_csv: str = "gt_runs/gt_bboxes_run_05_merge.csv"

    # ── Train / Val split (scene-level) ───────────────────────────────────
    # Hold out 2 scenes: one "similar" (scene_9), one "different" (scene_10)
    # to mimic the eval setup (known + unknown scenes)
    val_scenes: List[str] = field(default_factory=lambda: ["scene_9", "scene_10"])

    # ── Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max] ─────
    # Covers 99%+ of objects.  Objects outside are clipped.
    point_cloud_range: List[float] = field(
        default_factory=lambda: [-20.0, -224.0, -100.0, 312.8, 224.0, 120.0]
    )

    # ── Pillar / voxel settings ───────────────────────────────────────────
    pillar_x: float = 0.8       # meters
    pillar_y: float = 0.8       # meters
    max_points_per_pillar: int = 32
    max_pillars: int = 40_000   # budget per frame

    # ── Derived grid size (computed in __post_init__) ─────────────────────
    grid_x: int = 0
    grid_y: int = 0

    # ── Model ─────────────────────────────────────────────────────────────
    num_point_features: int = 4         # x, y, z, reflectivity
    pillar_feat_channels: int = 64      # PillarVFE output dim

    # Backbone block configs: (out_channels, num_layers, stride)
    backbone_blocks: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (64, 3, 2),     # Block 1: stride 2 from input
            (128, 5, 2),    # Block 2: stride 4 from input
            (256, 5, 2),    # Block 3: stride 8 from input
        ]
    )

    # Neck: upsample each block to stride=2 and concatenate
    # (upsample_stride, out_channels) per block
    neck_config: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 128),       # Block 1 already at stride 2 → no upsample
            (2, 128),       # Block 2 at stride 4 → upsample 2x
            (4, 128),       # Block 3 at stride 8 → upsample 4x
        ]
    )

    # CenterHead
    head_channels: int = 128            # intermediate conv before outputs
    head_stride: int = 2                # heatmap resolution = grid / head_stride

    # ── Classes ───────────────────────────────────────────────────────────
    class_names: List[str] = field(
        default_factory=lambda: ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]
    )
    num_classes: int = 4

    # Gaussian radius min (in heatmap pixels) — per class
    # Larger for big objects, smaller for compact ones
    gaussian_min_radius: Dict[str, int] = field(
        default_factory=lambda: {
            "Antenna": 2,
            "Cable": 3,       # cables are long, larger radius helps
            "Electric Pole": 2,
            "Wind Turbine": 3,
        }
    )

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 2
    num_epochs: int = 80
    lr: float = 2.25e-4           # AdamW
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_clip: float = 10.0

    # Loss weights
    heatmap_weight: float = 1.0
    offset_weight: float = 1.0
    z_weight: float = 1.0
    dim_weight: float = 1.0
    rot_weight: float = 1.0

    # ── Inference ─────────────────────────────────────────────────────────
    score_threshold: float = 0.3
    nms_kernel: int = 3             # max-pool NMS kernel size
    max_detections: int = 50        # per frame

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda"
    num_workers: int = 4

    def __post_init__(self):
        pcr = self.point_cloud_range
        self.grid_x = int((pcr[3] - pcr[0]) / self.pillar_x)   # 416
        self.grid_y = int((pcr[4] - pcr[1]) / self.pillar_y)   # 560

        # Sanity: grid must be divisible by the total backbone stride
        total_stride = 1
        for _, _, s in self.backbone_blocks:
            total_stride *= s
        assert self.grid_x % total_stride == 0, \
            f"grid_x={self.grid_x} not divisible by backbone stride {total_stride}"
        assert self.grid_y % total_stride == 0, \
            f"grid_y={self.grid_y} not divisible by backbone stride {total_stride}"

    @property
    def heatmap_h(self):
        return self.grid_y // self.head_stride  # H = Y axis

    @property
    def heatmap_w(self):
        return self.grid_x // self.head_stride  # W = X axis

    def print_summary(self):
        print(f"{'='*60}")
        print(f"  PointPillars + CenterHead Config")
        print(f"{'='*60}")
        pcr = self.point_cloud_range
        print(f"  Range X: [{pcr[0]}, {pcr[3]}] m  ({pcr[3]-pcr[0]:.0f}m)")
        print(f"  Range Y: [{pcr[1]}, {pcr[4]}] m  ({pcr[4]-pcr[1]:.0f}m)")
        print(f"  Range Z: [{pcr[2]}, {pcr[5]}] m  ({pcr[5]-pcr[2]:.0f}m)")
        print(f"  Pillar size: {self.pillar_x} × {self.pillar_y} m")
        print(f"  Grid: {self.grid_x} × {self.grid_y} = {self.grid_x*self.grid_y:,} cells")
        print(f"  Heatmap: {self.heatmap_h} × {self.heatmap_w} (stride {self.head_stride})")
        print(f"  Max pillars/frame: {self.max_pillars:,}")
        print(f"  Max pts/pillar: {self.max_points_per_pillar}")
        print(f"  Backbone: {self.backbone_blocks}")
        print(f"  Classes: {self.class_names}")
        print(f"  Val scenes: {self.val_scenes}")
        print(f"  Batch: {self.batch_size}  Epochs: {self.num_epochs}  LR: {self.lr}")
        print(f"{'='*60}")
