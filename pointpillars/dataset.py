"""
PointPillars + CenterHead — Dataset (with Augmentation)
========================================================
Loads preprocessed .npz frames (from preprocess_frames.py) and ground-truth
bboxes from the cleaned CSV.  Returns pillarized tensors + CenterHead targets.

Augmentation (train only):
  1. Random Y-flip          — flips points AND box centers/yaw
  2. Random Z-rotation      — rotates points AND box centers/yaw
  3. Random uniform scaling  — scales points AND box centers + dimensions
  4. Gaussian jitter         — per-point noise (no box change needed)
  5. Random point dropout    — simulates 25%/50%/75%/100% density
                               (matches evaluation conditions)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .utils import pillarize, generate_targets


# ═══════════════════════════════════════════════════════════════════════════════
# Online Augmentation (applied in __getitem__, transforms points + boxes)
# ═══════════════════════════════════════════════════════════════════════════════

def augment_frame(xyz, gt_boxes, cfg_aug):
    """
    Apply random augmentations to BOTH the point cloud and the GT boxes.

    Parameters
    ----------
    xyz      : (N, 3) float32 — point coordinates
    gt_boxes : list of dicts, each with keys:
               class_id, bbox_center_x/y/z, bbox_width/length/height, bbox_yaw
    cfg_aug  : dict with augmentation hyperparameters

    Returns
    -------
    xyz_aug  : (N', 3) float32 — augmented points (N' <= N if dropout applied)
    gt_boxes : list of dicts    — augmented boxes (modified in-place copies)
    refl_mask: (N,) bool or None — if dropout applied, mask to apply to reflectivity
    """
    xyz_aug = xyz.copy()
    # Deep copy boxes so we don't corrupt the cached GT
    boxes = [dict(b) for b in gt_boxes]

    flip_prob      = cfg_aug.get('flip_prob', 0.5)
    max_rot_deg    = cfg_aug.get('max_rotation_deg', 180)
    scale_range    = cfg_aug.get('scale_range', (0.95, 1.05))
    jitter_sigma   = cfg_aug.get('jitter_sigma', 0.01)
    dropout_ratios = cfg_aug.get('dropout_ratios', [1.0, 0.75, 0.50, 0.25])

    # ── 1. Random Y-flip ─────────────────────────────────────────────────
    if np.random.rand() < flip_prob:
        xyz_aug[:, 1] = -xyz_aug[:, 1]
        for b in boxes:
            b['bbox_center_y'] = -b['bbox_center_y']
            b['bbox_yaw']      = -b['bbox_yaw']

    # ── 2. Random Z-rotation ─────────────────────────────────────────────
    angle_deg = np.random.uniform(-max_rot_deg, max_rot_deg)
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1],
    ], dtype=np.float32)

    xyz_aug = xyz_aug @ rot_matrix.T

    for b in boxes:
        cx, cy = b['bbox_center_x'], b['bbox_center_y']
        b['bbox_center_x'] = cos_a * cx - sin_a * cy
        b['bbox_center_y'] = sin_a * cx + cos_a * cy
        b['bbox_yaw']      = b['bbox_yaw'] + angle_rad

    # ── 3. Random uniform scaling ────────────────────────────────────────
    scale = np.random.uniform(scale_range[0], scale_range[1])
    xyz_aug = xyz_aug * scale

    for b in boxes:
        b['bbox_center_x'] *= scale
        b['bbox_center_y'] *= scale
        b['bbox_center_z'] *= scale
        b['bbox_width']    *= scale
        b['bbox_length']   *= scale
        b['bbox_height']   *= scale

    # ── 4. Gaussian jitter (points only — boxes unchanged) ───────────────
    clip_val = jitter_sigma * 5
    jitter = np.clip(
        jitter_sigma * np.random.randn(*xyz_aug.shape),
        -clip_val, clip_val
    ).astype(np.float32)
    xyz_aug = xyz_aug + jitter

    # ── 5. Random point dropout (density robustness) ─────────────────────
    #    Simulates the 25/50/75/100% evaluation conditions.
    #    We pick a random keep-ratio each time so the model sees all densities.
    keep_ratio = np.random.choice(dropout_ratios)
    if keep_ratio < 1.0:
        n_points = len(xyz_aug)
        n_keep = max(int(n_points * keep_ratio), 1)
        keep_idx = np.random.choice(n_points, size=n_keep, replace=False)
        keep_idx.sort()
        xyz_aug = xyz_aug[keep_idx]
        return xyz_aug, boxes, keep_idx
    else:
        return xyz_aug, boxes, None


# ═══════════════════════════════════════════════════════════════════════════════
# Default augmentation config
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_AUG_CFG = {
    'flip_prob': 0.5,
    'max_rotation_deg': 180,
    'scale_range': (0.95, 1.05),
    'jitter_sigma': 0.01,
    # Include all eval densities + full density (weighted toward full)
    'dropout_ratios': [1.0, 1.0, 1.0, 0.75, 0.50, 0.25],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AirbusLidarDataset(Dataset):
    """
    One item = one frame.

    Loads the .npz point cloud (xyz + reflectivity) and builds:
      - pillar_features, num_points, pillar_coords  (model inputs)
      - heatmap, offset, z, dim, rot, reg_mask      (training targets)

    For inference (no GT), set gt_csv=None.
    """

    def __init__(self, cfg, split='train', gt_csv=None, processed_dir=None,
                 augment=None, aug_cfg=None):
        """
        Args:
            cfg          : Config object
            split        : 'train' or 'val' — determines which scenes to use
            gt_csv       : path to GT CSV (None for inference)
            processed_dir: override cfg.processed_dir
            augment      : bool or None — if None, auto-set to True for train
            aug_cfg      : dict — augmentation hyperparameters (see DEFAULT_AUG_CFG)
        """
        self.cfg = cfg
        self.split = split
        self.has_gt = gt_csv is not None

        # Auto-enable augmentation for training only
        if augment is None:
            self.augment = (split == 'train')
        else:
            self.augment = augment

        self.aug_cfg = aug_cfg or DEFAULT_AUG_CFG

        proc_dir = Path(processed_dir or cfg.processed_dir)
        self.processed_dir = proc_dir
        manifest_path = proc_dir / "manifest.csv"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run preprocess_frames.py first."
            )

        manifest = pd.read_csv(manifest_path)

        # ── Split by scene ────────────────────────────────────────────────
        if split == 'train':
            self.manifest = manifest[~manifest['scene'].isin(cfg.val_scenes)].reset_index(drop=True)
        elif split == 'val':
            self.manifest = manifest[manifest['scene'].isin(cfg.val_scenes)].reset_index(drop=True)
        else:
            # 'all' or 'test' — use everything
            self.manifest = manifest.reset_index(drop=True)

        # ── Verify files exist (avoid mid-training crashes) ──────────────
        valid_mask = []
        n_missing = 0
        for _, row in self.manifest.iterrows():
            npz_path = proc_dir / row['scene'] / f"frame_{int(row['pose_index']):03d}.npz"
            exists = npz_path.exists()
            valid_mask.append(exists)
            if not exists:
                n_missing += 1
        self.manifest = self.manifest[valid_mask].reset_index(drop=True)
        if n_missing > 0:
            print(f"  ⚠ {n_missing} missing .npz files filtered out from {split} set")

        # ── Load GT boxes ─────────────────────────────────────────────────
        self.gt_by_frame = {}  # (scene, pose_index) -> list of box dicts
        if self.has_gt:
            csv_path = gt_csv or cfg.gt_csv
            gt_df = pd.read_csv(csv_path)

            # Map class labels to IDs
            label_to_id = {name: i for i, name in enumerate(cfg.class_names)}

            for (scene, pose), group in gt_df.groupby(['scene', 'pose_index']):
                boxes = []
                for _, row in group.iterrows():
                    cid = label_to_id.get(row['class_label'], -1)
                    if cid < 0:
                        continue
                    boxes.append({
                        'class_id': cid,
                        'bbox_center_x': row['bbox_center_x'],
                        'bbox_center_y': row['bbox_center_y'],
                        'bbox_center_z': row['bbox_center_z'],
                        'bbox_width': row['bbox_width'],
                        'bbox_length': row['bbox_length'],
                        'bbox_height': row['bbox_height'],
                        'bbox_yaw': row['bbox_yaw'],
                    })
                self.gt_by_frame[(scene, pose)] = boxes

        aug_status = "ON" if self.augment else "OFF"
        print(f"Dataset [{split}]: {len(self.manifest)} frames, "
              f"{len(self.gt_by_frame)} frames with GT, augment={aug_status}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        scene = row['scene']
        pose_index = int(row['pose_index'])

        # Build path from processed_dir (manifest paths may be stale/relative)
        npz_path = Path(self.processed_dir) / scene / f"frame_{pose_index:03d}.npz"

        # ── Load point cloud ──────────────────────────────────────────────
        data = np.load(npz_path)
        xyz = data['xyz'].astype(np.float32)              # (N, 3)
        reflectivity = data['reflectivity'].astype(np.float32)  # (N,)
        ego_pose = data['ego_pose']                        # (4,) for inference CSV

        # Normalize reflectivity to [0, 1]
        reflectivity = reflectivity / 255.0

        # ── Load GT boxes for this frame ──────────────────────────────────
        gt_boxes = self.gt_by_frame.get((scene, pose_index), [])

        # ══════════════════════════════════════════════════════════════════
        # AUGMENTATION (train only) — transforms points AND boxes together
        # ══════════════════════════════════════════════════════════════════
        if self.augment and self.has_gt:
            xyz, gt_boxes, keep_idx = augment_frame(xyz, gt_boxes, self.aug_cfg)

            # If point dropout was applied, subsample reflectivity too
            if keep_idx is not None:
                reflectivity = reflectivity[keep_idx]

        # ── Combine: (N, 4) = [x, y, z, reflectivity] ────────────────────
        points = np.column_stack([xyz, reflectivity])

        # ── Pillarize ─────────────────────────────────────────────────────
        pillar_features, num_points, pillar_coords = pillarize(points, self.cfg)

        # ── Build targets ─────────────────────────────────────────────────
        if self.has_gt:
            heatmap, offset, z_tgt, dim_tgt, rot_tgt, reg_mask, indices, num_objs = \
                generate_targets(gt_boxes, self.cfg)
        else:
            hm_h, hm_w = self.cfg.heatmap_h, self.cfg.heatmap_w
            heatmap = np.zeros((self.cfg.num_classes, hm_h, hm_w), dtype=np.float32)
            offset = np.zeros((2, hm_h, hm_w), dtype=np.float32)
            z_tgt = np.zeros((1, hm_h, hm_w), dtype=np.float32)
            dim_tgt = np.zeros((3, hm_h, hm_w), dtype=np.float32)
            rot_tgt = np.zeros((2, hm_h, hm_w), dtype=np.float32)
            reg_mask = np.zeros((hm_h, hm_w), dtype=np.float32)
            indices = np.zeros((64, 2), dtype=np.int32)
            num_objs = 0

        return {
            'pillar_features': pillar_features,              # (P, max_pts, 9)
            'num_points': num_points,                        # (P,)
            'pillar_coords': pillar_coords,                  # (P, 2)
            'heatmap': heatmap,                              # (C, H, W)
            'offset': offset,                                # (2, H, W)
            'z_target': z_tgt,                               # (1, H, W)
            'dim_target': dim_tgt,                           # (3, H, W)
            'rot_target': rot_tgt,                           # (2, H, W)
            'reg_mask': reg_mask,                            # (H, W)
            'num_objs': num_objs,
            # Metadata (for inference CSV output)
            'scene': scene,
            'pose_index': pose_index,
            'ego_pose': ego_pose.astype(np.float64),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Collate & DataLoader builder
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """
    Custom collate that stacks pillars across the batch and prepends
    a batch index to pillar_coords.
    """
    # Stack simple tensors
    heatmaps = torch.from_numpy(np.stack([b['heatmap'] for b in batch]))
    offsets = torch.from_numpy(np.stack([b['offset'] for b in batch]))
    z_targets = torch.from_numpy(np.stack([b['z_target'] for b in batch]))
    dim_targets = torch.from_numpy(np.stack([b['dim_target'] for b in batch]))
    rot_targets = torch.from_numpy(np.stack([b['rot_target'] for b in batch]))
    reg_masks = torch.from_numpy(np.stack([b['reg_mask'] for b in batch]))

    # Concatenate pillars with batch index
    all_pillar_feats = []
    all_num_points = []
    all_pillar_coords = []

    for b_idx, b in enumerate(batch):
        P = len(b['num_points'])
        feats = b['pillar_features']    # (P, max_pts, 9)
        npts = b['num_points']          # (P,)
        coords = b['pillar_coords']     # (P, 2)

        # Prepend batch index
        batch_col = np.full((P, 1), b_idx, dtype=np.int32)
        coords_with_batch = np.concatenate([batch_col, coords], axis=1)  # (P, 3)

        all_pillar_feats.append(feats)
        all_num_points.append(npts)
        all_pillar_coords.append(coords_with_batch)

    pillar_features = torch.from_numpy(np.concatenate(all_pillar_feats, axis=0))
    num_points = torch.from_numpy(np.concatenate(all_num_points, axis=0))
    pillar_coords = torch.from_numpy(np.concatenate(all_pillar_coords, axis=0))

    # Metadata
    scenes = [b['scene'] for b in batch]
    pose_indices = [b['pose_index'] for b in batch]
    ego_poses = [b['ego_pose'] for b in batch]
    num_objs = [b['num_objs'] for b in batch]

    return {
        'pillar_features': pillar_features,      # (P_total, max_pts, 9)
        'num_points': num_points,                # (P_total,)
        'pillar_coords': pillar_coords,          # (P_total, 3)
        'heatmap': heatmaps,                     # (B, C, hm_H, hm_W)
        'offset': offsets,                       # (B, 2, hm_H, hm_W)
        'z_target': z_targets,                   # (B, 1, hm_H, hm_W)
        'dim_target': dim_targets,               # (B, 3, hm_H, hm_W)
        'rot_target': rot_targets,               # (B, 2, hm_H, hm_W)
        'reg_mask': reg_masks,                   # (B, hm_H, hm_W)
        'num_objs': num_objs,
        'scenes': scenes,
        'pose_indices': pose_indices,
        'ego_poses': ego_poses,
    }


def build_dataloaders(cfg, aug_cfg=None):
    """Create train and val DataLoaders."""
    # Use cfg.train_augment if it exists, default to True for backward compatibility
    do_augment = getattr(cfg, 'train_augment', True)
    
    train_ds = AirbusLidarDataset(
        cfg, split='train', gt_csv=cfg.gt_csv,
        augment=do_augment, aug_cfg=aug_cfg,  # ← now configurable
    )
    val_ds = AirbusLidarDataset(
        cfg, split='val', gt_csv=cfg.gt_csv,
        augment=False,  # NEVER augment validation
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
