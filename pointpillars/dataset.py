"""
PointPillars + CenterHead — Dataset
=====================================
Loads preprocessed .npz frames (from preprocess_frames.py) and ground-truth
bboxes from the cleaned CSV.  Returns pillarized tensors + CenterHead targets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .utils import pillarize, generate_targets


class AirbusLidarDataset(Dataset):
    """
    One item = one frame.

    Loads the .npz point cloud (xyz + reflectivity) and builds:
      - pillar_features, num_points, pillar_coords  (model inputs)
      - heatmap, offset, z, dim, rot, reg_mask      (training targets)

    For inference (no GT), set gt_csv=None.
    """

    def __init__(self, cfg, split='train', gt_csv=None, processed_dir=None):
        """
        Args:
            cfg: Config object
            split: 'train' or 'val' — determines which scenes to use
            gt_csv: path to GT CSV (None for inference)
            processed_dir: override cfg.processed_dir
        """
        self.cfg = cfg
        self.split = split
        self.has_gt = gt_csv is not None

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

        print(f"Dataset [{split}]: {len(self.manifest)} frames, "
              f"{len(self.gt_by_frame)} frames with GT")

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

        # Combine: (N, 4) = [x, y, z, reflectivity]
        points = np.column_stack([xyz, reflectivity])

        # ── Pillarize ─────────────────────────────────────────────────────
        pillar_features, num_points, pillar_coords = pillarize(points, self.cfg)

        # ── Build targets ─────────────────────────────────────────────────
        if self.has_gt:
            gt_boxes = self.gt_by_frame.get((scene, pose_index), [])
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


def build_dataloaders(cfg):
    """Create train and val DataLoaders."""
    train_ds = AirbusLidarDataset(cfg, split='train', gt_csv=cfg.gt_csv)
    val_ds = AirbusLidarDataset(cfg, split='val', gt_csv=cfg.gt_csv)

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