"""
═══════════════════════════════════════════════════════════════════════════════
Visualize BEV Point Cloud + Heatmap Targets
═══════════════════════════════════════════════════════════════════════════════
This script helps you visually verify:
  1. The BEV projection of your point cloud
  2. The heatmap Gaussian targets for each class
  3. That bounding boxes align with heatmap peaks

Usage:
    python visualize_dataset.py
    python visualize_dataset.py --scene scene_1 --pose 5
    python visualize_dataset.py --idx 42
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import argparse

from pointpillars.config import Config
from pointpillars.dataset import AirbusLidarDataset


# Class colors matching your project
CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']
CLASS_COLORS = ['#5B4FFF', '#FFD060', '#FF6B8A', '#40FF40']  # Bright colors


def load_raw_frame(scene, pose_index, processed_dir):
    """Load raw .npz frame data."""
    from pathlib import Path
    path = Path(processed_dir) / scene / f"frame_{pose_index:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frame not found: {path}")
    data = np.load(path)
    return data['xyz'], data['reflectivity']


def bbox_corners_2d(box):
    """Get 2D BEV corners of a bounding box."""
    cx = box['bbox_center_x']
    cy = box['bbox_center_y']
    w = box['bbox_width']
    l = box['bbox_length']
    yaw = box['bbox_yaw']
    
    # Half extents
    dx, dy = w / 2, l / 2
    corners = np.array([
        [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy], [-dx, -dy]
    ])
    
    # Rotate
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = (R @ corners.T).T + np.array([cx, cy])
    
    return corners


def visualize_sample(sample, cfg, raw_xyz=None, gt_boxes=None, save_path=None):
    """
    Visualize a single sample from the dataset.
    
    Creates a figure with:
      - Top row: BEV point cloud with bboxes, pillar occupancy
      - Bottom row: Heatmaps for each class
    """
    
    # Extract data from sample
    pillar_features = sample['pillar_features']  # (P, max_pts, 9)
    pillar_coords = sample['pillar_coords']      # (P, 2) = [x_idx, y_idx]
    heatmap = sample['heatmap']                  # (C, H, W)
    reg_mask = sample['reg_mask']                # (H, W)
    scene = sample['scene']
    pose_index = sample['pose_index']
    
    num_classes = heatmap.shape[0]
    hm_h, hm_w = heatmap.shape[1], heatmap.shape[2]
    
    # Config values
    pcr = cfg.point_cloud_range
    x_min, y_min = pcr[0], pcr[1]
    x_max, y_max = pcr[3], pcr[4]
    
    # Create figure
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Dataset Sample: {scene} / pose {pose_index}\n'
                 f'Heatmap shape: (H={hm_h}, W={hm_w}) | '
                 f'Grid: ({cfg.grid_y}, {cfg.grid_x}) | '
                 f'Pillars: {len(pillar_coords)}',
                 fontsize=14, fontweight='bold')
    
    # ═══════════════════════════════════════════════════════════════════════
    # Plot 1: BEV Point Cloud (top-left)
    # ═══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(2, 3, 1)
    
    if raw_xyz is not None:
        # Plot actual points
        # Subsample for performance
        max_pts = 50000
        if len(raw_xyz) > max_pts:
            idx = np.random.choice(len(raw_xyz), max_pts, replace=False)
            xyz_plot = raw_xyz[idx]
        else:
            xyz_plot = raw_xyz
        
        ax1.scatter(xyz_plot[:, 0], xyz_plot[:, 1], s=0.5, c='gray', alpha=0.3)
    
    # Draw bounding boxes
    if gt_boxes is not None:
        drawn_classes = set()
        for box in gt_boxes:
            corners = bbox_corners_2d(box)
            cid = box['class_id']
            color = CLASS_COLORS[cid]
            label = CLASS_NAMES[cid] if cid not in drawn_classes else ''
            drawn_classes.add(cid)
            ax1.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2, label=label)
            ax1.plot(box['bbox_center_x'], box['bbox_center_y'], 'x', 
                    color=color, markersize=8, markeredgewidth=2)
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('BEV Point Cloud + GT Boxes')
    ax1.set_aspect('equal')
    if gt_boxes:
        ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Plot 2: Pillar Occupancy Grid (top-middle)
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Create occupancy grid
    occupancy = np.zeros((cfg.grid_y, cfg.grid_x), dtype=np.float32)
    x_idx = pillar_coords[:, 0]  # X grid indices
    y_idx = pillar_coords[:, 1]  # Y grid indices
    
    # Count points per pillar from pillar_features
    pts_per_pillar = (pillar_features[:, :, 0] != 0).sum(axis=1)  # non-zero x coords
    
    # Scatter into grid (y_idx = row, x_idx = col)
    for i in range(len(pillar_coords)):
        if y_idx[i] < cfg.grid_y and x_idx[i] < cfg.grid_x:
            occupancy[y_idx[i], x_idx[i]] = pts_per_pillar[i]
    
    im2 = ax2.imshow(occupancy, origin='lower', cmap='viridis',
                     extent=[x_min, x_max, y_min, y_max], aspect='auto')
    plt.colorbar(im2, ax=ax2, label='Points per pillar')
    
    # Overlay bbox centers
    if gt_boxes is not None:
        for box in gt_boxes:
            cid = box['class_id']
            ax2.plot(box['bbox_center_x'], box['bbox_center_y'], 'x',
                    color=CLASS_COLORS[cid], markersize=10, markeredgewidth=2)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Pillar Occupancy ({len(pillar_coords)} pillars)')
    
    # ═══════════════════════════════════════════════════════════════════════
    # Plot 3: Combined Heatmap (top-right)
    # ═══════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Max across classes for visualization
    combined_hm = heatmap.max(axis=0)  # (H, W)
    
    im3 = ax3.imshow(combined_hm, origin='lower', cmap='hot',
                     extent=[x_min, x_max, y_min, y_max], aspect='auto',
                     vmin=0, vmax=1)
    plt.colorbar(im3, ax=ax3, label='Heatmap value')
    
    # Mark GT centers
    if gt_boxes is not None:
        for box in gt_boxes:
            cid = box['class_id']
            ax3.plot(box['bbox_center_x'], box['bbox_center_y'], 'o',
                    color=CLASS_COLORS[cid], markersize=8, markeredgewidth=2,
                    markerfacecolor='none')
    
    # Mark regression mask locations
    mask_y, mask_x = np.where(reg_mask > 0)
    if len(mask_y) > 0:
        # Convert heatmap coords back to world coords
        res_x = cfg.pillar_x * cfg.head_stride
        res_y = cfg.pillar_y * cfg.head_stride
        world_x = mask_x * res_x + x_min
        world_y = mask_y * res_y + y_min
        ax3.scatter(world_x, world_y, s=50, c='cyan', marker='+', linewidths=1,
                   label=f'Reg mask ({len(mask_y)} pts)')
        ax3.legend(loc='upper right', fontsize=8)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Combined Heatmap (max across classes)\nShape: (H={hm_h}, W={hm_w})')
    
    # ═══════════════════════════════════════════════════════════════════════
    # Bottom row: Per-class heatmaps
    # ═══════════════════════════════════════════════════════════════════════
    for c in range(min(num_classes, 3)):  # Show up to 3 classes
        ax = fig.add_subplot(2, 3, 4 + c)
        
        class_hm = heatmap[c]  # (H, W)
        
        im = ax.imshow(class_hm, origin='lower', cmap='hot',
                       extent=[x_min, x_max, y_min, y_max], aspect='auto',
                       vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Heatmap')
        
        # Mark GT boxes of this class
        if gt_boxes is not None:
            for box in gt_boxes:
                if box['class_id'] == c:
                    ax.plot(box['bbox_center_x'], box['bbox_center_y'], 'o',
                           color='cyan', markersize=10, markeredgewidth=2,
                           markerfacecolor='none')
                    corners = bbox_corners_2d(box)
                    ax.plot(corners[:, 0], corners[:, 1], 'c-', linewidth=1.5)
        
        # Count Gaussian peaks
        peak_val = class_hm.max()
        num_peaks = (class_hm > 0.5).sum()
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{CLASS_NAMES[c]} Heatmap\nmax={peak_val:.3f}, peaks(>0.5)={num_peaks}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description='Visualize dataset BEV and heatmaps')
    parser.add_argument('--scene', type=str, default=None, help='Scene name (e.g., scene_1)')
    parser.add_argument('--pose', type=int, default=None, help='Pose index')
    parser.add_argument('--idx', type=int, default=0, help='Dataset index (if scene/pose not specified)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--save', type=str, default=None, help='Save path for figure')
    args = parser.parse_args()
    
    # Load config and dataset
    cfg = Config()
    
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    print(f"Config:")
    print(f"  Point cloud range X: [{cfg.point_cloud_range[0]}, {cfg.point_cloud_range[3]}]")
    print(f"  Point cloud range Y: [{cfg.point_cloud_range[1]}, {cfg.point_cloud_range[4]}]")
    print(f"  Grid: ({cfg.grid_x}, {cfg.grid_y})")
    print(f"  Heatmap: (H={cfg.heatmap_h}, W={cfg.heatmap_w})")
    print(f"  Head stride: {cfg.head_stride}")
    print()
    
    dataset = AirbusLidarDataset(cfg, split=args.split, gt_csv=cfg.gt_csv, augment=False)
    print(f"Dataset size: {len(dataset)} frames")
    
    # Find the sample
    if args.scene is not None and args.pose is not None:
        # Find by scene and pose using the manifest DataFrame
        found_idx = None
        for i in range(len(dataset.manifest)):
            row = dataset.manifest.iloc[i]
            if row['scene'] == args.scene and row['pose_index'] == args.pose:
                found_idx = i
                break
        if found_idx is None:
            print(f"ERROR: Could not find {args.scene} pose {args.pose}")
            print(f"Available scenes: {dataset.manifest['scene'].unique().tolist()}")
            return
        sample_idx = found_idx
    else:
        sample_idx = args.idx
    
    print(f"\nLoading sample {sample_idx}...")
    sample = dataset[sample_idx]
    
    scene = sample['scene']
    pose_index = sample['pose_index']
    
    print(f"  Scene: {scene}")
    print(f"  Pose: {pose_index}")
    print(f"  Num pillars: {len(sample['pillar_coords'])}")
    print(f"  Num objects (from reg_mask): {sample['reg_mask'].sum():.0f}")
    print(f"  Heatmap shape: {sample['heatmap'].shape}")
    
    # Load raw point cloud for visualization
    try:
        raw_xyz, _ = load_raw_frame(scene, pose_index, cfg.processed_dir)
        print(f"  Raw points: {len(raw_xyz)}")
    except FileNotFoundError:
        print("  (Raw point cloud not found, skipping)")
        raw_xyz = None
    
    # Get GT boxes for this frame
    gt_boxes = dataset.gt_by_frame.get((scene, pose_index), [])
    print(f"  GT boxes: {len(gt_boxes)}")
    for box in gt_boxes:
        print(f"    - {CLASS_NAMES[box['class_id']]}: center=({box['bbox_center_x']:.1f}, {box['bbox_center_y']:.1f})")
    
    # Visualize
    print("\nGenerating visualization...")
    save_path = args.save or f"bev_heatmap_{scene}_{pose_index}.png"
    visualize_sample(sample, cfg, raw_xyz=raw_xyz, gt_boxes=gt_boxes, save_path=save_path)
    
    # Print coordinate verification
    print("\n" + "=" * 70)
    print("COORDINATE VERIFICATION")
    print("=" * 70)
    
    if len(gt_boxes) > 0:
        box = gt_boxes[0]
        cx, cy = box['bbox_center_x'], box['bbox_center_y']
        
        # World → heatmap coords
        res_x = cfg.pillar_x * cfg.head_stride
        res_y = cfg.pillar_y * cfg.head_stride
        hm_x = (cx - cfg.point_cloud_range[0]) / res_x
        hm_y = (cy - cfg.point_cloud_range[1]) / res_y
        
        print(f"First GT box: {CLASS_NAMES[box['class_id']]}")
        print(f"  World coords: ({cx:.1f}, {cy:.1f})")
        print(f"  → Heatmap coords: ({hm_x:.1f}, {hm_y:.1f})")
        print(f"  → Heatmap int: (row={int(hm_y)}, col={int(hm_x)})")
        
        # Check heatmap value at that location
        hm_y_int, hm_x_int = int(hm_y), int(hm_x)
        if 0 <= hm_y_int < cfg.heatmap_h and 0 <= hm_x_int < cfg.heatmap_w:
            hm_val = sample['heatmap'][box['class_id'], hm_y_int, hm_x_int]
            print(f"  Heatmap value at [class={box['class_id']}, row={hm_y_int}, col={hm_x_int}]: {hm_val:.4f}")
            
            if hm_val > 0.9:
                print("  ✓ Gaussian peak found at expected location!")
            elif hm_val > 0.1:
                print("  ~ Gaussian present but not exactly at center (may be offset)")
            else:
                print("  ✗ No Gaussian at expected location - possible coordinate bug!")
        else:
            print(f"  ✗ Heatmap coords out of bounds! hm_y={hm_y_int} vs H={cfg.heatmap_h}, hm_x={hm_x_int} vs W={cfg.heatmap_w}")
    else:
        print("No GT boxes in this frame - try a different frame with objects.")
        print(f"Tip: Run with --idx N where N is a frame with objects")


if __name__ == "__main__":
    main()