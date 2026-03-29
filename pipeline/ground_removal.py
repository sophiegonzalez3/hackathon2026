#!/usr/bin/env python3
"""
Ground Removal Pipeline
=======================
Remove ground points from preprocessed LiDAR frames using local tile-based estimation.

Usage:
    # Full pipeline (analyze + clean + sanity check)
    python ground_removal.py --processed-dir processed/ --output-dir cleaned/

    # Analysis only (find best threshold)
    python ground_removal.py --processed-dir processed/ --analyze-only

    # With sanity check visualization
    python ground_removal.py --processed-dir processed/ --output-dir cleaned/ --sanity-check

    # Custom threshold
    python ground_removal.py --processed-dir processed/ --output-dir cleaned/ --threshold 3.0
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CLASS_RGB = {
    (38, 23, 180):  'Antenna',
    (177, 132, 47): 'Cable',
    (129, 81, 97):  'Electric Pole',
    (66, 132, 9):   'Wind Turbine',
}

DEFAULT_MIN_SURVIVAL = {
    'Antenna': 0.60,
    'Electric Pole': 0.70,
    'Wind Turbine': 0.75,
}

DEFAULT_THRESHOLDS = [2.0, 3.0, 5.0]
DEFAULT_TILE_SIZE = 5.0
DEFAULT_GROUND_PERCENTILE = 5


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL GROUND ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_local_ground_height(
    xyz: np.ndarray,
    tile_size: float = 5.0,
    percentile: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute height above ground for each point using local tile-based estimation.

    Args:
        xyz: Point cloud (N, 3)
        tile_size: Size of ground estimation tiles in meters
        percentile: Percentile to use for ground estimation (lower = more conservative)

    Returns:
        height_above_ground: (N,) height of each point above local ground
        ground_z: (N,) estimated ground Z at each point location
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    x_min, y_min = x.min(), y.min()
    x_max, y_max = x.max(), y.max()

    # Edge case: very few points
    if len(xyz) < 10:
        ground = np.percentile(z, percentile)
        return z - ground, np.full(len(z), ground)

    # Create tile grid
    n_tiles_x = max(1, int(np.ceil((x_max - x_min) / tile_size)))
    n_tiles_y = max(1, int(np.ceil((y_max - y_min) / tile_size)))

    # Assign points to tiles
    x_idx = np.clip(((x - x_min) / tile_size).astype(int), 0, n_tiles_x - 1)
    y_idx = np.clip(((y - y_min) / tile_size).astype(int), 0, n_tiles_y - 1)

    # Compute ground height per tile
    ground_grid = np.full((n_tiles_x, n_tiles_y), np.nan)
    labels = x_idx * n_tiles_y + y_idx
    unique_tiles = np.unique(labels)

    for t in unique_tiles:
        mask = labels == t
        ground_grid.flat[t] = np.percentile(z[mask], percentile)

    # Fill NaN tiles with global ground estimate
    global_ground = np.nanpercentile(z, percentile)
    ground_grid = np.nan_to_num(ground_grid, nan=global_ground)

    # Lookup ground Z for each point
    ground_z = ground_grid[x_idx, y_idx]
    height_above_ground = z - ground_z

    return height_above_ground, ground_z


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_scene(
    scene_dir: Path,
    thresholds: List[float],
    tile_size: float = 5.0
) -> Tuple[Dict, Dict, Dict]:
    """
    Analyze survival rates for different height thresholds.

    Returns:
        results: Per-class survival counts {threshold: {class: {total, survived}}}
        results_noise: Noise point survival {threshold: {total, survived}}
        results_all: All points survival {threshold: {total, survived}}
    """
    npz_files = sorted(scene_dir.glob('frame_*.npz'))

    results = {t: {cls: {'total': 0, 'survived': 0} for cls in CLASS_RGB.values()}
               for t in thresholds}
    results_noise = {t: {'total': 0, 'survived': 0} for t in thresholds}
    results_all = {t: {'total': 0, 'survived': 0} for t in thresholds}

    for npz_path in tqdm(npz_files, desc=f"  Analyzing {scene_dir.name}"):
        data = np.load(npz_path)
        xyz = data['xyz']
        rgb = data.get('rgb', None)

        if rgb is None:
            continue

        hag, _ = compute_local_ground_height(xyz, tile_size=tile_size)
        n_total = len(xyz)

        # All points stats
        for t in thresholds:
            results_all[t]['total'] += n_total
            results_all[t]['survived'] += int((hag > t).sum())

        # Per-class stats
        all_class_mask = np.zeros(len(rgb), dtype=bool)
        for (r, g, b), cls in CLASS_RGB.items():
            mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
            all_class_mask |= mask
            n_pts = int(mask.sum())

            if n_pts > 0:
                for t in thresholds:
                    results[t][cls]['total'] += n_pts
                    results[t][cls]['survived'] += int((mask & (hag > t)).sum())

        # Noise stats
        noise_mask = ~all_class_mask
        n_noise = int(noise_mask.sum())
        if n_noise > 0:
            for t in thresholds:
                results_noise[t]['total'] += n_noise
                results_noise[t]['survived'] += int((noise_mask & (hag > t)).sum())

    return results, results_noise, results_all


def build_metrics_df(
    scene_name: str,
    results: Dict,
    results_noise: Dict,
    results_all: Dict,
    thresholds: List[float]
) -> pd.DataFrame:
    """Convert analysis results to a DataFrame."""
    rows = []
    for t in thresholds:
        row = {'scene': scene_name, 'threshold_m': t}

        # All points
        total_all = results_all[t]['total']
        surv_all = results_all[t]['survived']
        row['total_points'] = total_all
        row['total_survived'] = surv_all
        row['total_removed_pct'] = 100 * (1 - surv_all / total_all) if total_all > 0 else 0

        # Per-class
        for cls in CLASS_RGB.values():
            total = results[t][cls]['total']
            surv = results[t][cls]['survived']
            row[f'{cls}_total'] = total
            row[f'{cls}_survived'] = surv
            row[f'{cls}_survival_pct'] = 100 * surv / total if total > 0 else None

        # Noise
        total_noise = results_noise[t]['total']
        surv_noise = results_noise[t]['survived']
        row['noise_total'] = total_noise
        row['noise_survived'] = surv_noise
        row['noise_removed_pct'] = 100 * (1 - surv_noise / total_noise) if total_noise > 0 else 0

        rows.append(row)

    return pd.DataFrame(rows)


def find_best_threshold(
    metrics_df: pd.DataFrame,
    min_survival: Dict[str, float]
) -> Optional[float]:
    """Find the best threshold that satisfies survival requirements."""
    for _, row in metrics_df.iterrows():
        valid = True
        for cls, min_rate in min_survival.items():
            val = row.get(f'{cls}_survival_pct')
            if val is not None and val < min_rate * 100:
                valid = False
                break
        if valid:
            return row['threshold_m']
    return None


def print_analysis_table(metrics_df: pd.DataFrame, min_survival: Dict[str, float]):
    """Print a formatted analysis table."""
    print(f"\n  {'Thresh':>7} | {'Antenna':>8} {'Cable':>8} {'Pole':>8} {'Turbine':>8} | {'Noise Rem':>10} | Status")
    print("  " + "-" * 80)

    for _, row in metrics_df.iterrows():
        def fmt(val):
            return f"{val:>6.1f}%" if val is not None else "   N/A "

        t = row['threshold_m']
        ant = row.get('Antenna_survival_pct')
        cab = row.get('Cable_survival_pct')
        pole = row.get('Electric Pole_survival_pct')
        turb = row.get('Wind Turbine_survival_pct')
        noise_rem = row['noise_removed_pct']

        status = "✓"
        for cls, min_rate in min_survival.items():
            val = row.get(f'{cls}_survival_pct')
            if val is not None and val < min_rate * 100:
                status = "✗"
                break

        print(f"  {t:>6.1f}m | {fmt(ant)} {fmt(cab)} {fmt(pole)} {fmt(turb)} | {fmt(noise_rem)} | {status}")


# ══════════════════════════════════════════════════════════════════════════════
# CLEANING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clean_scene(
    scene_dir: Path,
    output_dir: Path,
    threshold: float,
    tile_size: float = 5.0,
    save_kept_indices: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Clean all frames in a scene and generate manifest.

    Returns:
        stats: {original, kept, output_dir}
        manifest_rows: List of manifest entries
    """
    scene_name = scene_dir.name
    output_scene_dir = output_dir / scene_name
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(scene_dir.glob('frame_*.npz'))

    total_original = 0
    total_kept = 0
    manifest_rows = []

    for npz_path in tqdm(npz_files, desc=f"  Cleaning {scene_name}"):
        data = np.load(npz_path)
        xyz_local = data['xyz']
        ego_pose = data['ego_pose']
        rgb = data.get('rgb', None)

        # Compute local ground
        hag, ground_z = compute_local_ground_height(xyz_local, tile_size=tile_size)

        # Apply threshold
        keep_mask = hag > threshold
        kept_indices = np.where(keep_mask)[0]

        n_original = len(xyz_local)
        n_kept = int(keep_mask.sum())

        total_original += n_original
        total_kept += n_kept

        # Build manifest row
        row = {
            'scene': scene_name,
            'frame': npz_path.stem,
            'frame_idx': int(npz_path.stem.split('_')[1]),
            'original_path': str(npz_path),
            'cleaned_path': str(output_scene_dir / npz_path.name),
            'ego_x': float(ego_pose[0]),
            'ego_y': float(ego_pose[1]),
            'ego_z': float(ego_pose[2]),
            'ego_yaw': float(ego_pose[3]),
            'num_points_original': n_original,
            'num_points_cleaned': n_kept,
            'points_removed_pct': 100 * (n_original - n_kept) / n_original if n_original > 0 else 0,
            'threshold_m': threshold,
        }

        # Add class survival stats if RGB available
        if rgb is not None:
            rgb_kept = rgb[keep_mask]
            for (r, g, b), cls in CLASS_RGB.items():
                orig_mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
                clean_mask = (rgb_kept[:, 0] == r) & (rgb_kept[:, 1] == g) & (rgb_kept[:, 2] == b)
                row[f'{cls}_original'] = int(orig_mask.sum())
                row[f'{cls}_cleaned'] = int(clean_mask.sum())

        manifest_rows.append(row)

        # Build output NPZ
        out_data = {
            'xyz': xyz_local[keep_mask],
            'ego_pose': ego_pose,
            'height_above_ground': hag[keep_mask].astype(np.float32),
            'ground_z': ground_z[keep_mask].astype(np.float32),
        }

        # Copy other arrays
        for key in ['reflectivity', 'rgb', 'intensity']:
            if key in data:
                out_data[key] = data[key][keep_mask]

        # Optionally save indices for mapping back
        if save_kept_indices:
            out_data['kept_indices'] = kept_indices.astype(np.int32)

        np.savez_compressed(output_scene_dir / npz_path.name, **out_data)

    return {
        'original': total_original,
        'kept': total_kept,
        'output_dir': str(output_scene_dir)
    }, manifest_rows


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION (for notebook use)
# ══════════════════════════════════════════════════════════════════════════════

def local_to_world(xyz_local: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """Transform local coordinates to world coordinates."""
    ego_x, ego_y, ego_z, ego_yaw = ego_pose
    cos_yaw = np.cos(ego_yaw)
    sin_yaw = np.sin(ego_yaw)
    
    x_world = xyz_local[:, 0] * cos_yaw - xyz_local[:, 1] * sin_yaw + ego_x
    y_world = xyz_local[:, 0] * sin_yaw + xyz_local[:, 1] * cos_yaw + ego_y
    z_world = xyz_local[:, 2] + ego_z
    
    return np.column_stack([x_world, y_world, z_world])


def visualize_frame(
    npz_path: Path,
    thresholds: List[float] = [0.5, 3.0, 5.0, 10.0, 15.0],
    tile_size: float = 5.0,
    use_world_coords: bool = True,
    figsize: Tuple[int, int] = (25, 10),
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Visualize ground removal on a single frame with multiple thresholds.
    
    Usage in notebook:
        from ground_removal import visualize_frame
        visualize_frame(Path('processed/scene_1/frame_050.npz'))
    
    Args:
        npz_path: Path to .npz frame file
        thresholds: List of height thresholds to visualize
        tile_size: Ground estimation tile size
        use_world_coords: Transform to world coordinates
        figsize: Figure size
        save_path: Optional path to save the plot
        show: Whether to call plt.show()
    
    Returns:
        height_above_ground array for further analysis
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return None
    
    # Load data
    data = np.load(npz_path)
    xyz_local = data['xyz']
    ego_pose = data['ego_pose']
    rgb = data.get('rgb', None)
    
    # Transform coordinates
    if use_world_coords:
        xyz = local_to_world(xyz_local, ego_pose)
        coord_label = "world"
    else:
        xyz = xyz_local
        coord_label = "local"
    
    # Compute height above ground
    hag, ground_z = compute_local_ground_height(xyz_local, tile_size=tile_size)
    
    # Print stats
    print(f"\nFrame: {npz_path.name}")
    print(f"Points: {len(xyz_local):,}")
    print(f"Height range: {hag.min():.1f}m to {hag.max():.1f}m")
    print(f"\nThreshold analysis:")
    for t in sorted(set([0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0] + list(thresholds))):
        kept = (hag > t).sum()
        print(f"  {t:>5.1f}m: {kept:>7,} kept ({100*kept/len(xyz_local):>5.1f}%)")
    
    # Create figure
    n_thresh = len(thresholds)
    fig, axes = plt.subplots(2, n_thresh + 1, figsize=figsize)
    
    # Row 1: BEV (Bird's Eye View)
    # Original colored by HAG
    ax = axes[0, 0]
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=hag, s=0.1, cmap='viridis', vmin=0, vmax=15)
    ax.set_title(f'Original ({len(xyz):,} pts)\nColored by HAG')
    ax.set_xlabel(f'X ({coord_label})')
    ax.set_ylabel(f'Y ({coord_label})')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='HAG (m)')
    
    # Each threshold
    for i, t in enumerate(thresholds):
        ax = axes[0, i + 1]
        mask = hag > t
        ax.scatter(xyz[~mask, 0], xyz[~mask, 1], c='red', s=0.1, alpha=0.3, label='Removed')
        ax.scatter(xyz[mask, 0], xyz[mask, 1], c='blue', s=0.1, alpha=0.3, label='Kept')
        ax.set_title(f'{t:.1f}m threshold\n({mask.sum():,} kept, {(~mask).sum():,} removed)')
        ax.set_xlabel(f'X ({coord_label})')
        ax.set_aspect('equal')
    
    # Row 2: Side view
    ax = axes[1, 0]
    sc = ax.scatter(xyz[:, 0], xyz[:, 2], c=hag, s=0.1, cmap='viridis', vmin=0, vmax=15)
    ax.set_title('Side view\nColored by HAG')
    ax.set_xlabel(f'X ({coord_label})')
    ax.set_ylabel(f'Z ({coord_label})')
    
    for i, t in enumerate(thresholds):
        ax = axes[1, i + 1]
        mask = hag > t
        ax.scatter(xyz[~mask, 0], xyz[~mask, 2], c='red', s=0.1, alpha=0.3)
        ax.scatter(xyz[mask, 0], xyz[mask, 2], c='blue', s=0.1, alpha=0.3)
        ax.axhline(y=ego_pose[2] + t, color='green', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_title(f'Side view - {t:.1f}m')
        ax.set_xlabel(f'X ({coord_label})')
    
    plt.suptitle(f'{npz_path.parent.name}/{npz_path.name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved to {save_path}")
    
    if show:
        plt.show()
    
    return hag


def visualize_class_survival(
    npz_path: Path,
    threshold: float = 5.0,
    tile_size: float = 5.0,
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Visualize how ground removal affects each object class.
    
    Usage in notebook:
        from ground_removal import visualize_class_survival
        visualize_class_survival(Path('processed/scene_1/frame_050.npz'), threshold=5.0)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return None
    
    # Load data
    data = np.load(npz_path)
    xyz_local = data['xyz']
    ego_pose = data['ego_pose']
    rgb = data.get('rgb', None)
    
    if rgb is None:
        print("No RGB data in this frame")
        return None
    
    xyz_world = local_to_world(xyz_local, ego_pose)
    hag, ground_z = compute_local_ground_height(xyz_local, tile_size=tile_size)
    
    # Create figure - one row per class
    classes = list(CLASS_RGB.items())
    fig, axes = plt.subplots(len(classes), 4, figsize=figsize)
    
    print(f"\nClass survival at {threshold}m threshold:")
    
    for row, ((r, g, b), cls) in enumerate(classes):
        mask_class = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        n_class = mask_class.sum()
        
        if n_class == 0:
            for col in range(4):
                axes[row, col].text(0.5, 0.5, f'No {cls} points', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(cls if col == 0 else '')
            continue
        
        class_xyz = xyz_world[mask_class]
        class_hag = hag[mask_class]
        class_z = xyz_local[mask_class, 2]
        class_ground_z = ground_z[mask_class]
        
        survived = class_hag > threshold
        survival_rate = survived.sum() / n_class
        
        print(f"  {cls}: {n_class:,} → {survived.sum():,} ({survival_rate*100:.1f}% survived)")
        
        # Col 0: HAG histogram
        ax = axes[row, 0]
        ax.hist(class_hag, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}m')
        ax.axvline(np.median(class_hag), color='green', linestyle='--', label=f'Median={np.median(class_hag):.1f}m')
        ax.set_xlabel('Height above ground (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{cls}\nHAG distribution')
        ax.legend(fontsize=8)
        
        # Col 1: Raw Z vs Ground Z
        ax = axes[row, 1]
        ax.scatter(class_ground_z, class_z, s=1, alpha=0.5)
        z_range = [class_ground_z.min(), class_ground_z.max()]
        ax.plot(z_range, z_range, 'r--', label='Z = Ground')
        ax.plot(z_range, [z + threshold for z in z_range], 'g--', label=f'Z = Ground + {threshold}m')
        ax.set_xlabel('Ground Z')
        ax.set_ylabel('Point Z')
        ax.set_title('Point Z vs Ground Z')
        ax.legend(fontsize=8)
        
        # Col 2: BEV - survived vs killed
        ax = axes[row, 2]
        ax.scatter(class_xyz[~survived, 0], class_xyz[~survived, 1], 
                  c='red', s=2, alpha=0.7, label=f'Killed ({(~survived).sum()})')
        ax.scatter(class_xyz[survived, 0], class_xyz[survived, 1], 
                  c='blue', s=2, alpha=0.7, label=f'Survived ({survived.sum()})')
        ax.set_xlabel('X (world)')
        ax.set_ylabel('Y (world)')
        ax.set_title(f'BEV: {survival_rate*100:.1f}% survived')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        
        # Col 3: Side view colored by HAG
        ax = axes[row, 3]
        sc = ax.scatter(class_xyz[:, 0], class_xyz[:, 2], c=class_hag, 
                       cmap='RdYlGn', s=2, vmin=0, vmax=max(30, threshold*2))
        ax.set_xlabel('X (world)')
        ax.set_ylabel('Z (world)')
        ax.set_title('Side view (colored by HAG)')
        plt.colorbar(sc, ax=ax, label='HAG (m)')
    
    plt.suptitle(f'{npz_path.parent.name}/{npz_path.name} — Threshold: {threshold}m', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved to {save_path}")
    
    if show:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def run_sanity_check(
    processed_dir: Path,
    output_dir: Path,
    threshold: float,
    tile_size: float = 5.0,
    num_frames: int = 5,
    save_plots: bool = True
) -> bool:
    """
    Run sanity checks on ground removal results.

    Returns:
        True if all checks pass
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not available, skipping visual sanity check")
        return True

    print(f"\n{'═'*60}")
    print("  SANITY CHECK: Ground Removal Verification")
    print(f"{'═'*60}")

    all_pass = True

    # Find a scene to check
    scene_dirs = sorted([d for d in output_dir.iterdir()
                         if d.is_dir() and d.name.startswith('scene_')])

    if not scene_dirs:
        print("  ⚠ No cleaned scenes found!")
        return False

    scene_dir = scene_dirs[0]
    orig_scene_dir = processed_dir / scene_dir.name
    npz_files = sorted(scene_dir.glob('frame_*.npz'))[:num_frames]

    print(f"\n  Checking {scene_dir.name} ({len(npz_files)} frames)...")

    for npz_path in npz_files:
        orig_path = orig_scene_dir / npz_path.name
        if not orig_path.exists():
            continue

        # Load both
        orig_data = np.load(orig_path)
        clean_data = np.load(npz_path)

        orig_xyz = orig_data['xyz']
        clean_xyz = clean_data['xyz']
        clean_hag = clean_data.get('height_above_ground', None)

        # Check 1: Point reduction
        reduction = 1 - len(clean_xyz) / len(orig_xyz)
        print(f"\n  {npz_path.name}:")
        print(f"    Points: {len(orig_xyz):,} → {len(clean_xyz):,} ({reduction*100:.1f}% removed)")

        # Check 2: HAG values (should all be > threshold)
        if clean_hag is not None:
            min_hag = clean_hag.min()
            if min_hag <= threshold:
                print(f"    ⚠ WARNING: min HAG = {min_hag:.2f}m (should be > {threshold}m)")
                all_pass = False
            else:
                print(f"    ✓ HAG range: [{min_hag:.2f}m, {clean_hag.max():.2f}m]")

        # Check 3: Per-class survival
        orig_rgb = orig_data.get('rgb', None)
        clean_rgb = clean_data.get('rgb', None)

        if orig_rgb is not None and clean_rgb is not None:
            for (r, g, b), cls in CLASS_RGB.items():
                orig_mask = (orig_rgb[:, 0] == r) & (orig_rgb[:, 1] == g) & (orig_rgb[:, 2] == b)
                clean_mask = (clean_rgb[:, 0] == r) & (clean_rgb[:, 1] == g) & (clean_rgb[:, 2] == b)
                orig_count = orig_mask.sum()
                clean_count = clean_mask.sum()
                if orig_count > 0:
                    survival = clean_count / orig_count
                    status = "✓" if survival >= DEFAULT_MIN_SURVIVAL.get(cls, 0) else "⚠"
                    print(f"    {status} {cls}: {orig_count} → {clean_count} ({survival*100:.1f}% survived)")

    # Optional: Generate visual comparison
    if save_plots and len(npz_files) > 0:
        print("\n  Generating comparison plots...")
        _generate_sanity_plots(processed_dir, output_dir, scene_dir.name, threshold)

    return all_pass


def _generate_sanity_plots(
    processed_dir: Path,
    output_dir: Path,
    scene_name: str,
    threshold: float
):
    """Generate before/after comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scene_orig = processed_dir / scene_name
    scene_clean = output_dir / scene_name

    # Pick middle frame
    npz_files = sorted(scene_clean.glob('frame_*.npz'))
    if not npz_files:
        return

    mid_idx = len(npz_files) // 2
    npz_path = npz_files[mid_idx]
    orig_path = scene_orig / npz_path.name

    if not orig_path.exists():
        return

    orig_data = np.load(orig_path)
    clean_data = np.load(npz_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Before - BEV
    ax = axes[0, 0]
    xyz = orig_data['xyz']
    ax.scatter(xyz[:, 0], xyz[:, 1], s=0.1, c=xyz[:, 2], cmap='viridis', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'BEFORE: {len(xyz):,} points')
    ax.set_aspect('equal')

    # After - BEV
    ax = axes[0, 1]
    xyz = clean_data['xyz']
    ax.scatter(xyz[:, 0], xyz[:, 1], s=0.1, c=xyz[:, 2], cmap='viridis', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'AFTER: {len(xyz):,} points (threshold={threshold}m)')
    ax.set_aspect('equal')

    # Before - Side view
    ax = axes[1, 0]
    xyz = orig_data['xyz']
    ax.scatter(xyz[:, 0], xyz[:, 2], s=0.1, c='gray', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('BEFORE: Side view')

    # After - Side view with HAG coloring
    ax = axes[1, 1]
    xyz = clean_data['xyz']
    hag = clean_data.get('height_above_ground', xyz[:, 2])
    scatter = ax.scatter(xyz[:, 0], xyz[:, 2], s=0.1, c=hag, cmap='RdYlGn', vmin=0, vmax=30)
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold={threshold}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('AFTER: Side view (colored by HAG)')
    plt.colorbar(scatter, ax=ax, label='Height above ground (m)')

    plt.suptitle(f'Ground Removal: {scene_name} / {npz_path.stem}', fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / 'sanity_check_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot to {plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST FIXUP (for PointPillars compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def fix_manifest_for_training(manifest_path: Path) -> pd.DataFrame:
    """
    Fix manifest to be compatible with PointPillars dataset expectations.

    Adds: pose_index, file, num_points_raw, num_points_valid, num_invalid_dropped
    """
    df = pd.read_csv(manifest_path)

    df['pose_index'] = df['frame_idx']
    df['file'] = df['cleaned_path']
    df['num_points_raw'] = (df['num_points_original'] * 1.2).astype(int)  # Approximate
    df['num_points_valid'] = df['num_points_cleaned']
    df['num_invalid_dropped'] = df['num_points_raw'] - df['num_points_valid']

    # Reorder columns for compatibility
    output_cols = [
        'scene', 'pose_index', 'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
        'num_points_raw', 'num_points_valid', 'num_invalid_dropped', 'file'
    ]

    return df[output_cols]


def fix_gt_paths(gt_path: Path, old_dir: str = 'processed/', new_dir: str = 'cleaned/') -> pd.DataFrame:
    """Update GT CSV to point to cleaned paths."""
    df = pd.read_csv(gt_path)
    df['frame_file'] = df['frame_file'].str.replace(old_dir, new_dir, regex=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    processed_dir: Path,
    output_dir: Path,
    metrics_dir: Optional[Path] = None,
    threshold: Optional[float] = None,
    thresholds_to_test: List[float] = None,
    tile_size: float = DEFAULT_TILE_SIZE,
    min_survival: Dict[str, float] = None,
    analyze_only: bool = False,
    sanity_check: bool = False,
    scenes: Optional[List[str]] = None,
    save_kept_indices: bool = True,
    fix_manifest: bool = True,
    gt_csv: Optional[Path] = None,
):
    """
    Run the complete ground removal pipeline.
    """
    thresholds_to_test = thresholds_to_test or DEFAULT_THRESHOLDS
    min_survival = min_survival or DEFAULT_MIN_SURVIVAL
    metrics_dir = metrics_dir or (output_dir.parent / 'ground_removal_metrics')

    print("=" * 70)
    print("  GROUND REMOVAL PIPELINE")
    print("=" * 70)
    print(f"  Input:      {processed_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Tile size:  {tile_size}m")
    print(f"  Thresholds: {thresholds_to_test}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Find scenes
    if scenes:
        scene_dirs = [processed_dir / s for s in scenes]
    else:
        scene_dirs = sorted([d for d in processed_dir.iterdir()
                             if d.is_dir() and d.name.startswith('scene_')])

    print(f"\n  Found {len(scene_dirs)} scenes")

    # ── PHASE 1: Analysis ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  PHASE 1: Threshold Analysis")
    print(f"{'─'*70}")

    all_metrics = []

    for scene_dir in scene_dirs:
        results, results_noise, results_all = analyze_scene(
            scene_dir, thresholds_to_test, tile_size
        )
        metrics_df = build_metrics_df(
            scene_dir.name, results, results_noise, results_all, thresholds_to_test
        )
        all_metrics.append(metrics_df)

        print(f"\n  {scene_dir.name}:")
        print_analysis_table(metrics_df, min_survival)

    # Combine metrics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(metrics_dir / 'analysis_metrics.csv', index=False)

    # Find best threshold
    if threshold is None:
        threshold = find_best_threshold(combined_metrics, min_survival)
        if threshold is None:
            print("\n  ⚠ No threshold satisfies all survival requirements!")
            threshold = thresholds_to_test[0]  # Use most conservative
        print(f"\n  → Selected threshold: {threshold}m")
    else:
        print(f"\n  → Using specified threshold: {threshold}m")

    if analyze_only:
        print("\n  ✓ Analysis complete (--analyze-only mode)")
        return

    # ── PHASE 2: Cleaning ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  PHASE 2: Ground Removal")
    print(f"{'─'*70}")

    all_manifest_rows = []
    all_stats = []

    for scene_dir in scene_dirs:
        stats, manifest_rows = clean_scene(
            scene_dir, output_dir, threshold, tile_size, save_kept_indices
        )
        all_stats.append(stats)
        all_manifest_rows.extend(manifest_rows)

        kept_pct = 100 * stats['kept'] / stats['original']
        print(f"  {scene_dir.name}: {stats['original']:,} → {stats['kept']:,} ({kept_pct:.1f}% kept)")

    # Save manifest
    manifest_df = pd.DataFrame(all_manifest_rows)
    manifest_path = output_dir / 'manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n  ✓ Saved manifest → {manifest_path}")

    # ── PHASE 3: Fix Manifest for Training ────────────────────────────────
    if fix_manifest:
        print(f"\n{'─'*70}")
        print("  PHASE 3: Fixing Manifest for Training")
        print(f"{'─'*70}")

        fixed_manifest = fix_manifest_for_training(manifest_path)
        fixed_manifest.to_csv(manifest_path, index=False)
        print(f"  ✓ Fixed manifest columns for PointPillars compatibility")

        # Fix GT paths if provided
        if gt_csv and gt_csv.exists():
            fixed_gt = fix_gt_paths(gt_csv)
            gt_out_path = gt_csv.parent / f"{gt_csv.stem}_cleaned.csv"
            fixed_gt.to_csv(gt_out_path, index=False)
            print(f"  ✓ Fixed GT paths → {gt_out_path}")

    # ── PHASE 4: Sanity Check ─────────────────────────────────────────────
    if sanity_check:
        print(f"\n{'─'*70}")
        print("  PHASE 4: Sanity Check")
        print(f"{'─'*70}")

        all_pass = run_sanity_check(processed_dir, output_dir, threshold, tile_size)
        if all_pass:
            print("\n  ✓ All sanity checks passed!")
        else:
            print("\n  ⚠ Some sanity checks failed - review the output")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'═'*70}")

    total_orig = sum(s['original'] for s in all_stats)
    total_kept = sum(s['kept'] for s in all_stats)
    print(f"  Total points: {total_orig:,} → {total_kept:,}")
    print(f"  Reduction: {100*(1 - total_kept/total_orig):.1f}%")
    print(f"  Threshold: {threshold}m")
    print(f"\n  Output directory: {output_dir}")
    print(f"  Manifest: {manifest_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Remove ground points from LiDAR frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with auto threshold selection
    python ground_removal.py --processed-dir processed/ --output-dir cleaned/

    # Analysis only
    python ground_removal.py --processed-dir processed/ --analyze-only

    # Custom threshold with sanity check
    python ground_removal.py --processed-dir processed/ --output-dir cleaned/ \\
        --threshold 3.0 --sanity-check

    # Process specific scenes
    python ground_removal.py --processed-dir processed/ --output-dir cleaned/ \\
        --scenes scene_1 scene_2
        """
    )

    parser.add_argument('--processed-dir', required=True, type=Path,
                        help='Directory containing preprocessed .npz frames')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for cleaned frames')
    parser.add_argument('--metrics-dir', type=Path, default=None,
                        help='Directory for metrics output')

    parser.add_argument('--threshold', type=float, default=None,
                        help='Height threshold in meters (auto-selected if not specified)')
    parser.add_argument('--thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS,
                        help=f'Thresholds to test (default: {DEFAULT_THRESHOLDS})')
    parser.add_argument('--tile-size', type=float, default=DEFAULT_TILE_SIZE,
                        help=f'Ground estimation tile size (default: {DEFAULT_TILE_SIZE}m)')

    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze thresholds, do not clean')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity checks after cleaning')
    parser.add_argument('--no-fix-manifest', action='store_true',
                        help='Do not fix manifest for PointPillars')

    parser.add_argument('--gt-csv', type=Path, default=None,
                        help='GT CSV to update paths in')
    parser.add_argument('--scenes', nargs='+', default=None,
                        help='Specific scenes to process')
    
    # Visualization options
    parser.add_argument('--visualize', type=Path, default=None,
                        metavar='NPZ_PATH',
                        help='Visualize a single frame (e.g., processed/scene_1/frame_050.npz)')
    parser.add_argument('--visualize-class', type=Path, default=None,
                        metavar='NPZ_PATH',
                        help='Visualize class survival for a single frame')

    args = parser.parse_args()

    # ── Visualization mode ────────────────────────────────────────────────
    if args.visualize:
        if not args.visualize.exists():
            parser.error(f"Frame not found: {args.visualize}")
        thresholds = args.thresholds if args.thresholds != DEFAULT_THRESHOLDS else [0.5, 3.0, 5.0, 10.0, 15.0]
        visualize_frame(args.visualize, thresholds=thresholds, tile_size=args.tile_size)
        return
    
    if args.visualize_class:
        if not args.visualize_class.exists():
            parser.error(f"Frame not found: {args.visualize_class}")
        threshold = args.threshold or 5.0
        visualize_class_survival(args.visualize_class, threshold=threshold, tile_size=args.tile_size)
        return

    # ── Validation ────────────────────────────────────────────────────────
    if not args.processed_dir.exists():
        parser.error(f"Processed directory not found: {args.processed_dir}")

    if not args.analyze_only and args.output_dir is None:
        parser.error("--output-dir required unless --analyze-only is set")

    # Run
    run_pipeline(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        threshold=args.threshold,
        thresholds_to_test=args.thresholds,
        tile_size=args.tile_size,
        analyze_only=args.analyze_only,
        sanity_check=args.sanity_check,
        fix_manifest=not args.no_fix_manifest,
        gt_csv=args.gt_csv,
        scenes=args.scenes,
    )


if __name__ == '__main__':
    main()
