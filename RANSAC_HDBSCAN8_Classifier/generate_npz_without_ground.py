# ══════════════════════════════════════════════════════════════════════════════
# LOCAL GROUND REMOVAL PIPELINE v3 (with manifest + extra metrics)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
LOCAL_ROOT = Path("/content/hackathon")
PROCESSED_DIR = LOCAL_ROOT / "processed"
OUTPUT_DIR = LOCAL_ROOT / "cleaned"
METRICS_DIR = LOCAL_ROOT / "ground_removal_metrics"

SCENES_TO_PROCESS = None  # None = all, or ["scene_1"] for testing

CLASS_RGB = {
    (38, 23, 180):  'Antenna',
    (177, 132, 47): 'Cable',
    (129, 81, 97):  'Electric Pole',
    (66, 132, 9):   'Wind Turbine',
}

MIN_SURVIVAL_PER_CLASS = {
    'Antenna': 0.60,
    'Electric Pole': 0.70,
    'Wind Turbine': 0.75,
}

THRESHOLDS_TO_TEST = [2,5]
TILE_SIZE = 5.0
GROUND_PERCENTILE = 5

# Whether to save indices for mapping back to original frames
SAVE_KEPT_INDICES = True  # Useful for mapping detections back to original

# ══════════════════════════════════════════════════════════════════════════════
# LOCAL GROUND ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_local_ground_height_fast(xyz, tile_size=5.0, percentile=5):
    """Compute height above ground for each point using LOCAL tiling."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    x_min, y_min = x.min(), y.min()
    x_max, y_max = x.max(), y.max()

    if len(xyz) < 10:
        ground = np.percentile(z, percentile)
        return z - ground, np.full(len(z), ground)

    n_tiles_x = max(1, int(np.ceil((x_max - x_min) / tile_size)))
    n_tiles_y = max(1, int(np.ceil((y_max - y_min) / tile_size)))

    x_idx = np.clip(((x - x_min) / tile_size).astype(int), 0, n_tiles_x - 1)
    y_idx = np.clip(((y - y_min) / tile_size).astype(int), 0, n_tiles_y - 1)

    ground_grid = np.full((n_tiles_x, n_tiles_y), np.nan)
    labels = x_idx * n_tiles_y + y_idx
    unique_tiles = np.unique(labels)

    for t in unique_tiles:
        mask = labels == t
        ground_grid.flat[t] = np.percentile(z[mask], percentile)

    global_ground = np.nanpercentile(z, percentile)
    ground_grid = np.nan_to_num(ground_grid, nan=global_ground)

    ground_z = ground_grid[x_idx, y_idx]
    height_above_ground = z - ground_z

    return height_above_ground, ground_z


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE SCENE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_scene_local_ground(scene_dir, thresholds, tile_size=5.0):
    """Analyze survival rates using LOCAL ground estimation per frame."""
    scene_dir = Path(scene_dir)
    npz_files = sorted(scene_dir.glob('frame_*.npz'))

    print(f"  Analyzing {len(npz_files)} frames with LOCAL ground model (tile={tile_size}m)...")

    results = {t: {cls: {'total': 0, 'survived': 0} for cls in CLASS_RGB.values()}
               for t in thresholds}
    results_noise = {t: {'total': 0, 'survived': 0} for t in thresholds}
    results_all = {t: {'total': 0, 'survived': 0} for t in thresholds}

    for npz_path in tqdm(npz_files, desc="  Processing frames"):
        data = np.load(npz_path)
        xyz_local = data['xyz']
        rgb = data.get('rgb', None)

        if rgb is None:
            continue

        hag, _ = compute_local_ground_height_fast(xyz_local, tile_size=tile_size)

        n_total = len(xyz_local)
        for t in thresholds:
            results_all[t]['total'] += n_total
            results_all[t]['survived'] += int((hag > t).sum())

        all_class_mask = np.zeros(len(rgb), dtype=bool)

        for (r, g, b), cls in CLASS_RGB.items():
            mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
            all_class_mask |= mask
            n_pts = int(mask.sum())

            if n_pts > 0:
                for t in thresholds:
                    results[t][cls]['total'] += n_pts
                    results[t][cls]['survived'] += int((mask & (hag > t)).sum())

        noise_mask = ~all_class_mask
        n_noise = int(noise_mask.sum())
        if n_noise > 0:
            for t in thresholds:
                results_noise[t]['total'] += n_noise
                results_noise[t]['survived'] += int((noise_mask & (hag > t)).sum())

    return results, results_noise, results_all


def build_metrics_dataframe(scene_name, results, results_noise, results_all, thresholds):
    """Build metrics DataFrame."""
    rows = []

    for t in thresholds:
        row = {'scene': scene_name, 'threshold_m': int(t)}

        for cls in CLASS_RGB.values():
            total = int(results[t][cls]['total'])
            survived = int(results[t][cls]['survived'])
            row[f'{cls}_total'] = total
            row[f'{cls}_survived'] = survived
            row[f'{cls}_survival_pct'] = (survived / total * 100) if total > 0 else None

        noise_total = int(results_noise[t]['total'])
        noise_survived = int(results_noise[t]['survived'])
        row['noise_total'] = noise_total
        row['noise_survived'] = noise_survived
        row['noise_survival_pct'] = (noise_survived / noise_total * 100) if noise_total > 0 else 0
        row['noise_removed_pct'] = 100 - row['noise_survival_pct']

        all_total = int(results_all[t]['total'])
        all_survived = int(results_all[t]['survived'])
        row['all_points_total'] = all_total
        row['all_points_survived'] = all_survived
        row['all_points_survival_pct'] = (all_survived / all_total * 100) if all_total > 0 else 0

        rows.append(row)

    return pd.DataFrame(rows)


def find_optimal_threshold(metrics_df, min_survival_per_class):
    """Find highest threshold that meets ALL per-class survival requirements."""
    thresholds = sorted(metrics_df['threshold_m'].unique())
    optimal = thresholds[0]

    for t in thresholds:
        row = metrics_df[metrics_df['threshold_m'] == t].iloc[0]

        meets_all = True
        for cls, min_rate in min_survival_per_class.items():
            col = f'{cls}_survival_pct'
            val = row.get(col)
            if val is not None and val < min_rate * 100:
                meets_all = False
                break

        if meets_all:
            optimal = t

    return optimal


def print_metrics_table(metrics_df, min_survival_per_class):
    """Print metrics table with status based on per-class requirements."""
    print(f"\n  {'Thresh':>7} | {'Antenna':>8} {'Cable':>8} {'Pole':>8} {'Turbine':>8} | {'Noise Rem':>10} | {'Status'}")
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
        for cls, min_rate in min_survival_per_class.items():
            val = row.get(f'{cls}_survival_pct')
            if val is not None and val < min_rate * 100:
                status = "✗"
                break

        print(f"  {t:>6.0f}m | {fmt(ant)} {fmt(cab)} {fmt(pole)} {fmt(turb)} | {fmt(noise_rem)} | {status}")

    print(f"\n  Requirements: ", end="")
    print(", ".join([f"{cls}≥{rate*100:.0f}%" for cls, rate in min_survival_per_class.items()]))


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN AND SAVE WITH MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

def clean_and_save_scene_with_manifest(scene_dir, output_dir, threshold, tile_size=5.0,
                                        save_kept_indices=True):
    """
    Clean all frames using LOCAL ground model, save cleaned NPZ + manifest.

    Returns:
        stats: dict with totals
        manifest_rows: list of dicts for manifest.csv
    """
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.name
    output_scene_dir = Path(output_dir) / scene_name
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(scene_dir.glob('frame_*.npz'))

    total_original = 0
    total_kept = 0
    manifest_rows = []

    print(f"  Cleaning {len(npz_files)} frames (threshold={threshold}m, tile={tile_size}m)...")

    for npz_path in tqdm(npz_files, desc="  Saving"):
        data = np.load(npz_path)
        xyz_local = data['xyz']
        ego_pose = data['ego_pose']
        rgb = data.get('rgb', None)
        reflectivity = data.get('reflectivity', None)

        # Compute local ground
        hag, ground_z = compute_local_ground_height_fast(xyz_local, tile_size=tile_size)

        # Keep mask
        keep_mask = hag > threshold
        kept_indices = np.where(keep_mask)[0]

        n_original = len(xyz_local)
        n_kept = int(keep_mask.sum())

        total_original += n_original
        total_kept += n_kept

        # ── Build manifest row ──
        row = {
            # Frame identification
            'scene': scene_name,
            'frame': npz_path.stem,
            'frame_idx': int(npz_path.stem.split('_')[1]),

            # Paths
            'original_path': str(npz_path),
            'cleaned_path': str(output_scene_dir / npz_path.name),

            # Ego pose (for inference - need to transform detections)
            'ego_x': float(ego_pose[0]),
            'ego_y': float(ego_pose[1]),
            'ego_z': float(ego_pose[2]),
            'ego_yaw': float(ego_pose[3]),

            # Point counts
            'num_points_original': n_original,
            'num_points_cleaned': n_kept,
            'points_removed_pct': round(100 * (1 - n_kept / n_original), 2) if n_original > 0 else 0,

            # Ground stats (useful for understanding the scene)
            'ground_z_min': float(ground_z.min()),
            'ground_z_max': float(ground_z.max()),
            'ground_z_mean': float(ground_z.mean()),

            # Height stats after cleaning
            'hag_min': float(hag[keep_mask].min()) if n_kept > 0 else None,
            'hag_max': float(hag[keep_mask].max()) if n_kept > 0 else None,
            'hag_mean': float(hag[keep_mask].mean()) if n_kept > 0 else None,

            # Spatial extent (for PointPillars pillar grid)
            'x_min': float(xyz_local[keep_mask, 0].min()) if n_kept > 0 else None,
            'x_max': float(xyz_local[keep_mask, 0].max()) if n_kept > 0 else None,
            'y_min': float(xyz_local[keep_mask, 1].min()) if n_kept > 0 else None,
            'y_max': float(xyz_local[keep_mask, 1].max()) if n_kept > 0 else None,
            'z_min': float(xyz_local[keep_mask, 2].min()) if n_kept > 0 else None,
            'z_max': float(xyz_local[keep_mask, 2].max()) if n_kept > 0 else None,

            # Cleaning params (for reproducibility)
            'threshold_m': float(threshold),
            'tile_size_m': float(tile_size),
        }

        # ── Per-class point counts (for training data analysis) ──
        if rgb is not None:
            rgb_kept = rgb[keep_mask]
            for (r, g, b), cls in CLASS_RGB.items():
                # Original counts
                mask_orig = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
                row[f'{cls}_original'] = int(mask_orig.sum())

                # Cleaned counts
                mask_clean = (rgb_kept[:, 0] == r) & (rgb_kept[:, 1] == g) & (rgb_kept[:, 2] == b)
                row[f'{cls}_cleaned'] = int(mask_clean.sum())

            # Noise counts
            all_class_mask_orig = np.zeros(len(rgb), dtype=bool)
            all_class_mask_clean = np.zeros(len(rgb_kept), dtype=bool)
            for (r, g, b), cls in CLASS_RGB.items():
                all_class_mask_orig |= (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
                all_class_mask_clean |= (rgb_kept[:, 0] == r) & (rgb_kept[:, 1] == g) & (rgb_kept[:, 2] == b)

            row['noise_original'] = int((~all_class_mask_orig).sum())
            row['noise_cleaned'] = int((~all_class_mask_clean).sum())

        manifest_rows.append(row)

        # ── Build output NPZ ──
        out_data = {
            'xyz': xyz_local[keep_mask],
            'ego_pose': ego_pose,
            'height_above_ground': hag[keep_mask].astype(np.float32),
            'ground_z': ground_z[keep_mask].astype(np.float32),  # Added: ground Z at each point
        }

        # Copy other arrays
        for key in ['reflectivity', 'rgb', 'intensity']:
            if key in data:
                out_data[key] = data[key][keep_mask]

        # Save indices for mapping back to original (useful for inference)
        if save_kept_indices:
            out_data['kept_indices'] = kept_indices.astype(np.int32)

        np.savez_compressed(output_scene_dir / npz_path.name, **out_data)

    return {
        'original': total_original,
        'kept': total_kept,
        'output_dir': str(output_scene_dir)
    }, manifest_rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_scene(scene_dir, output_dir, metrics_dir, tile_size=5.0,
                  min_survival_per_class=None, save_kept_indices=True):
    """Process one scene with LOCAL ground removal."""
    if min_survival_per_class is None:
        min_survival_per_class = {}

    scene_name = Path(scene_dir).name
    print(f"\n{'='*70}")
    print(f"  PROCESSING {scene_name} (LOCAL ground model)")
    print(f"{'='*70}")

    # Analyze
    results, results_noise, results_all = analyze_scene_local_ground(
        scene_dir, THRESHOLDS_TO_TEST, tile_size=tile_size
    )

    # Build metrics
    metrics_df = build_metrics_dataframe(
        scene_name, results, results_noise, results_all, THRESHOLDS_TO_TEST
    )

    # Save metrics
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_dir / f"{scene_name}_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n  ✓ Metrics saved to: {metrics_csv}")

    # Print table
    print_metrics_table(metrics_df, min_survival_per_class)

    # Find optimal threshold
    optimal = find_optimal_threshold(metrics_df, min_survival_per_class)
    print(f"\n  ★ OPTIMAL THRESHOLD: {optimal}m")

    # Clean and save with manifest
    stats, manifest_rows = clean_and_save_scene_with_manifest(
        scene_dir, output_dir, optimal,
        tile_size=tile_size,
        save_kept_indices=save_kept_indices
    )
    stats['threshold'] = int(optimal)
    stats['scene'] = scene_name

    # Save summary JSON
    summary = {
        'scene': scene_name,
        'timestamp': datetime.now().isoformat(),
        'method': 'local_ground_model',
        'tile_size': float(tile_size),
        'threshold_used': int(optimal),
        'thresholds_tested': [int(t) for t in THRESHOLDS_TO_TEST],
        'min_survival_requirements': {k: float(v) for k, v in min_survival_per_class.items()},
        'points_original': int(stats['original']),
        'points_kept': int(stats['kept']),
        'points_removed_pct': round(100 * (1 - stats['kept'] / stats['original']), 2),
        'output_dir': stats['output_dir'],
        'kept_indices_saved': save_kept_indices,
    }

    with open(metrics_dir / f"{scene_name}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return stats, metrics_df, manifest_rows


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("  LOCAL GROUND REMOVAL PIPELINE v3 (with manifest)")
print("="*70)
print(f"  Method:     LOCAL ground estimation (per-frame)")
print(f"  Tile size:  {TILE_SIZE}m")
print(f"  Thresholds: {THRESHOLDS_TO_TEST}")
print(f"  Save kept_indices: {SAVE_KEPT_INDICES}")
print(f"\n  Per-class survival requirements:")
for cls, rate in MIN_SURVIVAL_PER_CLASS.items():
    print(f"    {cls}: ≥{rate*100:.0f}%")
print("="*70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Find scenes
if SCENES_TO_PROCESS:
    scene_dirs = [PROCESSED_DIR / s for s in SCENES_TO_PROCESS]
else:
    scene_dirs = sorted([d for d in PROCESSED_DIR.iterdir()
                         if d.is_dir() and d.name.startswith('scene_')])

print(f"\nProcessing {len(scene_dirs)} scenes...")

all_stats = []
all_metrics = []
all_manifest_rows = []

for scene_dir in scene_dirs:
    stats, metrics_df, manifest_rows = process_scene(
        scene_dir, OUTPUT_DIR, METRICS_DIR,
        tile_size=TILE_SIZE,
        min_survival_per_class=MIN_SURVIVAL_PER_CLASS,
        save_kept_indices=SAVE_KEPT_INDICES
    )
    all_stats.append(stats)
    all_metrics.append(metrics_df)
    all_manifest_rows.extend(manifest_rows)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE COMBINED OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

# Combined metrics
combined_metrics = pd.concat(all_metrics, ignore_index=True)
combined_metrics.to_csv(METRICS_DIR / "all_scenes_metrics.csv", index=False)

# Combined manifest
manifest_df = pd.DataFrame(all_manifest_rows)
manifest_path = OUTPUT_DIR / "manifest.csv"
manifest_df.to_csv(manifest_path, index=False)

print(f"\n✓ Manifest saved to: {manifest_path}")
print(f"  Columns: {list(manifest_df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  FINAL SUMMARY")
print("="*70)
print(f"{'Scene':<12} {'Threshold':>10} {'Original':>14} {'Kept':>14} {'Removed':>10}")
print("-"*65)

total_orig = 0
total_kept = 0
for s in all_stats:
    rem = 100 * (1 - s['kept'] / s['original'])
    print(f"{s['scene']:<12} {s['threshold']:>9}m {s['original']:>14,} {s['kept']:>14,} {rem:>9.1f}%")
    total_orig += s['original']
    total_kept += s['kept']

print("-"*65)
total_rem = 100 * (1 - total_kept / total_orig)
print(f"{'TOTAL':<12} {'-':>10} {total_orig:>14,} {total_kept:>14,} {total_rem:>9.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  MANIFEST STATISTICS")
print("="*70)
print(f"  Total frames: {len(manifest_df)}")
print(f"  Scenes: {manifest_df['scene'].nunique()}")
print(f"\n  Point counts (cleaned):")
print(f"    Min:    {manifest_df['num_points_cleaned'].min():,}")
print(f"    Max:    {manifest_df['num_points_cleaned'].max():,}")
print(f"    Mean:   {manifest_df['num_points_cleaned'].mean():,.0f}")
print(f"    Median: {manifest_df['num_points_cleaned'].median():,.0f}")

print(f"\n  Spatial extent (cleaned, local coords):")
print(f"    X: [{manifest_df['x_min'].min():.1f}, {manifest_df['x_max'].max():.1f}]")
print(f"    Y: [{manifest_df['y_min'].min():.1f}, {manifest_df['y_max'].max():.1f}]")
print(f"    Z: [{manifest_df['z_min'].min():.1f}, {manifest_df['z_max'].max():.1f}]")

# Per-class totals
if 'Antenna_cleaned' in manifest_df.columns:
    print(f"\n  Class totals (cleaned):")
    for cls in CLASS_RGB.values():
        col = f'{cls}_cleaned'
        if col in manifest_df.columns:
            total = manifest_df[col].sum()
            frames_with = (manifest_df[col] > 0).sum()
            print(f"    {cls}: {total:,} points in {frames_with} frames")

print("\n" + "="*70)
print("  OUTPUT FILES")
print("="*70)
print(f"  Cleaned NPZs:     {OUTPUT_DIR}/scene_X/frame_XXX.npz")
print(f"  Manifest:         {manifest_path}")
print(f"  Per-scene CSV:    {METRICS_DIR}/scene_X_metrics.csv")
print(f"  Per-scene JSON:   {METRICS_DIR}/scene_X_summary.json")
print(f"  Combined metrics: {METRICS_DIR}/all_scenes_metrics.csv")
print("="*70)

print("""
  NPZ CONTENTS (per cleaned frame):
  ----------------------------------
  - xyz:                  (N, 3) float32 - cleaned point cloud
  - ego_pose:             (4,) - [x, y, z, yaw] for this frame
  - height_above_ground:  (N,) float32 - HAG for each kept point
  - ground_z:             (N,) float32 - ground Z at each point's XY
  - reflectivity:         (N,) if present in original
  - rgb:                  (N, 3) if present in original
  - kept_indices:         (N,) int32 - indices into ORIGINAL frame
                          (for mapping detections back)

  MANIFEST COLUMNS:
  -----------------
  Frame ID:      scene, frame, frame_idx, original_path, cleaned_path
  Ego pose:      ego_x, ego_y, ego_z, ego_yaw
  Point counts:  num_points_original, num_points_cleaned, points_removed_pct
  Ground stats:  ground_z_min, ground_z_max, ground_z_mean
  HAG stats:     hag_min, hag_max, hag_mean
  Spatial:       x_min, x_max, y_min, y_max, z_min, z_max
  Cleaning:      threshold_m, tile_size_m
  Per-class:     {class}_original, {class}_cleaned for each class
  Noise:         noise_original, noise_cleaned
""")