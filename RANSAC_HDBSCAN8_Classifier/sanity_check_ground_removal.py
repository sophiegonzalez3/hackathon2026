# ══════════════════════════════════════════════════════════════════════════════
# FIND FRAMES WITH EACH CLASS & ANALYZE ACROSS SCENE
# ══════════════════════════════════════════════════════════════════════════════
CLASS_RGB = {
    (38, 23, 180):  'Antenna',
    (177, 132, 47): 'Cable',
    (129, 81, 97):  'Electric Pole',
    (66, 132, 9):   'Wind Turbine',
}

scene_path = PROCESSED_DIR / TEST_SCENE
npz_files = sorted(scene_path.glob('frame_*.npz'))

print(f"Scanning {len(npz_files)} frames in {TEST_SCENE}...\n")

# Track which frames have which classes
frames_with_class = {cls: [] for cls in CLASS_RGB.values()}
class_totals = {cls: {'total': 0, 'survived_05': 0, 'survived_30': 0, 'survived_50': 0, 'survived_10': 0, 'survived_15': 0} for cls in CLASS_RGB.values()}

for frame_idx, npz_path in enumerate(npz_files):
    data = np.load(npz_path)
    xyz_local = data['xyz']
    ego_pose = data['ego_pose']
    rgb = data.get('rgb', None)

    if rgb is None:
        continue

    # Transform and compute height above ground
    xyz_world = local_to_world(xyz_local, ego_pose)
    ground_z = ground_fn(xyz_world[:, 0], xyz_world[:, 1])
    hag = xyz_world[:, 2] - ground_z

    for (r, g, b), cls in CLASS_RGB.items():
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        n_pts = mask.sum()

        if n_pts > 0:
            frames_with_class[cls].append((frame_idx, n_pts))
            class_totals[cls]['total'] += n_pts
            class_totals[cls]['survived_05'] += (mask & (hag > 0.5)).sum()
            class_totals[cls]['survived_30'] += (mask & (hag > 3.0)).sum()
            class_totals[cls]['survived_50'] += (mask & (hag > 5.0)).sum()
            class_totals[cls]['survived_10'] += (mask & (hag > 10.0)).sum()
            class_totals[cls]['survived_15'] += (mask & (hag > 15.0)).sum()

# Print summary
print("="*70)
print("  FRAMES WITH EACH CLASS")
print("="*70)
for cls, frames in frames_with_class.items():
    if frames:
        frame_nums = [f[0] for f in frames[:10]]  # Show first 10
        total_pts = sum(f[1] for f in frames)
        print(f"\n  {cls}:")
        print(f"    Found in {len(frames)} frames, {total_pts:,} total points")
        print(f"    Example frames: {frame_nums}")
    else:
        print(f"\n  {cls}: NOT FOUND in any frame!")

# Print survival rates across ALL frames
print("\n" + "="*70)
print("  SURVIVAL RATES ACROSS ENTIRE SCENE")
print("="*70)
print(f"{'Class':<15} {'Total Pts':>12} {'@0.5m':>10} {'@3.0m':>10}")
print("-"*50)

for cls, counts in class_totals.items():
    if counts['total'] > 0:
        pct_05 = 100 * counts['survived_05'] / counts['total']
        pct_30 = 100 * counts['survived_30'] / counts['total']
        pct_50 = 100 * counts['survived_50'] / counts['total']
        pct_100 = 100 * counts['survived_10'] / counts['total']
        pct_150 = 100 * counts['survived_15'] / counts['total']
        print(f"{cls:<15} {counts['total']:>12,} {pct_05:>9.1f}% {pct_30:>9.1f}% {pct_50:>9.1f}% {pct_100:>9.1f}% {pct_150:>9.1f}%")
    else:
        print(f"{cls:<15} {'N/A':>12} {'N/A':>10} {'N/A':>10}")





# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Find frames with worst survival rates
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Config
LOCAL_ROOT = Path("/content/hackathon")
PROCESSED_DIR = LOCAL_ROOT / "processed"
TEST_SCENE = "scene_3"  # Change this
TEST_THRESHOLD = 15.0   # The threshold you're investigating

CLASS_RGB = {
    (38, 23, 180):  'Antenna',
    (177, 132, 47): 'Cable',
    (129, 81, 97):  'Electric Pole',
    (66, 132, 9):   'Wind Turbine',
}

# ══════════════════════════════════════════════════════════════════════════════
# Reuse ground model from before (make sure you've run the previous cells)
# ══════════════════════════════════════════════════════════════════════════════

def local_to_world(xyz_local, ego_pose):
    x_ego, y_ego, z_ego, yaw_deg = ego_pose
    yaw_rad = np.radians(yaw_deg)
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0,                0,               1]
    ])
    return xyz_local @ R.T + np.array([x_ego, y_ego, z_ego])

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Compute per-frame survival rates
# ══════════════════════════════════════════════════════════════════════════════

scene_path = PROCESSED_DIR / TEST_SCENE
npz_files = sorted(scene_path.glob('frame_*.npz'))

print(f"Analyzing {len(npz_files)} frames in {TEST_SCENE}...")
print(f"Testing threshold: {TEST_THRESHOLD}m\n")

frame_stats = []

for npz_path in npz_files:
    data = np.load(npz_path)
    xyz_local = data['xyz']
    ego_pose = data['ego_pose']
    rgb = data.get('rgb', None)

    if rgb is None:
        continue

    # Compute height above ground
    xyz_world = local_to_world(xyz_local, ego_pose)
    ground_z = ground_fn(xyz_world[:, 0], xyz_world[:, 1])
    hag = xyz_world[:, 2] - ground_z

    frame_stat = {
        'frame': npz_path.stem,
        'frame_idx': int(npz_path.stem.split('_')[1]),
        'ego_x': ego_pose[0],
        'ego_y': ego_pose[1],
        'ego_z': ego_pose[2],
    }

    # Per-class stats
    for (r, g, b), cls in CLASS_RGB.items():
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        n_total = mask.sum()

        if n_total > 0:
            class_hag = hag[mask]
            n_survived = (class_hag > TEST_THRESHOLD).sum()
            survival_pct = 100 * n_survived / n_total

            frame_stat[f'{cls}_total'] = n_total
            frame_stat[f'{cls}_survived'] = n_survived
            frame_stat[f'{cls}_survival_pct'] = survival_pct
            frame_stat[f'{cls}_hag_min'] = class_hag.min()
            frame_stat[f'{cls}_hag_max'] = class_hag.max()
            frame_stat[f'{cls}_hag_median'] = np.median(class_hag)
            frame_stat[f'{cls}_hag_mean'] = class_hag.mean()
        else:
            frame_stat[f'{cls}_total'] = 0

    frame_stats.append(frame_stat)

df = pd.DataFrame(frame_stats)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Find worst frames per class
# ══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("  WORST FRAMES PER CLASS (lowest survival at threshold={:.0f}m)".format(TEST_THRESHOLD))
print("="*80)

for cls in ['Antenna', 'Electric Pole', 'Wind Turbine', 'Cable']:
    col_total = f'{cls}_total'
    col_surv = f'{cls}_survival_pct'
    col_hag_min = f'{cls}_hag_min'
    col_hag_median = f'{cls}_hag_median'

    # Filter frames that have this class
    has_class = df[col_total] > 0
    if not has_class.any():
        print(f"\n  {cls}: NOT PRESENT in any frame")
        continue

    subset = df[has_class].copy()
    subset_sorted = subset.sort_values(col_surv)

    print(f"\n  {cls}: {len(subset)} frames with this class")
    print(f"  {'Frame':<12} {'Total':>8} {'Survived':>10} {'Surv%':>8} {'HAG min':>10} {'HAG med':>10}")
    print("  " + "-"*65)

    # Show 10 worst frames
    for _, row in subset_sorted.head(10).iterrows():
        print(f"  {row['frame']:<12} {row[col_total]:>8.0f} {row[f'{cls}_survived']:>10.0f} "
              f"{row[col_surv]:>7.1f}% {row[col_hag_min]:>9.1f}m {row[col_hag_median]:>9.1f}m")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Detailed analysis of THE WORST frame
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  DETAILED ANALYSIS OF WORST FRAMES")
print("="*80)

def analyze_worst_frame(cls, n_worst=1):
    """Deep dive into the worst frame for a class."""
    col_total = f'{cls}_total'
    col_surv = f'{cls}_survival_pct'

    has_class = df[col_total] > 0
    if not has_class.any():
        return

    subset = df[has_class].sort_values(col_surv)
    worst_frames = subset.head(n_worst)

    for _, row in worst_frames.iterrows():
        frame_name = row['frame']
        print(f"\n  === {cls}: {frame_name} (survival={row[col_surv]:.1f}%) ===")

        # Load the frame
        npz_path = scene_path / f"{frame_name}.npz"
        data = np.load(npz_path)
        xyz_local = data['xyz']
        ego_pose = data['ego_pose']
        rgb = data['rgb']

        xyz_world = local_to_world(xyz_local, ego_pose)
        ground_z = ground_fn(xyz_world[:, 0], xyz_world[:, 1])
        hag = xyz_world[:, 2] - ground_z

        # Get class points
        r, g, b = [k for k, v in CLASS_RGB.items() if v == cls][0]
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)

        class_xyz_world = xyz_world[mask]
        class_hag = hag[mask]
        class_ground_z = ground_z[mask]
        class_z = class_xyz_world[:, 2]

        print(f"    Total {cls} points: {mask.sum()}")
        print(f"    Ego position: ({ego_pose[0]:.1f}, {ego_pose[1]:.1f}, {ego_pose[2]:.1f})")
        print(f"")
        print(f"    Raw Z (world):    min={class_z.min():.2f}, max={class_z.max():.2f}, median={np.median(class_z):.2f}")
        print(f"    Ground Z:         min={class_ground_z.min():.2f}, max={class_ground_z.max():.2f}, median={np.median(class_ground_z):.2f}")
        print(f"    Height above gnd: min={class_hag.min():.2f}, max={class_hag.max():.2f}, median={np.median(class_hag):.2f}")
        print(f"")
        print(f"    HAG distribution:")
        for pct in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(class_hag, pct)
            print(f"      {pct:>3}th percentile: {val:>8.2f}m")

        # How many would survive at different thresholds
        print(f"\n    Survival at different thresholds:")
        for t in [2, 5, 10, 15, 20, 25]:
            surv = (class_hag > t).sum()
            pct = 100 * surv / len(class_hag)
            bar = "█" * int(pct // 5)
            print(f"      {t:>3}m: {surv:>6} / {len(class_hag)} ({pct:>5.1f}%) {bar}")

        # VISUALIZATION
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{cls} in {frame_name} - Why survival is only {row[col_surv]:.1f}%?', fontsize=14)

        # 1. HAG histogram
        ax = axes[0, 0]
        ax.hist(class_hag, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(TEST_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={TEST_THRESHOLD}m')
        ax.axvline(np.median(class_hag), color='green', linestyle='--', label=f'Median={np.median(class_hag):.1f}m')
        ax.set_xlabel('Height above ground (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{cls} HAG distribution')
        ax.legend()

        # 2. Raw Z vs Ground Z
        ax = axes[0, 1]
        ax.scatter(class_ground_z, class_z, s=1, alpha=0.5)
        ax.plot([class_ground_z.min(), class_ground_z.max()],
                [class_ground_z.min(), class_ground_z.max()], 'r--', label='Z = Ground (HAG=0)')
        ax.plot([class_ground_z.min(), class_ground_z.max()],
                [class_ground_z.min() + TEST_THRESHOLD, class_ground_z.max() + TEST_THRESHOLD],
                'g--', label=f'Z = Ground + {TEST_THRESHOLD}m')
        ax.set_xlabel('Ground Z (from model)')
        ax.set_ylabel('Point Z (raw)')
        ax.set_title('Point Z vs Ground Z')
        ax.legend()

        # 3. Side view: all points
        ax = axes[0, 2]
        ax.scatter(xyz_world[:, 0], xyz_world[:, 2], c='gray', s=0.1, alpha=0.2, label='All points')
        ax.scatter(class_xyz_world[:, 0], class_xyz_world[:, 2], c='red', s=2, alpha=0.8, label=cls)
        ax.set_xlabel('X (world)')
        ax.set_ylabel('Z (world)')
        ax.set_title(f'Side view - {cls} in red')
        ax.legend()

        # 4. BEV: class points colored by HAG
        ax = axes[1, 0]
        scatter = ax.scatter(class_xyz_world[:, 0], class_xyz_world[:, 1],
                             c=class_hag, cmap='RdYlGn', s=5, vmin=0, vmax=30)
        plt.colorbar(scatter, ax=ax, label='HAG (m)')
        ax.set_xlabel('X (world)')
        ax.set_ylabel('Y (world)')
        ax.set_title(f'{cls} colored by HAG (green=high, red=low)')
        ax.set_aspect('equal')

        # 5. BEV: survived vs killed
        ax = axes[1, 1]
        killed = class_hag <= TEST_THRESHOLD
        survived = class_hag > TEST_THRESHOLD
        ax.scatter(class_xyz_world[killed, 0], class_xyz_world[killed, 1],
                   c='red', s=5, alpha=0.7, label=f'Killed ({killed.sum()})')
        ax.scatter(class_xyz_world[survived, 0], class_xyz_world[survived, 1],
                   c='blue', s=5, alpha=0.7, label=f'Survived ({survived.sum()})')
        ax.set_xlabel('X (world)')
        ax.set_ylabel('Y (world)')
        ax.set_title(f'{cls}: Killed (red) vs Survived (blue) at {TEST_THRESHOLD}m')
        ax.set_aspect('equal')
        ax.legend()

        # 6. Ground model in this area
        ax = axes[1, 2]
        x_min, x_max = class_xyz_world[:, 0].min() - 50, class_xyz_world[:, 0].max() + 50
        y_min, y_max = class_xyz_world[:, 1].min() - 50, class_xyz_world[:, 1].max() + 50

        # Sample ground model
        xx = np.linspace(x_min, x_max, 100)
        yy = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = ground_fn(XX.flatten(), YY.flatten()).reshape(XX.shape)

        im = ax.contourf(XX, YY, ZZ, levels=20, cmap='terrain')
        plt.colorbar(im, ax=ax, label='Ground Z')
        ax.scatter(class_xyz_world[:, 0], class_xyz_world[:, 1], c='red', s=10, label=cls)
        ax.set_xlabel('X (world)')
        ax.set_ylabel('Y (world)')
        ax.set_title('Ground model around object')
        ax.set_aspect('equal')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'/content/debug_{cls}_{frame_name}.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n    ✓ Saved debug plot to /content/debug_{cls}_{frame_name}.png")

# Analyze worst frame for each class
for cls in ['Antenna', 'Electric Pole', 'Wind Turbine']:
    analyze_worst_frame(cls, n_worst=1)