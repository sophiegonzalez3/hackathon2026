# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Ground Model Coverage Analysis
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LOCAL_ROOT = Path("/content/hackathon")
PROCESSED_DIR = LOCAL_ROOT / "processed"
TEST_SCENE = "scene_3"

scene_path = PROCESSED_DIR / TEST_SCENE
npz_files = sorted(scene_path.glob('frame_*.npz'))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Check ego positions for ALL frames
# ══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("  EGO POSITIONS FOR ALL FRAMES")
print("="*80)

ego_positions = []
for npz_path in npz_files:
    data = np.load(npz_path)
    ego_pose = data['ego_pose']
    ego_positions.append({
        'frame': npz_path.stem,
        'frame_idx': int(npz_path.stem.split('_')[1]),
        'ego_x': ego_pose[0],
        'ego_y': ego_pose[1],
        'ego_z': ego_pose[2],
        'ego_yaw': ego_pose[3],
    })

ego_df = pd.DataFrame(ego_positions)

print(f"\nEgo X range: {ego_df['ego_x'].min():.1f} to {ego_df['ego_x'].max():.1f}")
print(f"Ego Y range: {ego_df['ego_y'].min():.1f} to {ego_df['ego_y'].max():.1f}")
print(f"Ego Z range: {ego_df['ego_z'].min():.1f} to {ego_df['ego_z'].max():.1f}")

# Check for suspicious Z values
print(f"\n  Frames with unusual ego_z:")
suspicious = ego_df[ego_df['ego_z'].abs() > 100]  # Z > 100m is unusual
if len(suspicious) > 0:
    print(suspicious[['frame', 'ego_x', 'ego_y', 'ego_z']].head(20).to_string())
else:
    print("  None found")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Visualize ego trajectory
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# XY trajectory
ax = axes[0]
scatter = ax.scatter(ego_df['ego_x'], ego_df['ego_y'], c=ego_df['frame_idx'],
                     cmap='viridis', s=20)
plt.colorbar(scatter, ax=ax, label='Frame index')
ax.set_xlabel('Ego X')
ax.set_ylabel('Ego Y')
ax.set_title('Ego trajectory (XY)')
ax.set_aspect('equal')

# Highlight problematic frames (80-99)
problem_frames = ego_df[ego_df['frame_idx'] >= 80]
ax.scatter(problem_frames['ego_x'], problem_frames['ego_y'],
           c='red', s=50, marker='x', label='Frames 80-99 (NaN issues)')
ax.legend()

# Z over time
ax = axes[1]
ax.plot(ego_df['frame_idx'], ego_df['ego_z'], marker='o', markersize=3)
ax.axhline(y=100, color='red', linestyle='--', label='Z=100m')
ax.axhline(y=0, color='green', linestyle='--', label='Z=0m')
ax.set_xlabel('Frame index')
ax.set_ylabel('Ego Z')
ax.set_title('Ego Z over frames')
ax.legend()

# XY + Z as color
ax = axes[2]
scatter = ax.scatter(ego_df['ego_x'], ego_df['ego_y'], c=ego_df['ego_z'],
                     cmap='coolwarm', s=20)
plt.colorbar(scatter, ax=ax, label='Ego Z')
ax.set_xlabel('Ego X')
ax.set_ylabel('Ego Y')
ax.set_title('Ego trajectory colored by Z')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/content/ego_trajectory_debug.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Check ground model coverage vs frame locations
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  GROUND MODEL COVERAGE CHECK")
print("="*80)

# Sample a few frames from different parts of the trajectory
test_frames = [0, 25, 50, 75, 85, 95]

for frame_idx in test_frames:
    frame_name = f"frame_{frame_idx:03d}"
    npz_path = scene_path / f"{frame_name}.npz"

    if not npz_path.exists():
        print(f"\n  {frame_name}: FILE NOT FOUND")
        continue

    data = np.load(npz_path)
    xyz_local = data['xyz']
    ego_pose = data['ego_pose']

    xyz_world = local_to_world(xyz_local, ego_pose)
    ground_z = ground_fn(xyz_world[:, 0], xyz_world[:, 1])
    hag = xyz_world[:, 2] - ground_z

    n_nan = np.isnan(hag).sum()
    n_total = len(hag)

    print(f"\n  {frame_name}:")
    print(f"    Ego: ({ego_pose[0]:.1f}, {ego_pose[1]:.1f}, {ego_pose[2]:.1f})")
    print(f"    Points: {n_total:,}")
    print(f"    NaN HAG: {n_nan:,} ({100*n_nan/n_total:.1f}%)")

    if n_nan < n_total:  # Some valid values
        valid_hag = hag[~np.isnan(hag)]
        print(f"    Valid HAG range: {valid_hag.min():.1f}m to {valid_hag.max():.1f}m")

    # Check if ground_z is sensible
    valid_ground = ground_z[~np.isnan(ground_z)]
    if len(valid_ground) > 0:
        print(f"    Ground Z range: {valid_ground.min():.1f}m to {valid_ground.max():.1f}m")
        print(f"    Point Z range: {xyz_world[:, 2].min():.1f}m to {xyz_world[:, 2].max():.1f}m")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Check the ground model grid itself
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  GROUND MODEL GRID ANALYSIS")
print("="*80)

# Access the ground_info from when we built the model
# (You may need to rebuild with this modification to get the info)

# Let's check by querying the ground model at ego positions
for frame_idx in test_frames:
    row = ego_df[ego_df['frame_idx'] == frame_idx].iloc[0]
    ego_x, ego_y = row['ego_x'], row['ego_y']

    ground_at_ego = ground_fn(np.array([ego_x]), np.array([ego_y]))[0]

    print(f"  Frame {frame_idx:03d}: Ego=({ego_x:.1f}, {ego_y:.1f}), Ground model returns: {ground_at_ego}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Rebuild ground model and check its bounds
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  REBUILDING GROUND MODEL WITH DEBUG INFO")
print("="*80)

def build_ground_model_debug(scene_dir, tile_size=10.0, max_frames=None):
    """Build ground model and return debug info."""
    scene_dir = Path(scene_dir)
    npz_files = sorted(scene_dir.glob('frame_*.npz'))

    if max_frames:
        npz_files = npz_files[:max_frames]

    print(f"  Using {len(npz_files)} frames")

    # Collect ALL world points to find true bounds
    all_x, all_y, all_z = [], [], []
    frame_bounds = []

    for npz_path in npz_files:
        data = np.load(npz_path)
        xyz_world = local_to_world(data['xyz'], data['ego_pose'])

        all_x.append(xyz_world[:, 0])
        all_y.append(xyz_world[:, 1])
        all_z.append(xyz_world[:, 2])

        frame_bounds.append({
            'frame': npz_path.stem,
            'x_min': xyz_world[:, 0].min(),
            'x_max': xyz_world[:, 0].max(),
            'y_min': xyz_world[:, 1].min(),
            'y_max': xyz_world[:, 1].max(),
            'z_min': xyz_world[:, 2].min(),
            'z_max': xyz_world[:, 2].max(),
        })

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    print(f"\n  World coordinate ranges (from {len(npz_files)} frames):")
    print(f"    X: {all_x.min():.1f} to {all_x.max():.1f} (span: {all_x.max()-all_x.min():.1f}m)")
    print(f"    Y: {all_y.min():.1f} to {all_y.max():.1f} (span: {all_y.max()-all_y.min():.1f}m)")
    print(f"    Z: {all_z.min():.1f} to {all_z.max():.1f} (span: {all_z.max()-all_z.min():.1f}m)")

    return pd.DataFrame(frame_bounds)

frame_bounds_df = build_ground_model_debug(scene_path, max_frames=None)

# Check if there's a gap in coverage
print("\n  Frame Z ranges (looking for outliers):")
print(frame_bounds_df[['frame', 'z_min', 'z_max']].sort_values('z_min').head(10))
print("  ...")
print(frame_bounds_df[['frame', 'z_min', 'z_max']].sort_values('z_min').tail(10))