import numpy as np
import matplotlib.pyplot as plt
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

def build_ground_model_FAST(scene_dir, tile_size=10.0, percentile=5, max_frames=20):
    """FAST version - uses vectorized binning instead of slow loops."""
    scene_dir = Path(scene_dir)
    npz_files = sorted(scene_dir.glob('frame_*.npz'))[:max_frames]

    print(f"Building ground model from {len(npz_files)} frames...")

    # First pass: find bounds (sample a few frames)
    sample_pts = []
    for npz_path in npz_files[::5]:  # Every 5th frame
        data = np.load(npz_path)
        xyz_world = local_to_world(data['xyz'], data['ego_pose'])
        sample_pts.append(xyz_world[::10])  # Subsample
    sample_pts = np.vstack(sample_pts)

    x_min, x_max = sample_pts[:, 0].min() - tile_size, sample_pts[:, 0].max() + tile_size
    y_min, y_max = sample_pts[:, 1].min() - tile_size, sample_pts[:, 1].max() + tile_size

    n_tiles_x = int(np.ceil((x_max - x_min) / tile_size))
    n_tiles_y = int(np.ceil((y_max - y_min) / tile_size))

    print(f"  Grid: {n_tiles_x} x {n_tiles_y} tiles")

    # Accumulate min Z per tile using histogram approach
    z_min_grid = np.full((n_tiles_x, n_tiles_y), np.inf)
    z_percentile_accum = [[[] for _ in range(n_tiles_y)] for _ in range(n_tiles_x)]

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        xyz_world = local_to_world(data['xyz'], data['ego_pose'])

        # Compute tile indices (vectorized)
        xi = np.clip(((xyz_world[:, 0] - x_min) / tile_size).astype(int), 0, n_tiles_x - 1)
        yi = np.clip(((xyz_world[:, 1] - y_min) / tile_size).astype(int), 0, n_tiles_y - 1)

        # Use np.minimum.at for fast accumulation
        np.minimum.at(z_min_grid, (xi, yi), xyz_world[:, 2])

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(npz_files)} frames")

    # Replace inf with NaN, then fill with neighbors
    z_min_grid[z_min_grid == np.inf] = np.nan

    # Fill NaN with local median (simple approach)
    valid_mask = ~np.isnan(z_min_grid)
    if not valid_mask.all():
        # Use scipy's generic_filter to fill NaN with neighbor median
        from scipy.ndimage import generic_filter
        def nanmedian(x):
            return np.nanmedian(x) if np.any(~np.isnan(x)) else np.nan

        filled = generic_filter(z_min_grid, nanmedian, size=3, mode='nearest')
        z_min_grid = np.where(valid_mask, z_min_grid, filled)

    # Build simple lookup function
    def ground_fn(x_query, y_query):
        x_query = np.asarray(x_query)
        y_query = np.asarray(y_query)
        xi = np.clip(((x_query - x_min) / tile_size).astype(int), 0, n_tiles_x - 1)
        yi = np.clip(((y_query - y_min) / tile_size).astype(int), 0, n_tiles_y - 1)
        return z_min_grid[xi, yi]

    print(f"✓ Ground model ready!")
    return ground_fn, z_min_grid, (x_min, y_min, tile_size)