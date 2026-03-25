"""
consolidate_scene.py — Multi-frame scene consolidation
=======================================================
For each scene, loads all 100 frames, transforms points from local (ego)
frame to a shared world frame, and accumulates them into a voxel grid.

Each voxel stores rich features:
  - hit_count         : how many points landed here across all frames
  - frame_count       : how many different frames observed this voxel
  - mean_reflectivity : average reflectivity
  - std_reflectivity  : reflectivity variance
  - z_min, z_max      : vertical extent of points in this voxel
  - z_mean, z_std     : vertical statistics
  - mean_elevation    : average elevation angle from ego (indicates "looking up")

For TRAINING only (RGB available):
  - class_votes       : per-class point counts (for majority-vote labeling)

Usage:
    # Training (with labels)
    python consolidate_scene.py --npz-root processed/ --output-dir consolidated/ --with-labels

    # Inference (no labels)
    python consolidate_scene.py --npz-root eval_data/ --output-dir consolidated_eval/
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
# RGB class mapping
CLASS_RGB = {
    (38, 23, 180):  0,  # Antenna
    (177, 132, 47): 1,  # Cable
    (129, 81, 97):  2,  # Electric Pole
    (66, 132, 9):   3,  # Wind Turbine
}
CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']
N_CLASSES = 4
BACKGROUND_ID = -1


# ─────────────────────────────────────────────────────────────
# COORDINATE TRANSFORMS
# ─────────────────────────────────────────────────────────────
def local_to_world(xyz_local, ego_pose):
    """
    Transform points from local (ego/lidar) frame to world frame.

    Args:
        xyz_local: (N, 3) points in local frame (meters, ego at origin)
        ego_pose: (4,) array [ego_x, ego_y, ego_z, ego_yaw] in RAW units
                  (centimeters for position, hundredths of degrees for yaw)

    Returns:
        xyz_world: (N, 3) points in world frame (meters)
    """
    # Convert raw units
    tx = ego_pose[0] / 100.0   # cm -> m
    ty = ego_pose[1] / 100.0
    tz = ego_pose[2] / 100.0
    yaw = np.radians(ego_pose[3])  #already in degree!!

    # Rotation around Z axis
    c, s = np.cos(yaw), np.sin(yaw)

    # Rotate then translate
    x_world = xyz_local[:, 0] * c - xyz_local[:, 1] * s + tx
    y_world = xyz_local[:, 0] * s + xyz_local[:, 1] * c + ty
    z_world = xyz_local[:, 2] + tz

    return np.column_stack([x_world, y_world, z_world]).astype(np.float32)


def world_to_local(xyz_world, ego_pose):
    """
    Transform points from world frame back to local (ego) frame.
    Inverse of local_to_world.
    """
    tx = ego_pose[0] / 100.0
    ty = ego_pose[1] / 100.0
    tz = ego_pose[2] / 100.0
    yaw = np.radians(ego_pose[3] / 100.0)

    # Translate then rotate by -yaw
    dx = xyz_world[:, 0] - tx
    dy = xyz_world[:, 1] - ty
    dz = xyz_world[:, 2] - tz

    c, s = np.cos(-yaw), np.sin(-yaw)
    x_local = dx * c - dy * s
    y_local = dx * s + dy * c
    z_local = dz

    return np.column_stack([x_local, y_local, z_local]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# VOXEL GRID
# ─────────────────────────────────────────────────────────────
class VoxelGrid:
    """
    Sparse voxel grid that accumulates points from multiple frames.

    Uses a dictionary keyed by (ix, iy, iz) voxel indices for memory
    efficiency — only occupied voxels are stored.
    """

    def __init__(self, voxel_size=0.5):
        self.voxel_size = voxel_size
        self.voxels = {}  # (ix, iy, iz) -> VoxelData

    def _to_index(self, xyz):
        """Convert (N, 3) world coords to (N, 3) integer voxel indices."""
        return np.floor(xyz / self.voxel_size).astype(np.int32)

    def accumulate(self, xyz_world, reflectivity, frame_id,
                   class_ids=None, elevation_angles=None):
        """
        Add points from one frame into the voxel grid.

        Args:
            xyz_world: (N, 3) world-frame coordinates
            reflectivity: (N,) uint8 reflectivity values
            frame_id: int, unique frame identifier
            class_ids: (N,) int, per-point class labels (-1=background). Training only.
            elevation_angles: (N,) float, elevation angle from ego. Optional.
        """
        indices = self._to_index(xyz_world)

        for i in range(len(xyz_world)):
            key = (int(indices[i, 0]), int(indices[i, 1]), int(indices[i, 2]))

            if key not in self.voxels:
                self.voxels[key] = {
                    'hit_count': 0,
                    'frame_ids': set(),
                    'ref_sum': 0.0,
                    'ref_sq_sum': 0.0,
                    'z_values': [],
                    'elev_sum': 0.0,
                    'class_votes': np.zeros(N_CLASSES, dtype=np.int32),
                    'bg_count': 0,
                }

            v = self.voxels[key]
            v['hit_count'] += 1
            v['frame_ids'].add(frame_id)

            ref_val = float(reflectivity[i])
            v['ref_sum'] += ref_val
            v['ref_sq_sum'] += ref_val * ref_val

            v['z_values'].append(float(xyz_world[i, 2]))

            if elevation_angles is not None:
                v['elev_sum'] += float(elevation_angles[i])

            if class_ids is not None:
                cid = int(class_ids[i])
                if 0 <= cid < N_CLASSES:
                    v['class_votes'][cid] += 1
                else:
                    v['bg_count'] += 1

    def accumulate_batch(self, xyz_world, reflectivity, frame_id,
                         class_ids=None, elevation_angles=None):
        """
        Vectorized batch accumulation — much faster than per-point loop.
        Groups points by voxel index first, then updates each voxel once.
        """
        indices = self._to_index(xyz_world)
        n = len(xyz_world)

        # Build a dict of point indices per voxel
        voxel_groups = defaultdict(list)
        for i in range(n):
            key = (int(indices[i, 0]), int(indices[i, 1]), int(indices[i, 2]))
            voxel_groups[key].append(i)

        for key, pt_indices in voxel_groups.items():
            pt_indices = np.array(pt_indices)

            if key not in self.voxels:
                self.voxels[key] = {
                    'hit_count': 0,
                    'frame_ids': set(),
                    'ref_sum': 0.0,
                    'ref_sq_sum': 0.0,
                    'z_values': [],
                    'elev_sum': 0.0,
                    'class_votes': np.zeros(N_CLASSES, dtype=np.int32),
                    'bg_count': 0,
                }

            v = self.voxels[key]
            count = len(pt_indices)
            v['hit_count'] += count
            v['frame_ids'].add(frame_id)

            refs = reflectivity[pt_indices].astype(np.float64)
            v['ref_sum'] += refs.sum()
            v['ref_sq_sum'] += (refs * refs).sum()

            zs = xyz_world[pt_indices, 2].tolist()
            v['z_values'].extend(zs)

            if elevation_angles is not None:
                v['elev_sum'] += float(elevation_angles[pt_indices].sum())

            if class_ids is not None:
                cids = class_ids[pt_indices]
                for c in range(N_CLASSES):
                    v['class_votes'][c] += int((cids == c).sum())
                v['bg_count'] += int((cids < 0).sum() + (cids >= N_CLASSES).sum())

    def to_dataframe(self, with_labels=False):
        """
        Convert the sparse voxel grid to a DataFrame with computed features.

        Returns one row per occupied voxel.
        """
        rows = []
        for (ix, iy, iz), v in self.voxels.items():
            n = v['hit_count']
            if n == 0:
                continue

            zs = np.array(v['z_values'])
            ref_mean = v['ref_sum'] / n
            ref_var = max(0, v['ref_sq_sum'] / n - ref_mean ** 2)

            row = {
                'vx': ix, 'vy': iy, 'vz': iz,
                # World position (voxel center)
                'world_x': (ix + 0.5) * self.voxel_size,
                'world_y': (iy + 0.5) * self.voxel_size,
                'world_z': (iz + 0.5) * self.voxel_size,
                # Accumulation features
                'hit_count': n,
                'frame_count': len(v['frame_ids']),
                'observation_ratio': len(v['frame_ids']) / 100.0,  # fraction of frames
                # Reflectivity
                'ref_mean': ref_mean,
                'ref_std': np.sqrt(ref_var),
                # Vertical stats
                'z_min': zs.min(),
                'z_max': zs.max(),
                'z_range': zs.max() - zs.min(),
                'z_mean': zs.mean(),
                'z_std': zs.std() if len(zs) > 1 else 0.0,
                # Elevation
                'elev_mean': v['elev_sum'] / n if n > 0 else 0.0,
            }

            if with_labels:
                votes = v['class_votes']
                total_obstacle = votes.sum()
                bg = v['bg_count']

                if total_obstacle > 0:
                    row['class_id'] = int(np.argmax(votes))
                    row['class_confidence'] = float(votes.max()) / float(total_obstacle)
                    row['is_obstacle'] = True
                else:
                    row['class_id'] = BACKGROUND_ID
                    row['class_confidence'] = 1.0
                    row['is_obstacle'] = False

                # Store raw vote fractions
                for c in range(N_CLASSES):
                    row[f'vote_frac_{c}'] = float(votes[c]) / max(n, 1)
                row['bg_frac'] = float(bg) / max(n, 1)

            rows.append(row)

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# RGB → CLASS ID
# ─────────────────────────────────────────────────────────────
def rgb_to_class_ids(rgb):
    """Convert (N, 3) RGB array to (N,) class IDs. -1 = background."""
    class_ids = np.full(len(rgb), BACKGROUND_ID, dtype=np.int32)
    for (r, g, b), cid in CLASS_RGB.items():
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        class_ids[mask] = cid
    return class_ids


# ─────────────────────────────────────────────────────────────
# CONSOLIDATE ONE SCENE
# ─────────────────────────────────────────────────────────────
def consolidate_scene(scene_dir, voxel_size=0.5, with_labels=False, verbose=True):
    """
    Load all frames in a scene directory, transform to world frame,
    and accumulate into a VoxelGrid.

    Args:
        scene_dir: path to directory containing frame_*.npz
        voxel_size: voxel side length in meters
        with_labels: if True, use RGB to assign class labels to voxels
        verbose: print progress

    Returns:
        VoxelGrid object
    """
    scene_dir = Path(scene_dir)
    npz_files = sorted(scene_dir.glob('frame_*.npz'))

    if not npz_files:
        raise FileNotFoundError(f"No frame_*.npz files in {scene_dir}")

    if verbose:
        print(f"  Consolidating {len(npz_files)} frames from {scene_dir.name}")

    grid = VoxelGrid(voxel_size=voxel_size)

    for frame_id, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        xyz_local = data['xyz']         # (N, 3) local frame, meters
        reflectivity = data['reflectivity']  # (N,) uint8
        rgb = data['rgb']               # (N, 3) uint8
        ego_pose = data['ego_pose']     # (4,) raw units

        # Transform to world frame
        xyz_world = local_to_world(xyz_local, ego_pose)

        # Compute elevation angle (from ego, useful feature)
        dist_horizontal = np.sqrt(xyz_local[:, 0]**2 + xyz_local[:, 1]**2)
        elevation_angles = np.arctan2(xyz_local[:, 2], dist_horizontal + 1e-8)

        # Class labels (training only)
        class_ids = rgb_to_class_ids(rgb) if with_labels else None

        # Accumulate
        grid.accumulate_batch(
            xyz_world, reflectivity, frame_id,
            class_ids=class_ids,
            elevation_angles=elevation_angles,
        )

        if verbose and (frame_id + 1) % 20 == 0:
            print(f"    [{frame_id + 1}/{len(npz_files)}] "
                  f"{len(grid.voxels):,} voxels so far")

    if verbose:
        print(f"    Done: {len(grid.voxels):,} occupied voxels")

    return grid


# ─────────────────────────────────────────────────────────────
# PROCESS ALL SCENES
# ─────────────────────────────────────────────────────────────
def process_all_scenes(npz_root, output_dir, voxel_size=0.5,
                       with_labels=False, verbose=True):
    """
    Consolidate all scenes found under npz_root.

    Saves per-scene voxel grids as .parquet files.
    """
    npz_root = Path(npz_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find scene directories
    scene_dirs = sorted([d for d in npz_root.iterdir()
                         if d.is_dir() and d.name.startswith('scene_')])

    if not scene_dirs:
        raise FileNotFoundError(f"No scene_* directories in {npz_root}")

    print(f"Found {len(scene_dirs)} scenes in {npz_root}")
    print(f"Voxel size: {voxel_size}m")
    print(f"Labels: {'YES' if with_labels else 'NO'}")

    all_scene_stats = []

    for scene_dir in scene_dirs:
        t0 = time.time()
        scene_name = scene_dir.name

        grid = consolidate_scene(
            scene_dir, voxel_size=voxel_size,
            with_labels=with_labels, verbose=verbose,
        )

        # Convert to DataFrame
        df = grid.to_dataframe(with_labels=with_labels)
        df['scene'] = scene_name

        # Save
        out_path = output_dir / f'{scene_name}_voxels.csv'
        df.to_csv(out_path, index=False)

        elapsed = time.time() - t0

        # Stats
        n_voxels = len(df)
        if with_labels and 'is_obstacle' in df.columns:
            n_obstacle = df['is_obstacle'].sum()
            n_bg = (~df['is_obstacle']).sum()
            print(f"  {scene_name}: {n_voxels:,} voxels "
                  f"({n_obstacle:,} obstacle, {n_bg:,} background) [{elapsed:.1f}s]")

            # Per-class breakdown
            obstacle_df = df[df['is_obstacle']]
            if len(obstacle_df) > 0:
                for cid, name in enumerate(CLASS_NAMES):
                    count = (obstacle_df['class_id'] == cid).sum()
                    print(f"    {name}: {count:,} voxels")
        else:
            print(f"  {scene_name}: {n_voxels:,} voxels [{elapsed:.1f}s]")

        all_scene_stats.append({
            'scene': scene_name,
            'n_voxels': n_voxels,
            'output': str(out_path),
        })

        # Also save the VoxelGrid object for later use (back-projection)
        grid_path = output_dir / f'{scene_name}_grid.npz'
        _save_grid_compact(grid, grid_path)

    # Summary
    stats_df = pd.DataFrame(all_scene_stats)
    stats_df.to_csv(output_dir / 'consolidation_stats.csv', index=False)
    print(f"\nTotal: {stats_df['n_voxels'].sum():,} voxels across {len(scene_dirs)} scenes")

    return stats_df


def _save_grid_compact(grid, path):
    """Save VoxelGrid indices and voxel_size for back-projection."""
    keys = np.array(list(grid.voxels.keys()), dtype=np.int32)  # (M, 3)
    np.savez_compressed(
        path,
        voxel_indices=keys,
        voxel_size=np.array([grid.voxel_size]),
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Consolidate multi-frame scenes into voxel grids')
    parser.add_argument('--npz-root', type=str, default='processed/',
                        help='Root directory with scene_*/frame_*.npz')
    parser.add_argument('--output-dir', type=str, default='consolidated/',
                        help='Output directory for voxel grids')
    parser.add_argument('--voxel-size', type=float, default=0.5,
                        help='Voxel side length in meters (default: 0.5)')
    parser.add_argument('--with-labels', action='store_true',
                        help='Use RGB labels for training data')
    args = parser.parse_args()

    process_all_scenes(
        args.npz_root, args.output_dir,
        voxel_size=args.voxel_size,
        with_labels=args.with_labels,
        verbose=True,
    )


if __name__ == '__main__':
    main()
