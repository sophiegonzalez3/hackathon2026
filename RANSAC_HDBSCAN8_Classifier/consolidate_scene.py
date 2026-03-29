"""
consolidate_scene.py — Multi-frame scene consolidation (v2 — memory efficient)
================================================================================
Accumulates all frames into a voxel grid using RUNNING STATISTICS only.
No raw point lists are stored — O(1) memory per voxel regardless of hit count.

Usage:
    python consolidate_scene.py --npz-root processed/ --output-dir consolidated/ --with-labels
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CLASS_RGB = {
    (38, 23, 180):  0,
    (177, 132, 47): 1,
    (129, 81, 97):  2,
    (66, 132, 9):   3,
}
CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']
N_CLASSES = 4
BACKGROUND_ID = -1


# ─────────────────────────────────────────────────────────────
# COORDINATE TRANSFORMS
# ─────────────────────────────────────────────────────────────
def local_to_world(xyz_local, ego_pose):
    """
    Transform local frame → world frame.
    ego_pose: [ego_x(cm), ego_y(cm), ego_z(cm), ego_yaw(degrees)]
    """
    tx = ego_pose[0] / 100.0
    ty = ego_pose[1] / 100.0
    tz = ego_pose[2] / 100.0
    yaw = np.radians(ego_pose[3])  # already degrees

    c, s = np.cos(yaw), np.sin(yaw)
    x_w = xyz_local[:, 0] * c - xyz_local[:, 1] * s + tx
    y_w = xyz_local[:, 0] * s + xyz_local[:, 1] * c + ty
    z_w = xyz_local[:, 2] + tz
    return np.column_stack([x_w, y_w, z_w]).astype(np.float32)


def world_to_local(xyz_world, ego_pose):
    """Inverse of local_to_world."""
    tx = ego_pose[0] / 100.0
    ty = ego_pose[1] / 100.0
    tz = ego_pose[2] / 100.0
    yaw = np.radians(ego_pose[3])

    dx = xyz_world[:, 0] - tx
    dy = xyz_world[:, 1] - ty
    dz = xyz_world[:, 2] - tz

    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.column_stack([dx*c - dy*s, dx*s + dy*c, dz]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# VOXEL GRID — RUNNING STATS ONLY, ~112 bytes per voxel
# ─────────────────────────────────────────────────────────────
# Layout of the float64[14] array per voxel:
_H    = 0   # hit_count
_FC   = 1   # frame_count (updated once per frame via set tracking)
_RS   = 2   # ref_sum
_RSQ  = 3   # ref_sq_sum
_ZMIN = 4   # z_min
_ZMAX = 5   # z_max
_ZS   = 6   # z_sum
_ZSQ  = 7   # z_sq_sum
_EL   = 8   # elev_sum
_V0   = 9   # class_votes[0..3]
_BG   = 13  # bg_count
_NFIELDS = 14


class VoxelGrid:
    """Sparse voxel grid with O(1) memory per voxel."""

    def __init__(self, voxel_size=0.5):
        self.voxel_size = voxel_size
        self.voxels = {}       # (ix,iy,iz) -> float64[14]
        self._frame_sets = {}  # (ix,iy,iz) -> set(frame_ids)

    def _new_voxel(self):
        v = np.zeros(_NFIELDS, dtype=np.float64)
        v[_ZMIN] = 1e30
        v[_ZMAX] = -1e30
        return v

    def accumulate_batch(self, xyz_world, reflectivity, frame_id,
                         class_ids=None, elevation_angles=None):
        """Accumulate one frame into the grid. Groups by voxel first."""
        indices = np.floor(xyz_world / self.voxel_size).astype(np.int32)
        n = len(xyz_world)

        # Group points by voxel using pandas for speed
        # Pack 3 ints into a single int64 key
        keys = (indices[:, 0].astype(np.int64) * 2_000_003 +
                indices[:, 1].astype(np.int64)) * 2_000_003 + indices[:, 2].astype(np.int64)

        order = np.argsort(keys)
        keys_sorted = keys[order]

        # Find group boundaries
        breaks = np.where(np.diff(keys_sorted) != 0)[0] + 1
        groups = np.split(order, breaks)

        ref_f64 = reflectivity.astype(np.float64)
        z_f64 = xyz_world[:, 2].astype(np.float64)

        for grp in groups:
            i0 = grp[0]
            ix, iy, iz = int(indices[i0, 0]), int(indices[i0, 1]), int(indices[i0, 2])
            key = (ix, iy, iz)

            if key not in self.voxels:
                self.voxels[key] = self._new_voxel()
                self._frame_sets[key] = set()

            v = self.voxels[key]
            count = len(grp)

            v[_H] += count
            self._frame_sets[key].add(frame_id)

            r = ref_f64[grp]
            v[_RS] += r.sum()
            v[_RSQ] += (r * r).sum()

            z = z_f64[grp]
            v[_ZMIN] = min(v[_ZMIN], z.min())
            v[_ZMAX] = max(v[_ZMAX], z.max())
            v[_ZS] += z.sum()
            v[_ZSQ] += (z * z).sum()

            if elevation_angles is not None:
                v[_EL] += elevation_angles[grp].sum()

            if class_ids is not None:
                cids = class_ids[grp]
                for c in range(N_CLASSES):
                    v[_V0 + c] += (cids == c).sum()
                v[_BG] += ((cids < 0) | (cids >= N_CLASSES)).sum()

    def to_dataframe(self, with_labels=False):
        """Convert to DataFrame — fully vectorized, no per-voxel loops."""
        nv = len(self.voxels)
        if nv == 0:
            return pd.DataFrame()

        print(f"    Building DataFrame from {nv:,} voxels...", end=" ", flush=True)
        t0 = time.time()

        # Dump everything into arrays
        keys_arr = np.array(list(self.voxels.keys()), dtype=np.int32)   # (nv, 3)
        data = np.array(list(self.voxels.values()), dtype=np.float64)   # (nv, 14)
        fc = np.array([len(self._frame_sets[k]) for k in self.voxels], dtype=np.int32)

        vs = self.voxel_size
        hits = np.maximum(data[:, _H], 1)

        ref_mean = data[:, _RS] / hits
        ref_var = np.maximum(0, data[:, _RSQ] / hits - ref_mean**2)
        z_mean = data[:, _ZS] / hits
        z_var = np.maximum(0, data[:, _ZSQ] / hits - z_mean**2)

        result = {
            'vx': keys_arr[:, 0], 'vy': keys_arr[:, 1], 'vz': keys_arr[:, 2],
            'world_x': (keys_arr[:, 0] + 0.5) * vs,
            'world_y': (keys_arr[:, 1] + 0.5) * vs,
            'world_z': (keys_arr[:, 2] + 0.5) * vs,
            'hit_count': data[:, _H].astype(np.int32),
            'frame_count': fc,
            'observation_ratio': fc / 100.0,
            'ref_mean': ref_mean,
            'ref_std': np.sqrt(ref_var),
            'z_min': data[:, _ZMIN],
            'z_max': data[:, _ZMAX],
            'z_range': data[:, _ZMAX] - data[:, _ZMIN],
            'z_mean': z_mean,
            'z_std': np.sqrt(z_var),
            'elev_mean': data[:, _EL] / hits,
        }

        if with_labels:
            votes = data[:, _V0:_V0+N_CLASSES]
            bg = data[:, _BG]
            total_obs = votes.sum(axis=1)

            class_id = np.full(nv, BACKGROUND_ID, dtype=np.int32)
            class_conf = np.ones(nv, dtype=np.float32)
            is_obs = total_obs > 0

            if is_obs.any():
                class_id[is_obs] = votes[is_obs].argmax(axis=1)
                class_conf[is_obs] = votes[is_obs].max(axis=1) / np.maximum(total_obs[is_obs], 1)

            result['class_id'] = class_id
            result['class_confidence'] = class_conf
            result['is_obstacle'] = is_obs
            for c in range(N_CLASSES):
                result[f'vote_frac_{c}'] = votes[:, c] / hits
            result['bg_frac'] = bg / hits

        df = pd.DataFrame(result)
        print(f"[{time.time()-t0:.1f}s]")
        return df


# ─────────────────────────────────────────────────────────────
# RGB → CLASS ID
# ─────────────────────────────────────────────────────────────
def rgb_to_class_ids(rgb):
    class_ids = np.full(len(rgb), BACKGROUND_ID, dtype=np.int32)
    for (r, g, b), cid in CLASS_RGB.items():
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        class_ids[mask] = cid
    return class_ids


# ─────────────────────────────────────────────────────────────
# CONSOLIDATE ONE SCENE
# ─────────────────────────────────────────────────────────────
def consolidate_scene(scene_dir, voxel_size=0.5, with_labels=False, verbose=True):
    scene_dir = Path(scene_dir)
    npz_files = sorted(scene_dir.glob('frame_*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No frame_*.npz in {scene_dir}")

    if verbose:
        print(f"  Consolidating {len(npz_files)} frames from {scene_dir.name}")

    grid = VoxelGrid(voxel_size=voxel_size)

    for frame_id, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        xyz_local = data['xyz']
        reflectivity = data['reflectivity']
        rgb = data['rgb']
        ego_pose = data['ego_pose']

        xyz_world = local_to_world(xyz_local, ego_pose)
        dist_h = np.sqrt(xyz_local[:, 0]**2 + xyz_local[:, 1]**2)
        elev = np.arctan2(xyz_local[:, 2], dist_h + 1e-8)
        class_ids = rgb_to_class_ids(rgb) if with_labels else None

        grid.accumulate_batch(xyz_world, reflectivity, frame_id,
                              class_ids=class_ids, elevation_angles=elev)

        if verbose and (frame_id + 1) % 20 == 0:
            print(f"    [{frame_id+1}/{len(npz_files)}] "
                  f"{len(grid.voxels):,} voxels")

    if verbose:
        print(f"    Done: {len(grid.voxels):,} voxels")
    return grid


# ─────────────────────────────────────────────────────────────
# PROCESS ALL SCENES
# ─────────────────────────────────────────────────────────────
def process_all_scenes(npz_root, output_dir, voxel_size=0.5,
                       with_labels=False, verbose=True, force=False):
    npz_root = Path(npz_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([d for d in npz_root.iterdir()
                         if d.is_dir() and d.name.startswith('scene_')])
    if not scene_dirs:
        raise FileNotFoundError(f"No scene_* dirs in {npz_root}")

    print(f"Found {len(scene_dirs)} scenes | voxel={voxel_size}m | labels={'Y' if with_labels else 'N'}")

    for scene_dir in scene_dirs:
        out_path = output_dir / f'{scene_dir.name}_voxels.csv'
        if out_path.exists() and not force:
            print(f"  SKIP {scene_dir.name}: {out_path.name} already exists (use --force to overwrite)")
            continue

        t0 = time.time()
        grid = consolidate_scene(scene_dir, voxel_size, with_labels, verbose)
        df = grid.to_dataframe(with_labels=with_labels)
        df['scene'] = scene_dir.name

        out_path = output_dir / f'{scene_dir.name}_voxels.csv'
        df.to_csv(out_path, index=False)

        elapsed = time.time() - t0
        if with_labels and 'is_obstacle' in df.columns:
            print(f"  Saved {scene_dir.name}: {len(df):,} voxels "
                  f"({df['is_obstacle'].sum():,} obstacle) [{elapsed:.1f}s]")
        else:
            print(f"  Saved {scene_dir.name}: {len(df):,} voxels [{elapsed:.1f}s]")

        del grid, df  # free memory between scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-root', default='processed/')
    parser.add_argument('--output-dir', default='consolidated/')
    parser.add_argument('--voxel-size', type=float, default=0.5)
    parser.add_argument('--with-labels', action='store_true')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing consolidated files')
    args = parser.parse_args()
    process_all_scenes(args.npz_root, args.output_dir, args.voxel_size,
                       args.with_labels, verbose=True, force=args.force)


if __name__ == '__main__':
    main()
