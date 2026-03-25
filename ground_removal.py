"""
ground_removal.py — RANSAC-based ground plane removal
======================================================
Separates ground points from non-ground points WITHOUT using RGB labels.
This is critical because eval data has no class labels.

Two strategies:
  1. Global RANSAC on the full consolidated scene (simple, fast)
  2. Tile-based RANSAC for scenes with terrain variation (hills, slopes)

The idea: ground is the dominant flat surface. After removing it,
what remains are vertical structures (obstacles) + vegetation/noise.
The voxel classifier then only needs to distinguish obstacles from vegetation.

Usage:
    from ground_removal import remove_ground_voxels
    non_ground_df = remove_ground_voxels(voxel_df, method='tile')
"""

import numpy as np
import pandas as pd


def fit_ground_plane_ransac(points, n_iterations=1000, distance_threshold=0.5,
                            min_inlier_ratio=0.3):
    """
    Fit a ground plane using RANSAC on 3D points.

    Args:
        points: (N, 3) array of [x, y, z] positions
        n_iterations: RANSAC iterations
        distance_threshold: max distance from plane to count as inlier (meters)
        min_inlier_ratio: minimum fraction of inliers for a valid plane

    Returns:
        plane: (4,) array [a, b, c, d] where ax + by + cz + d = 0, or None
        inlier_mask: (N,) boolean mask of ground points
    """
    n = len(points)
    if n < 3:
        return None, np.zeros(n, dtype=bool)

    best_inlier_count = 0
    best_plane = None
    best_mask = np.zeros(n, dtype=bool)

    rng = np.random.RandomState(42)

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal = normal / norm_len

        # Plane equation: normal . (x - p1) = 0  =>  ax + by + cz + d = 0
        d = -np.dot(normal, p1)
        plane = np.array([normal[0], normal[1], normal[2], d])

        # Check that normal is roughly vertical (ground should be ~horizontal)
        # Allow up to ~30 degrees of tilt for terrain variation
        if abs(normal[2]) < 0.85:
            continue

        # Count inliers
        distances = np.abs(points @ normal + d)
        inlier_mask = distances < distance_threshold
        n_inliers = inlier_mask.sum()

        if n_inliers > best_inlier_count:
            best_inlier_count = n_inliers
            best_plane = plane
            best_mask = inlier_mask

    if best_plane is None or best_inlier_count / n < min_inlier_ratio:
        return None, np.zeros(n, dtype=bool)

    return best_plane, best_mask


def remove_ground_voxels(df, method='tile', distance_threshold=1.0,
                         tile_size=50.0, min_height_above_ground=1.5,
                         verbose=True):
    """
    Remove ground voxels from a consolidated voxel DataFrame.

    Two-pass approach:
      1. RANSAC fits the dominant ground plane
      2. Voxels close to the ground plane are flagged
      3. Remaining voxels above the ground are kept

    Args:
        df: voxel DataFrame with world_x, world_y, world_z columns
        method: 'global' (single plane) or 'tile' (per-tile planes for terrain)
        distance_threshold: RANSAC inlier distance (meters)
        tile_size: tile side length for tiled RANSAC (meters)
        min_height_above_ground: minimum height above estimated ground to keep (meters)
        verbose: print progress

    Returns:
        non_ground_df: DataFrame with ground voxels removed
        ground_heights: dict mapping (tile_x, tile_y) -> estimated ground z
    """
    if verbose:
        print(f"  Ground removal (method={method}, threshold={distance_threshold}m)")

    points = df[['world_x', 'world_y', 'world_z']].values
    n_total = len(df)

    if method == 'global':
        plane, ground_mask = fit_ground_plane_ransac(
            points, distance_threshold=distance_threshold
        )
        if plane is not None:
            # For each voxel, compute height above ground plane
            a, b, c, d = plane
            ground_z_at_xy = -(a * points[:, 0] + b * points[:, 1] + d) / (c + 1e-10)
            height_above = points[:, 2] - ground_z_at_xy
            keep_mask = height_above > min_height_above_ground
        else:
            if verbose:
                print("    WARNING: RANSAC failed, keeping all voxels")
            keep_mask = np.ones(n_total, dtype=bool)
        ground_heights = {'global': float(-(d) / (c + 1e-10)) if plane is not None else 0}

    elif method == 'tile':
        keep_mask = np.ones(n_total, dtype=bool)
        ground_heights = {}

        # Divide scene into tiles
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        n_tiles_x = max(1, int(np.ceil((x_max - x_min) / tile_size)))
        n_tiles_y = max(1, int(np.ceil((y_max - y_min) / tile_size)))

        if verbose:
            print(f"    Grid: {n_tiles_x}x{n_tiles_y} tiles ({tile_size}m each)")

        n_ground_removed = 0

        for tx in range(n_tiles_x):
            for ty in range(n_tiles_y):
                # Tile bounds
                x_lo = x_min + tx * tile_size
                x_hi = x_lo + tile_size
                y_lo = y_min + ty * tile_size
                y_hi = y_lo + tile_size

                # Select voxels in this tile
                tile_mask = (
                    (points[:, 0] >= x_lo) & (points[:, 0] < x_hi) &
                    (points[:, 1] >= y_lo) & (points[:, 1] < y_hi)
                )
                tile_indices = np.where(tile_mask)[0]

                if len(tile_indices) < 10:
                    continue

                tile_points = points[tile_indices]

                # Fit ground plane for this tile
                plane, _ = fit_ground_plane_ransac(
                    tile_points,
                    distance_threshold=distance_threshold,
                    n_iterations=500,
                )

                if plane is None:
                    continue

                a, b, c, d = plane

                # Compute height above ground for each voxel in tile
                ground_z = -(a * tile_points[:, 0] + b * tile_points[:, 1] + d) / (c + 1e-10)
                height_above = tile_points[:, 2] - ground_z

                # Mark ground voxels for removal
                is_ground = height_above <= min_height_above_ground
                keep_mask[tile_indices[is_ground]] = False
                n_ground_removed += is_ground.sum()

                # Store ground height estimate
                ground_heights[(tx, ty)] = float(np.median(ground_z))

        if verbose:
            print(f"    Removed {n_ground_removed:,} ground voxels")

    else:
        raise ValueError(f"Unknown method: {method}")

    non_ground_df = df[keep_mask].reset_index(drop=True)

    if verbose:
        n_kept = len(non_ground_df)
        print(f"    Kept {n_kept:,} / {n_total:,} voxels "
              f"({100*n_kept/n_total:.1f}%)")

    return non_ground_df, ground_heights


def remove_ground_from_points(xyz_local, ego_pose, ground_heights,
                              voxel_size=0.5, tile_size=50.0,
                              min_height=1.5):
    """
    Remove ground points from a single frame using pre-computed ground heights.
    Used during inference back-projection.

    Args:
        xyz_local: (N, 3) local frame points
        ego_pose: (4,) raw ego pose
        ground_heights: dict from remove_ground_voxels
        voxel_size: voxel size used during consolidation
        tile_size: tile size used during ground removal
        min_height: minimum height above ground

    Returns:
        non_ground_mask: (N,) boolean mask
    """
    from consolidate_scene import local_to_world

    xyz_world = local_to_world(xyz_local, ego_pose)

    if 'global' in ground_heights:
        # Simple case: single ground height
        non_ground_mask = xyz_world[:, 2] > (ground_heights['global'] + min_height)
    else:
        # Tile-based: look up ground height per tile
        non_ground_mask = np.ones(len(xyz_local), dtype=bool)

        # Get tile grid origin (we need to reconstruct from ground_heights keys)
        if not ground_heights:
            return non_ground_mask

        # Use median of all ground heights as fallback
        fallback_z = np.median(list(ground_heights.values()))

        # For simplicity, use the z of the lowest quartile as ground estimate
        z_threshold = fallback_z + min_height
        non_ground_mask = xyz_world[:, 2] > z_threshold

    return non_ground_mask
