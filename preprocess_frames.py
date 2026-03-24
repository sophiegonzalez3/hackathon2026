"""
Script 1 — Preprocess LiDAR Frames
====================================
Reads raw HDF5 scene files and exports one .npz file per frame containing
the cleaned, converted point cloud ready for both GT extraction and model training.

Each .npz contains:
  - xyz          : (N, 3) float32  — local Cartesian coordinates (meters)
  - reflectivity : (N,)   uint8    — laser return intensity
  - rgb          : (N, 3) uint8    — ground-truth color labels
  - ego_pose     : (4,)   float64  — [ego_x, ego_y, ego_z, ego_yaw] in raw units

A manifest CSV is also written listing every frame with metadata.

Pipeline per frame:
  1. Filter points by ego pose (unique frame identifier)
  2. Drop invalid beams (distance_cm == 0)
  3. Convert spherical coords → local Cartesian (meters)
  4. Keep reflectivity + RGB as-is
  5. Save to .npz

Usage:
    python preprocess_frames.py --data-dir trainingData/ --out-dir processed/

Output structure:
    processed/
    ├── manifest.csv
    ├── scene_1/
    │   ├── frame_000.npz
    │   ├── frame_001.npz
    │   └── ...
    ├── scene_2/
    │   └── ...
    └── ...

Requirements:
    pip install numpy pandas h5py
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_h5(file_path, dataset_name="lidar_points"):
    """Load HDF5 structured array → DataFrame."""
    with h5py.File(file_path, "r") as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in {file_path}")
        pts = f[dataset_name][:]
    return pd.DataFrame({name: pts[name] for name in pts.dtype.names})


def get_unique_poses(df):
    """Unique ego poses → DataFrame with pose_index and point count."""
    pose_fields = ["ego_x", "ego_y", "ego_z", "ego_yaw"]
    return (
        df.groupby(pose_fields)
        .size()
        .reset_index(name="num_points")
        .reset_index(names="pose_index")
    )


def filter_by_pose(df, pose_row):
    """Filter dataframe to a single frame by ego pose quadruplet."""
    return df[
        (df["ego_x"] == pose_row["ego_x"])
        & (df["ego_y"] == pose_row["ego_y"])
        & (df["ego_z"] == pose_row["ego_z"])
        & (df["ego_yaw"] == pose_row["ego_yaw"])
    ].reset_index(drop=True)


def spherical_to_local_cartesian(df):
    """Convert spherical coordinates (cm, 1/100 deg) → local Cartesian (meters)."""
    dist = df["distance_cm"].to_numpy() / 100.0
    az = np.radians(df["azimuth_raw"].to_numpy() / 100.0)
    el = np.radians(df["elevation_raw"].to_numpy() / 100.0)

    x = dist * np.cos(el) * np.cos(az)
    y = -dist * np.cos(el) * np.sin(az)
    z = dist * np.sin(el)

    return np.column_stack((x, y, z)).astype(np.float32)


# ─── Per-scene processing ────────────────────────────────────────────────────
def process_scene(file_path, out_dir, verbose=True):
    """
    Process one HDF5 scene file → per-frame .npz files.

    Returns a list of dicts (one per frame) for the manifest.
    """
    scene_name = Path(file_path).stem  # e.g. "scene_1"
    scene_dir = Path(out_dir) / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")

    df = load_h5(file_path)
    poses = get_unique_poses(df)

    if verbose:
        print(f"  {len(df):,} total points | {len(poses)} frames")

    manifest_rows = []

    for _, pose_row in poses.iterrows():
        pose_idx = int(pose_row["pose_index"])
        frame = filter_by_pose(df, pose_row)

        # ── Step 1: drop invalid beams ──
        n_before = len(frame)
        frame = frame[frame["distance_cm"] > 0].reset_index(drop=True)
        n_valid = len(frame)

        if n_valid == 0:
            if verbose:
                print(f"  Pose {pose_idx:3d}: 0 valid points — skipping")
            continue

        # ── Step 2: spherical → local Cartesian ──
        xyz = spherical_to_local_cartesian(frame)

        # ── Step 3: extract reflectivity ──
        reflectivity = frame["reflectivity"].to_numpy().astype(np.uint8)

        # ── Step 4: extract RGB ground-truth labels ──
        rgb = np.column_stack([
            frame["r"].to_numpy(),
            frame["g"].to_numpy(),
            frame["b"].to_numpy(),
        ]).astype(np.uint8)

        # ── Step 5: ego pose vector ──
        ego_pose = np.array([
            pose_row["ego_x"],
            pose_row["ego_y"],
            pose_row["ego_z"],
            pose_row["ego_yaw"],
        ], dtype=np.float64)

        # ── Save .npz ──
        frame_filename = f"frame_{pose_idx:03d}.npz"
        frame_path = scene_dir / frame_filename

        np.savez_compressed(
            frame_path,
            xyz=xyz,
            reflectivity=reflectivity,
            rgb=rgb,
            ego_pose=ego_pose,
        )

        # ── Manifest entry ──
        manifest_rows.append({
            "scene": scene_name,
            "pose_index": pose_idx,
            "ego_x": pose_row["ego_x"],
            "ego_y": pose_row["ego_y"],
            "ego_z": pose_row["ego_z"],
            "ego_yaw": pose_row["ego_yaw"],
            "num_points_raw": n_before,
            "num_points_valid": n_valid,
            "num_invalid_dropped": n_before - n_valid,
            "file": str(frame_path),
        })

        if verbose:
            pct_valid = 100.0 * n_valid / n_before if n_before > 0 else 0
            print(f"  Pose {pose_idx:3d}: {n_valid:>7,} valid / {n_before:>7,} total "
                  f"({pct_valid:.1f}%) → {frame_filename}")

    if verbose:
        print(f"  → Saved {len(manifest_rows)} frames to {scene_dir}/")

    return manifest_rows


# ─── Main pipeline ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw HDF5 LiDAR scenes into per-frame .npz files"
    )
    parser.add_argument(
        "--data-dir", default="trainingData/",
        help="Directory containing scene_*.h5 files",
    )
    parser.add_argument(
        "--out-dir", default="processed/",
        help="Output directory for .npz files and manifest",
    )
    args = parser.parse_args()

    # Find all scene files
    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "scene_*.h5")))
    if not h5_files:
        print(f"No scene_*.h5 files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(h5_files)} scene files in {args.data_dir}")
    print(f"Output directory: {args.out_dir}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Process each scene
    all_manifest_rows = []
    for fpath in h5_files:
        rows = process_scene(fpath, args.out_dir)
        all_manifest_rows.extend(rows)

    # Write manifest CSV
    manifest_df = pd.DataFrame(all_manifest_rows)
    manifest_path = Path(args.out_dir) / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE — {len(manifest_df)} frames preprocessed")
    print(f"Manifest: {manifest_path}")
    print(f"\nPer-scene breakdown:")
    print(manifest_df.groupby("scene")["num_points_valid"].agg(["count", "sum", "mean"])
          .rename(columns={"count": "frames", "sum": "total_pts", "mean": "avg_pts"})
          .to_string(float_format="%.0f"))

    total_dropped = manifest_df["num_invalid_dropped"].sum()
    total_raw = manifest_df["num_points_raw"].sum()
    print(f"\nInvalid beams dropped: {total_dropped:,} / {total_raw:,} "
          f"({100.0 * total_dropped / total_raw:.1f}%)")


if __name__ == "__main__":
    main()