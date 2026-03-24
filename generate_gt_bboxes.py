"""
Script 2 — Generate Ground-Truth 3D Bounding Boxes
====================================================
Reads the preprocessed .npz frames produced by preprocess_frames.py,
assigns class labels from RGB, clusters obstacle points with DBSCAN,
fits oriented 3D bounding boxes via PCA, and exports a CSV.

Input:  preprocessed/ directory with manifest.csv + per-frame .npz files
Output: gt_bboxes.csv

Pipeline per frame:
  1. Load xyz + rgb from .npz
  2. Assign class labels from RGB ground truth
  3. For each obstacle class: cluster with DBSCAN
  4. For each cluster: fit oriented bounding box via 2D PCA + Z extents
  5. Collect all boxes with ego pose + class info

Usage:
    python generate_gt_bboxes.py --processed-dir processed/ --out gt_bboxes.csv

    # Visualize one frame to sanity-check (requires open3d)
    python generate_gt_bboxes.py --processed-dir processed/ \
        --out gt_bboxes.csv --viz scene_1 --viz-pose 0

Requirements:
    pip install numpy pandas scikit-learn scipy
    pip install open3d   (optional, for --viz)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


# ─── Class definitions ────────────────────────────────────────────────────────
CLASS_RGB = {
    (38, 23, 180):  {"id": 0, "label": "Antenna"},
    (177, 132, 47): {"id": 1, "label": "Cable"},
    (129, 81, 97):  {"id": 2, "label": "Electric Pole"},
    (66, 132, 9):   {"id": 3, "label": "Wind Turbine"},
}

CLASS_ID_TO_LABEL = {0: "Antenna", 1: "Cable", 2: "Electric Pole", 3: "Wind Turbine"}

# ─── DBSCAN parameters per class ─────────────────────────────────────────────
# Tune these by running with --viz and inspecting results.
# Cables need larger eps because they're thin and elongated.
DBSCAN_PARAMS = {
    0: {"eps": 3.0,  "min_samples": 20},   # Antenna
    1: {"eps": 5.0,  "min_samples": 10},   # Cable
    2: {"eps": 2.0,  "min_samples": 20},   # Electric Pole
    3: {"eps": 5.0,  "min_samples": 30},   # Wind Turbine
}

# Minimum points for a cluster to count as a valid object
MIN_CLUSTER_POINTS = {
    0: 15,   # Antenna
    1: 10,   # Cable
    2: 15,   # Electric Pole
    3: 30,   # Wind Turbine
}


# ─── Class assignment ─────────────────────────────────────────────────────────
def assign_classes(rgb):
    """
    Assign class_id from RGB ground truth.

    Parameters
    ----------
    rgb : (N, 3) uint8 array

    Returns
    -------
    class_ids : (N,) int array  — -1 = Background, 0-3 = obstacle classes
    """
    class_ids = np.full(len(rgb), -1, dtype=np.int32)
    for (r, g, b), info in CLASS_RGB.items():
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        class_ids[mask] = info["id"]
    return class_ids


# ─── Oriented bounding box fitting ───────────────────────────────────────────
def fit_oriented_bbox(points_3d):
    """
    Fit a yaw-only oriented 3D bounding box via PCA.

    1. PCA on XY projection → yaw angle
    2. Rotate points by -yaw to axis-align
    3. Take axis-aligned extents → width, length, height
    4. Compute center, rotate back

    Returns dict: center_x/y/z, width, length, height, yaw (radians)
    """
    pts = np.asarray(points_3d, dtype=np.float64)

    if len(pts) < 3:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = (mn + mx) / 2
        dims = np.maximum(mx - mn, 0.1)
        return {
            "center_x": center[0], "center_y": center[1], "center_z": center[2],
            "width": dims[0], "length": dims[1], "height": dims[2],
            "yaw": 0.0,
        }

    # PCA on XY plane
    xy = pts[:, :2]
    xy_c = xy - xy.mean(axis=0)
    cov = np.cov(xy_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    yaw = np.arctan2(principal[1], principal[0])

    # Rotate by -yaw to axis-align
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])

    pts_rot = (R @ pts.T).T
    mn = pts_rot.min(axis=0)
    mx = pts_rot.max(axis=0)
    center_rot = (mn + mx) / 2
    dims = np.maximum(mx - mn, 0.1)

    # Rotate center back
    R_inv = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0,            0,            1]])
    center = R_inv @ center_rot

    return {
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "width":  dims[0],
        "length": dims[1],
        "height": dims[2],
        "yaw":    yaw,
    }


# ─── Per-frame extraction ────────────────────────────────────────────────────
def voxel_downsample(points, voxel_size=0.3):
    """
    Fast voxel downsampling: keep one point per voxel cell.
    Returns (downsampled_points, indices_into_original).
    Used to speed up DBSCAN on very dense point clouds.
    """
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    # Unique voxels — keep first occurrence
    _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx], unique_idx


# Maximum points before we voxel-downsample prior to DBSCAN.
# Ball_tree DBSCAN is still slow with large eps on dense clouds.
# Downsampling doesn't hurt bbox quality since we reassign original points after.
MAX_POINTS_FOR_DBSCAN = 15_000

def extract_bboxes(xyz, rgb, verbose=False, frame_label=""):
    """
    Cluster obstacle points per class and fit bounding boxes.

    Parameters
    ----------
    xyz     : (N, 3) float array — local Cartesian coordinates
    rgb     : (N, 3) uint8 array — ground-truth colors
    verbose : bool — print per-class timing and point counts
    frame_label : str — for logging

    Returns
    -------
    list of dicts, one per detected object
    """
    class_ids = assign_classes(rgb)
    results = []

    for cid in [0, 1, 2, 3]:
        mask = class_ids == cid
        n_pts = mask.sum()
        if n_pts == 0:
            continue

        class_xyz = xyz[mask]
        label = CLASS_ID_TO_LABEL[cid]

        # Voxel downsample if too many points (prevents DBSCAN from hanging)
        downsampled = False
        if n_pts > MAX_POINTS_FOR_DBSCAN:
            class_xyz_ds, _ = voxel_downsample(class_xyz, voxel_size=0.3)
            downsampled = True
            if verbose:
                print(f"      {label}: {n_pts:,} pts → downsampled to "
                      f"{len(class_xyz_ds):,} ... ", end="", flush=True)
        else:
            class_xyz_ds = class_xyz
            if verbose:
                print(f"      {label}: {n_pts:,} pts ... ", end="", flush=True)

        params = DBSCAN_PARAMS[cid]

        t0 = time.time()
        clustering = DBSCAN(
            eps=params["eps"],
            min_samples=params["min_samples"],
            algorithm="ball_tree",
        )
        cluster_labels = clustering.fit_predict(class_xyz_ds)
        dt = time.time() - t0

        n_clusters = len(set(cluster_labels) - {-1})

        if verbose:
            n_noise = (cluster_labels == -1).sum()
            print(f"{n_clusters} clusters, {n_noise} noise  [{dt:.2f}s]")

        # If we downsampled for clustering, assign original points to the
        # nearest cluster center so bounding boxes use all available points
        if downsampled and n_clusters > 0:
            # Compute cluster centers from downsampled data
            cluster_centers = {}
            for c_id in set(cluster_labels) - {-1}:
                cluster_centers[c_id] = class_xyz_ds[cluster_labels == c_id].mean(axis=0)

            # Assign each original point to the nearest cluster center
            # (only if within 2 * eps to avoid grabbing distant noise)
            centers_arr = np.array([cluster_centers[c] for c in sorted(cluster_centers)])
            center_ids = sorted(cluster_centers.keys())

            from scipy.spatial import cKDTree
            tree = cKDTree(centers_arr)
            dists, indices = tree.query(class_xyz, k=1)
            full_labels = np.full(len(class_xyz), -1, dtype=np.int32)
            close_enough = dists < (2 * params["eps"])
            full_labels[close_enough] = np.array(center_ids)[indices[close_enough]]

            # Use full point set for bbox fitting
            for c_id in set(full_labels) - {-1}:
                cluster_pts = class_xyz[full_labels == c_id]
                if len(cluster_pts) < MIN_CLUSTER_POINTS[cid]:
                    continue
                bbox = fit_oriented_bbox(cluster_pts)
                bbox["class_ID"] = cid
                bbox["class_label"] = label
                bbox["num_points"] = len(cluster_pts)
                results.append(bbox)
        else:
            # Standard path: use cluster labels directly
            for cluster_id in set(cluster_labels) - {-1}:
                cluster_pts = class_xyz_ds[cluster_labels == cluster_id]
                if len(cluster_pts) < MIN_CLUSTER_POINTS[cid]:
                    continue
                bbox = fit_oriented_bbox(cluster_pts)
                bbox["class_ID"] = cid
                bbox["class_label"] = label
                bbox["num_points"] = len(cluster_pts)
                results.append(bbox)

    return results


# ─── Load a preprocessed frame ───────────────────────────────────────────────
def load_frame(npz_path):
    """Load a .npz frame → xyz, reflectivity, rgb, ego_pose."""
    data = np.load(npz_path)
    return data["xyz"], data["reflectivity"], data["rgb"], data["ego_pose"]


# ─── Main pipeline ────────────────────────────────────────────────────────────
def process_all(processed_dir, output_csv, verbose=True):
    """Read manifest, process every frame, write GT CSV."""
    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        print("Run preprocess_frames.py first.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(manifest)} frames from {manifest['scene'].nunique()} scenes")

    all_bboxes = []
    total_frames = len(manifest)

    for frame_num, (_, row) in enumerate(manifest.iterrows(), 1):
        npz_path = row["file"]
        if not Path(npz_path).exists():
            print(f"  WARNING: {npz_path} not found — skipping")
            continue

        frame_label = f"{row['scene']} pose {row['pose_index']:3d}"

        if verbose:
            print(f"  [{frame_num}/{total_frames}] {frame_label}  "
                  f"({row.get('num_points_valid', '?')} pts)", flush=True)

        xyz, refl, rgb, ego_pose = load_frame(npz_path)

        t0 = time.time()
        bboxes = extract_bboxes(xyz, rgb, verbose=verbose, frame_label=frame_label)
        dt = time.time() - t0

        # Attach frame metadata
        for b in bboxes:
            b["scene"] = row["scene"]
            b["pose_index"] = row["pose_index"]
            b["ego_x"] = ego_pose[0]
            b["ego_y"] = ego_pose[1]
            b["ego_z"] = ego_pose[2]
            b["ego_yaw"] = ego_pose[3]
            b["frame_file"] = npz_path

        all_bboxes.extend(bboxes)

        if verbose:
            if len(bboxes) > 0:
                labels = sorted(set(b["class_label"] for b in bboxes))
                print(f"    → {len(bboxes)} objects ({', '.join(labels)})  [{dt:.2f}s]")
            else:
                print(f"    → no obstacles  [{dt:.2f}s]")

    if len(all_bboxes) == 0:
        print("No bounding boxes found across all frames.")
        print("Check DBSCAN_PARAMS — eps might be too small or min_samples too high.")
        sys.exit(1)

    # Build output CSV
    df_out = pd.DataFrame(all_bboxes)
    df_out = df_out.rename(columns={
        "center_x": "bbox_center_x",
        "center_y": "bbox_center_y",
        "center_z": "bbox_center_z",
        "width":    "bbox_width",
        "length":   "bbox_length",
        "height":   "bbox_height",
        "yaw":      "bbox_yaw",
    })

    csv_cols = [
        "scene", "pose_index", "frame_file",
        "ego_x", "ego_y", "ego_z", "ego_yaw",
        "bbox_center_x", "bbox_center_y", "bbox_center_z",
        "bbox_width", "bbox_length", "bbox_height", "bbox_yaw",
        "class_ID", "class_label", "num_points",
    ]
    df_out = df_out[csv_cols]
    df_out.to_csv(output_csv, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE — {len(df_out)} ground-truth bounding boxes")
    print(f"Saved to: {output_csv}")
    print(f"\nClass breakdown:")
    print(df_out["class_label"].value_counts().to_string())
    print(f"\nBbox size stats (meters):")
    for dim in ["bbox_width", "bbox_length", "bbox_height"]:
        print(f"  {dim}: mean={df_out[dim].mean():.2f}  "
              f"median={df_out[dim].median():.2f}  "
              f"max={df_out[dim].max():.2f}")
    print(f"\nObjects per frame: "
          f"mean={df_out.groupby(['scene','pose_index']).size().mean():.1f}  "
          f"max={df_out.groupby(['scene','pose_index']).size().max()}")

    return df_out


# ─── Optional visualization ──────────────────────────────────────────────────
def visualize_frame(processed_dir, scene_name, pose_index):
    """Open3D view of one frame with bounding boxes overlaid."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed — run: pip install open3d")
        return

    npz_path = Path(processed_dir) / scene_name / f"frame_{pose_index:03d}.npz"
    if not npz_path.exists():
        print(f"Frame not found: {npz_path}")
        return

    xyz, refl, rgb, ego_pose = load_frame(npz_path)
    class_ids = assign_classes(rgb)

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Color: gray background, class-colored obstacles
    colors = np.full((len(xyz), 3), 0.5)
    palette = {
        0: [0.15, 0.09, 0.70],   # Antenna — blue
        1: [0.69, 0.52, 0.18],   # Cable — gold
        2: [0.50, 0.32, 0.38],   # Electric Pole — mauve
        3: [0.26, 0.52, 0.04],   # Wind Turbine — green
    }
    for cid, col in palette.items():
        colors[class_ids == cid] = col
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Extract bboxes
    bboxes = extract_bboxes(xyz, rgb, verbose=True, frame_label=f"{scene_name} pose {pose_index}")
    print(f"\n{scene_name} pose {pose_index}: {len(bboxes)} objects")

    geometries = [pcd]
    bbox_colors = {
        0: [0.0, 0.0, 1.0],
        1: [1.0, 0.8, 0.0],
        2: [1.0, 0.0, 1.0],
        3: [0.0, 1.0, 0.0],
    }

    for b in bboxes:
        center = np.array([b["center_x"], b["center_y"], b["center_z"]])
        extent = np.array([b["width"], b["length"], b["height"]])
        yaw = b["yaw"]

        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,            1],
        ])

        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        obb.color = bbox_colors.get(b["class_ID"], [1, 1, 1])
        geometries.append(obb)

        print(f"  {b['class_label']:15s}  "
              f"center=({b['center_x']:.1f}, {b['center_y']:.1f}, {b['center_z']:.1f})  "
              f"size=({b['width']:.1f} x {b['length']:.1f} x {b['height']:.1f})  "
              f"pts={b['num_points']}")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{scene_name} — Pose {pose_index}",
        width=1280, height=720,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate GT bounding boxes from preprocessed LiDAR frames"
    )
    parser.add_argument(
        "--processed-dir", default="processed/",
        help="Directory with manifest.csv + per-frame .npz (output of preprocess_frames.py)",
    )
    parser.add_argument(
        "--out", default="gt_bboxes.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--viz", default=None, metavar="SCENE_NAME",
        help="(Optional) Visualize bboxes for a scene, e.g. 'scene_1'",
    )
    parser.add_argument(
        "--viz-pose", type=int, default=0,
        help="Pose index to visualize (default: 0)",
    )
    args = parser.parse_args()

    # Run full extraction
    process_all(args.processed_dir, args.out)

    # Optional visualization
    if args.viz is not None:
        visualize_frame(args.processed_dir, args.viz, args.viz_pose)


if __name__ == "__main__":
    main()