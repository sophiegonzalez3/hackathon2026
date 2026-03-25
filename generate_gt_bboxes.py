"""
Script 2 — Generate Ground-Truth 3D Bounding Boxes (config-driven)
====================================================================
Reads the preprocessed .npz frames produced by preprocess_frames.py,
assigns class labels from RGB, clusters obstacle points with DBSCAN,
fits oriented 3D bounding boxes via PCA, and exports a CSV.

All tunable parameters live in a YAML config file.  Duplicate the
config, tweak values, and compare CSV outputs side-by-side.

Usage:
    # Default config
    python generate_gt_bboxes.py --config bbox_config_default.yaml

    # Compare a second set of params
    python generate_gt_bboxes.py --config bbox_config_v2.yaml

    # Override output path
    python generate_gt_bboxes.py --config bbox_config_v2.yaml --out my_bboxes.csv

Requirements:
    pip install numpy pandas scikit-learn scipy pyyaml
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN


# ═══════════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG = {
    "processed_dir": "processed/",
    "output_dir": ".",
    "run_name": None,
    "verbose": True,
    "max_points_for_dbscan": 15_000,
    "voxel_size": 0.3,
    "dbscan": {
        "Antenna":       {"eps": 3.0, "min_samples": 20},
        "Cable":         {"eps": 5.0, "min_samples": 10},
        "Electric Pole": {"eps": 2.0, "min_samples": 20},
        "Wind Turbine":  {"eps": 5.0, "min_samples": 30},
    },
    "min_cluster_points": {
        "Antenna": 15, "Cable": 10, "Electric Pole": 15, "Wind Turbine": 30,
    },
    "reassign_radius_multiplier": 2.0,
    "bbox_padding": {
        "Antenna": 0.0, "Cable": 0.0, "Electric Pole": 0.0, "Wind Turbine": 0.0,
    },
    "merge": {
        "Antenna":       {"enabled": False, "merge_distance": 0.0},
        "Cable":         {"enabled": False, "merge_distance": 0.0},
        "Electric Pole": {"enabled": False, "merge_distance": 0.0},
        "Wind Turbine":  {"enabled": False, "merge_distance": 0.0},
    },
}


def load_config(config_path=None):
    """Load YAML config with fallback defaults for any missing keys."""
    cfg = copy.deepcopy(_DEFAULT_CONFIG)

    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            print(f"WARNING: config file {p} not found — using defaults")
            return cfg
        with open(p) as f:
            user = yaml.safe_load(f) or {}

        # ── Top-level scalars ─────────────────────────────────────────
        for key in ("processed_dir", "reassign_radius_multiplier",
                     "output_dir", "run_name", "verbose"):
            if key in user:
                cfg[key] = user[key]

        # ── Flat top-level keys (backward compat) ────────────────────
        if "max_points_for_dbscan" in user:
            cfg["max_points_for_dbscan"] = user["max_points_for_dbscan"]
        if "voxel_size" in user:
            cfg["voxel_size"] = user["voxel_size"]

        # ── Nested voxel_downsample section (preferred YAML layout) ──
        if "voxel_downsample" in user:
            vd = user["voxel_downsample"]
            if "max_points_for_dbscan" in vd:
                cfg["max_points_for_dbscan"] = vd["max_points_for_dbscan"]
            if "voxel_size" in vd:
                cfg["voxel_size"] = vd["voxel_size"]

        # ── Per-class dict sections ──────────────────────────────────
        for section in ("dbscan", "min_cluster_points", "bbox_padding", "merge"):
            if section in user:
                for cls_name, val in user[section].items():
                    # Normalize underscores to spaces so YAML keys
                    # like "Electric_Pole" match code keys "Electric Pole"
                    canonical = cls_name.replace("_", " ")
                    if isinstance(val, dict):
                        cfg[section].setdefault(canonical, {}).update(val)
                    else:
                        cfg[section][canonical] = val

        if "output_csv" in user:
            cfg["output_csv"] = user["output_csv"]

    return cfg


def derive_output_csv(cfg, config_path):
    """Build the output CSV path from run_name / output_dir / config filename.

    Priority:
      1. cfg["output_csv"]  (explicit override in YAML)
      2. output_dir / gt_bboxes_<run_name>.csv
      3. output_dir / gt_bboxes_<config_stem_tag>.csv
      4. gt_bboxes.csv  (fallback)
    """
    output_dir = Path(cfg.get("output_dir", "."))

    if "output_csv" in cfg:
        return str(output_dir / cfg["output_csv"])

    run_name = cfg.get("run_name")
    if run_name:
        return str(output_dir / f"gt_bboxes_{run_name}.csv")

    if config_path is not None:
        stem = Path(config_path).stem
        tag = stem.replace("bbox_config_", "").replace("bbox_config", "default")
        if not tag:
            tag = "default"
        return str(output_dir / f"gt_bboxes_{tag}.csv")

    return str(output_dir / "gt_bboxes.csv")


def print_config(cfg, config_path):
    """Pretty-print the active config for reproducibility."""
    print(f"\n{'─'*60}")
    print(f"  CONFIG: {config_path or 'built-in defaults'}")
    if cfg.get("run_name"):
        print(f"  run_name                 : {cfg['run_name']}")
    print(f"{'─'*60}")
    print(f"  processed_dir            : {cfg['processed_dir']}")
    print(f"  output_dir               : {cfg.get('output_dir', '.')}")
    print(f"  max_points_for_dbscan    : {cfg['max_points_for_dbscan']:,}")
    print(f"  voxel_size               : {cfg['voxel_size']}")
    print(f"  reassign_radius_multiplier: {cfg['reassign_radius_multiplier']}")
    print()
    for label in ("Antenna", "Cable", "Electric Pole", "Wind Turbine"):
        db = cfg["dbscan"][label]
        mc = cfg["min_cluster_points"][label]
        pad = cfg["bbox_padding"][label]
        mg = cfg["merge"][label]
        merge_str = f"merge<{mg['merge_distance']}m" if mg["enabled"] else "no merge"
        print(f"  {label:15s}  eps={db['eps']:<5}  min_s={db['min_samples']:<4}  "
              f"min_pts={mc:<4}  pad={pad:.1f}m  {merge_str}")
    print(f"{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Class definitions
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_RGB = {
    (38, 23, 180):  {"id": 0, "label": "Antenna"},
    (177, 132, 47): {"id": 1, "label": "Cable"},
    (129, 81, 97):  {"id": 2, "label": "Electric Pole"},
    (66, 132, 9):   {"id": 3, "label": "Wind Turbine"},
}

CLASS_ID_TO_LABEL = {0: "Antenna", 1: "Cable", 2: "Electric Pole", 3: "Wind Turbine"}
LABEL_TO_CLASS_ID = {v: k for k, v in CLASS_ID_TO_LABEL.items()}


def assign_classes(rgb):
    """Assign class_id from RGB ground truth. -1 = Background."""
    class_ids = np.full(len(rgb), -1, dtype=np.int32)
    for (r, g, b), info in CLASS_RGB.items():
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        class_ids[mask] = info["id"]
    return class_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Oriented bounding-box fitting
# ═══════════════════════════════════════════════════════════════════════════════

def fit_oriented_bbox(points_3d, padding=0.0):
    """
    Fit a yaw-only oriented 3D bounding box via PCA on XY.

    1. PCA on XY projection -> yaw angle
    2. Rotate points by -yaw to axis-align
    3. Take axis-aligned extents -> width, length, height
    4. Add padding to each dimension
    5. Compute center, rotate back

    Parameters
    ----------
    points_3d : (N, 3) array
    padding   : float — meters added to EACH side (total = 2*padding per axis)
    """
    pts = np.asarray(points_3d, dtype=np.float64)

    if len(pts) < 3:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = (mn + mx) / 2
        dims = np.maximum(mx - mn, 0.1) + 2 * padding
        return {
            "center_x": center[0], "center_y": center[1], "center_z": center[2],
            "width": dims[0], "length": dims[1], "height": dims[2],
            "yaw": 0.0,
        }

    xy = pts[:, :2]
    xy_c = xy - xy.mean(axis=0)
    cov = np.cov(xy_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    yaw = np.arctan2(principal[1], principal[0])

    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts_rot = (R @ pts.T).T
    mn = pts_rot.min(axis=0)
    mx = pts_rot.max(axis=0)
    center_rot = (mn + mx) / 2
    dims = np.maximum(mx - mn, 0.1) + 2 * padding

    R_inv = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0,            0,            1]])
    center = R_inv @ center_rot

    return {
        "center_x": center[0], "center_y": center[1], "center_z": center[2],
        "width": dims[0], "length": dims[1], "height": dims[2],
        "yaw": yaw,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Voxel downsampling
# ═══════════════════════════════════════════════════════════════════════════════

def voxel_downsample(points, voxel_size=0.3):
    """Fast voxel downsampling: one point per voxel cell."""
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx], unique_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Cluster merging (union-find)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_cluster_labels(points, cluster_labels, merge_distance):
    """
    Merge DBSCAN clusters whose centroids are within merge_distance.

    Uses union-find so transitive merges are handled correctly
    (A close to B, B close to C -> all three merge).
    """
    unique_ids = sorted(set(cluster_labels) - {-1})
    if len(unique_ids) <= 1:
        return cluster_labels

    centroids = {}
    for cid in unique_ids:
        centroids[cid] = points[cluster_labels == cid].mean(axis=0)

    parent = {cid: cid for cid in unique_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    id_list = list(unique_ids)
    for i in range(len(id_list)):
        for j in range(i + 1, len(id_list)):
            a, b = id_list[i], id_list[j]
            if np.linalg.norm(centroids[a] - centroids[b]) < merge_distance:
                union(a, b)

    merged = cluster_labels.copy()
    for cid in unique_ids:
        root = find(cid)
        if root != cid:
            merged[cluster_labels == cid] = root

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_bboxes(xyz, rgb, cfg, verbose=False, frame_label=""):
    """Cluster obstacle points per class and fit bounding boxes."""
    class_ids = assign_classes(rgb)
    results = []

    max_pts = cfg["max_points_for_dbscan"]
    vox_size = cfg["voxel_size"]
    reassign_mult = cfg["reassign_radius_multiplier"]

    for cid in [0, 1, 2, 3]:
        mask = class_ids == cid
        n_pts = mask.sum()
        if n_pts == 0:
            continue

        class_xyz = xyz[mask]
        label = CLASS_ID_TO_LABEL[cid]

        db_params = cfg["dbscan"][label]
        min_clust = cfg["min_cluster_points"][label]
        padding = cfg["bbox_padding"][label]
        merge_cfg = cfg["merge"][label]

        # Voxel downsample if too many points
        downsampled = False
        if n_pts > max_pts:
            class_xyz_ds, _ = voxel_downsample(class_xyz, voxel_size=vox_size)
            downsampled = True
            if verbose:
                print(f"      {label}: {n_pts:,} pts -> ds to "
                      f"{len(class_xyz_ds):,} ... ", end="", flush=True)
        else:
            class_xyz_ds = class_xyz
            if verbose:
                print(f"      {label}: {n_pts:,} pts ... ", end="", flush=True)

        t0 = time.time()
        clustering = DBSCAN(
            eps=db_params["eps"],
            min_samples=db_params["min_samples"],
            algorithm="ball_tree",
        )
        cluster_labels = clustering.fit_predict(class_xyz_ds)
        dt = time.time() - t0

        n_clusters = len(set(cluster_labels) - {-1})

        if verbose:
            n_noise = (cluster_labels == -1).sum()
            print(f"{n_clusters} clusters, {n_noise} noise  [{dt:.2f}s]")

        # ── Merge nearby clusters before fitting bboxes ───────────────
        if merge_cfg["enabled"] and merge_cfg["merge_distance"] > 0 and n_clusters > 1:
            old_n = n_clusters
            cluster_labels = merge_cluster_labels(
                class_xyz_ds, cluster_labels, merge_cfg["merge_distance"]
            )
            n_clusters = len(set(cluster_labels) - {-1})
            if verbose and n_clusters < old_n:
                print(f"        -> merged {old_n} -> {n_clusters} clusters "
                      f"(dist < {merge_cfg['merge_distance']}m)")

        # ── Reassign original points after downsampled clustering ─────
        if downsampled and n_clusters > 0:
            cluster_centers = {}
            for c_id in set(cluster_labels) - {-1}:
                cluster_centers[c_id] = class_xyz_ds[cluster_labels == c_id].mean(axis=0)

            centers_arr = np.array([cluster_centers[c] for c in sorted(cluster_centers)])
            center_ids = sorted(cluster_centers.keys())

            from scipy.spatial import cKDTree
            tree = cKDTree(centers_arr)
            dists, indices = tree.query(class_xyz, k=1)
            full_labels = np.full(len(class_xyz), -1, dtype=np.int32)
            max_reassign = reassign_mult * db_params["eps"]
            close_enough = dists < max_reassign
            full_labels[close_enough] = np.array(center_ids)[indices[close_enough]]

            for c_id in set(full_labels) - {-1}:
                cluster_pts = class_xyz[full_labels == c_id]
                if len(cluster_pts) < min_clust:
                    continue
                bbox = fit_oriented_bbox(cluster_pts, padding=padding)
                bbox["class_ID"] = cid
                bbox["class_label"] = label
                bbox["num_points"] = len(cluster_pts)
                results.append(bbox)
        else:
            for cluster_id in set(cluster_labels) - {-1}:
                cluster_pts = class_xyz_ds[cluster_labels == cluster_id]
                if len(cluster_pts) < min_clust:
                    continue
                bbox = fit_oriented_bbox(cluster_pts, padding=padding)
                bbox["class_ID"] = cid
                bbox["class_label"] = label
                bbox["num_points"] = len(cluster_pts)
                results.append(bbox)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Load a preprocessed frame
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(npz_path):
    """Load a .npz frame -> xyz, reflectivity, rgb, ego_pose."""
    data = np.load(npz_path)
    return data["xyz"], data["reflectivity"], data["rgb"], data["ego_pose"]


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_all(cfg, output_csv, verbose=True):
    """Read manifest, process every frame, write GT CSV."""
    processed_dir = Path(cfg["processed_dir"])
    manifest_path = processed_dir / "manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        print("Run preprocess_frames.py first.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(manifest)} frames from "
          f"{manifest['scene'].nunique()} scenes")

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
        bboxes = extract_bboxes(xyz, rgb, cfg, verbose=verbose,
                                frame_label=frame_label)
        dt = time.time() - t0

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
                print(f"    -> {len(bboxes)} objects ({', '.join(labels)})  "
                      f"[{dt:.2f}s]")
            else:
                print(f"    -> no obstacles  [{dt:.2f}s]")

    if len(all_bboxes) == 0:
        print("No bounding boxes found across all frames.")
        print("Check DBSCAN params — eps might be too small or min_samples too high.")
        sys.exit(1)

    # ── Ensure output directory exists ────────────────────────────────
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(all_bboxes)
    df_out = df_out.rename(columns={
        "center_x": "bbox_center_x", "center_y": "bbox_center_y",
        "center_z": "bbox_center_z", "width": "bbox_width",
        "length": "bbox_length", "height": "bbox_height", "yaw": "bbox_yaw",
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
          f"mean={df_out.groupby(['scene', 'pose_index']).size().mean():.1f}  "
          f"max={df_out.groupby(['scene', 'pose_index']).size().max()}")

    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate GT bounding boxes from preprocessed LiDAR frames"
    )
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (default: built-in defaults)")
    parser.add_argument("--processed-dir", default=None,
                        help="Override config's processed_dir")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: auto-derived from run_name/config)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-frame output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.processed_dir:
        cfg["processed_dir"] = args.processed_dir

    if args.out:
        output_csv = args.out
    else:
        output_csv = derive_output_csv(cfg, args.config)

    # Verbosity: CLI -q wins, otherwise use config value
    verbose = (not args.quiet) and cfg.get("verbose", True)

    print_config(cfg, args.config)

    df = process_all(cfg, output_csv, verbose=verbose)
    return df


if __name__ == "__main__":
    main()