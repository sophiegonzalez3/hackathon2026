"""
inference_pipeline.py — Full end-to-end inference
===================================================
1. Consolidate eval frames into world-frame voxel grid
2. Classify each voxel (Stage 1: obstacle/bg, Stage 2: class)
3. Back-project voxel labels to individual frame points
4. Per-class DBSCAN clustering in local frame
5. Fit oriented 3D bounding boxes
6. Output competition CSV

Usage:
    python inference_pipeline.py \
        --npz-root eval_data/ \
        --model-dir models_v2/ \
        --output-csv predictions.csv \
        --voxel-size 0.5

    # For training data (self-evaluation with GT comparison):
    python inference_pipeline.py \
        --npz-root processed/ \
        --model-dir models_v2/ \
        --output-csv self_eval.csv \
        --voxel-size 0.5
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from consolidate_scene import (
    VoxelGrid, consolidate_scene, local_to_world,
)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

# DBSCAN parameters per class (tuned from your config 5)
# These run on PREDICTED class segments, not GT
DBSCAN_PARAMS = {
    0: {'eps': 3.0, 'min_samples': 10},   # Antenna
    1: {'eps': 5.0, 'min_samples': 5},    # Cable
    2: {'eps': 2.0, 'min_samples': 10},   # Electric Pole
    3: {'eps': 5.0, 'min_samples': 15},   # Wind Turbine
}

MIN_CLUSTER_POINTS = {
    0: 10,   # Antenna
    1: 8,    # Cable
    2: 10,   # Electric Pole
    3: 15,   # Wind Turbine
}


# ─────────────────────────────────────────────────────────────
# LOAD TRAINED MODELS
# ─────────────────────────────────────────────────────────────
def load_models(model_dir):
    """Load both stage classifiers and preprocessing objects."""
    model_dir = Path(model_dir)

    with open(model_dir / 'stage1_model.pkl', 'rb') as f:
        clf_stage1 = pickle.load(f)
    with open(model_dir / 'stage2_model.pkl', 'rb') as f:
        clf_stage2 = pickle.load(f)
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    print(f"  Loaded 2-stage classifier")
    print(f"  Features: {len(config['feature_names'])}")
    return clf_stage1, clf_stage2, scaler, imputer, config


# ─────────────────────────────────────────────────────────────
# VOXEL CLASSIFICATION
# ─────────────────────────────────────────────────────────────
def classify_voxels(grid, clf_stage1, clf_stage2, scaler, imputer, config):
    """
    Classify all voxels in a consolidated grid.

    Returns:
        voxel_labels: dict mapping (ix, iy, iz) -> class_id (0-3) or -1 (background)
        voxel_proba:  dict mapping (ix, iy, iz) -> confidence score
    """
    # Convert grid to DataFrame
    df = grid.to_dataframe(with_labels=False)

    if len(df) == 0:
        return {}, {}

    # Compute neighborhood features (same as training)
    # Import the function from train module
    from train_voxel_classifier import compute_neighborhood_features, ALL_FEATURES
    df = compute_neighborhood_features(df, radius=2)

    # Extract features
    feature_names = config['feature_names']
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    X_raw = df[feature_names].values

    X_imputed = imputer.transform(X_raw)
    X = scaler.transform(X_imputed)

    # Stage 1: Obstacle vs Background
    y_stage1 = clf_stage1.predict(X)
    if hasattr(clf_stage1, 'predict_proba'):
        proba_stage1 = clf_stage1.predict_proba(X)[:, 1]  # P(obstacle)
    else:
        proba_stage1 = y_stage1.astype(float)

    # Stage 2: Classify obstacles
    obstacle_mask = y_stage1 == 1
    y_class = np.full(len(df), -1, dtype=int)
    confidence = np.zeros(len(df))

    if obstacle_mask.any():
        X_obstacle = X[obstacle_mask]
        y_class[obstacle_mask] = clf_stage2.predict(X_obstacle)

        if hasattr(clf_stage2, 'predict_proba'):
            proba_stage2 = clf_stage2.predict_proba(X_obstacle)
            confidence[obstacle_mask] = proba_stage2.max(axis=1)
        else:
            confidence[obstacle_mask] = 1.0

    # Build lookup dicts
    voxel_labels = {}
    voxel_proba = {}
    for i, (_, row) in enumerate(df.iterrows()):
        key = (int(row['vx']), int(row['vy']), int(row['vz']))
        voxel_labels[key] = int(y_class[i])
        voxel_proba[key] = float(confidence[i])

    n_obstacle = (y_class >= 0).sum()
    n_bg = (y_class < 0).sum()
    print(f"  Classified: {n_obstacle:,} obstacle voxels, {n_bg:,} background")
    if n_obstacle > 0:
        for cid, name in enumerate(CLASS_NAMES):
            print(f"    {name}: {(y_class == cid).sum():,} voxels")

    return voxel_labels, voxel_proba


# ─────────────────────────────────────────────────────────────
# BACK-PROJECTION: voxel labels → per-point labels
# ─────────────────────────────────────────────────────────────
def backproject_labels(xyz_local, ego_pose, voxel_labels, voxel_size):
    """
    For each point in a frame, look up its world-frame voxel
    and return the predicted class label.

    Args:
        xyz_local: (N, 3) points in local frame
        ego_pose: (4,) ego pose in raw units
        voxel_labels: dict (ix,iy,iz) -> class_id
        voxel_size: float

    Returns:
        labels: (N,) int array, class_id per point (-1 = background/unknown)
    """
    from consolidate_scene import local_to_world

    xyz_world = local_to_world(xyz_local, ego_pose)
    indices = np.floor(xyz_world / voxel_size).astype(np.int32)

    labels = np.full(len(xyz_local), -1, dtype=np.int32)
    for i in range(len(indices)):
        key = (int(indices[i, 0]), int(indices[i, 1]), int(indices[i, 2]))
        if key in voxel_labels:
            labels[i] = voxel_labels[key]

    return labels


# ─────────────────────────────────────────────────────────────
# ORIENTED BBOX FITTING (same as your generate_gt_bboxes.py)
# ─────────────────────────────────────────────────────────────
def fit_oriented_bbox(points_3d, padding=0.0):
    """Fit a yaw-only oriented 3D bounding box via PCA on XY."""
    pts = np.asarray(points_3d, dtype=np.float64)

    if len(pts) < 3:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = (mn + mx) / 2
        dims = np.maximum(mx - mn, 0.1) + 2 * padding
        return {
            'center_x': center[0], 'center_y': center[1], 'center_z': center[2],
            'width': dims[0], 'length': dims[1], 'height': dims[2],
            'yaw': 0.0,
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
        'center_x': center[0], 'center_y': center[1], 'center_z': center[2],
        'width': dims[0], 'length': dims[1], 'height': dims[2],
        'yaw': yaw,
    }


# ─────────────────────────────────────────────────────────────
# PER-FRAME DETECTION: labels → DBSCAN → bboxes
# ─────────────────────────────────────────────────────────────
def detect_objects_in_frame(xyz_local, point_labels):
    """
    Given per-point predicted class labels, run per-class DBSCAN
    and fit oriented bounding boxes.

    Args:
        xyz_local: (N, 3) points in local frame
        point_labels: (N,) predicted class IDs (-1 = background)

    Returns:
        list of bbox dicts
    """
    results = []

    for cid in range(4):
        mask = point_labels == cid
        n_pts = mask.sum()
        if n_pts == 0:
            continue

        class_xyz = xyz_local[mask]
        params = DBSCAN_PARAMS[cid]
        min_pts = MIN_CLUSTER_POINTS[cid]

        # DBSCAN clustering
        clustering = DBSCAN(
            eps=params['eps'],
            min_samples=params['min_samples'],
            algorithm='ball_tree',
        )
        cluster_labels = clustering.fit_predict(class_xyz)

        for cluster_id in set(cluster_labels) - {-1}:
            cluster_pts = class_xyz[cluster_labels == cluster_id]
            if len(cluster_pts) < min_pts:
                continue

            bbox = fit_oriented_bbox(cluster_pts)
            bbox['class_ID'] = cid
            bbox['class_label'] = CLASS_NAMES[cid]
            bbox['num_points'] = len(cluster_pts)
            results.append(bbox)

    return results


# ─────────────────────────────────────────────────────────────
# PROCESS ONE SCENE
# ─────────────────────────────────────────────────────────────
def process_scene(scene_dir, voxel_labels, voxel_size, verbose=True):
    """
    Process all frames in a scene: back-project labels, detect objects.

    Returns:
        list of bbox dicts (with ego_pose and frame info attached)
    """
    scene_dir = Path(scene_dir)
    npz_files = sorted(scene_dir.glob('frame_*.npz'))
    scene_name = scene_dir.name

    all_bboxes = []

    for frame_idx, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        xyz_local = data['xyz']
        ego_pose = data['ego_pose']

        # Back-project voxel labels to this frame's points
        point_labels = backproject_labels(xyz_local, ego_pose, voxel_labels, voxel_size)

        n_labeled = (point_labels >= 0).sum()

        # Detect objects
        bboxes = detect_objects_in_frame(xyz_local, point_labels)

        # Attach frame metadata
        for b in bboxes:
            b['scene'] = scene_name
            b['pose_index'] = frame_idx
            b['frame_file'] = str(npz_path)
            b['ego_x'] = ego_pose[0]
            b['ego_y'] = ego_pose[1]
            b['ego_z'] = ego_pose[2]
            b['ego_yaw'] = ego_pose[3]

        all_bboxes.extend(bboxes)

        if verbose and (frame_idx + 1) % 20 == 0:
            print(f"    [{frame_idx + 1}/{len(npz_files)}] "
                  f"{n_labeled:,} labeled points, {len(bboxes)} detections")

    if verbose:
        print(f"  {scene_name}: {len(all_bboxes)} total detections "
              f"across {len(npz_files)} frames")

    return all_bboxes


# ─────────────────────────────────────────────────────────────
# MAIN INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────
def run_inference(npz_root, model_dir, output_csv, voxel_size=0.5, verbose=True):
    """
    Full inference pipeline.
    """
    npz_root = Path(npz_root)
    t_start = time.time()

    # ── Load models ──
    print("=" * 70)
    print("STEP 1: Loading trained models")
    print("=" * 70)
    clf_stage1, clf_stage2, scaler, imputer, config = load_models(model_dir)

    # ── Find scenes ──
    scene_dirs = sorted([d for d in npz_root.iterdir()
                         if d.is_dir() and d.name.startswith('scene')])
    if not scene_dirs:
        raise FileNotFoundError(f"No scene directories in {npz_root}")
    print(f"\n  Found {len(scene_dirs)} scenes")

    all_bboxes = []

    for scene_dir in scene_dirs:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING: {scene_dir.name}")
        print("=" * 70)

        # ── Consolidate scene ──
        print("\n  Step A: Consolidating frames into voxel grid...")
        t0 = time.time()
        grid = consolidate_scene(
            scene_dir, voxel_size=voxel_size,
            with_labels=False,  # No RGB labels at inference time
            verbose=verbose,
        )
        print(f"  Consolidation: {len(grid.voxels):,} voxels [{time.time()-t0:.1f}s]")

        # ── Classify voxels ──
        print("\n  Step B: Classifying voxels...")
        t0 = time.time()
        voxel_labels, voxel_proba = classify_voxels(
            grid, clf_stage1, clf_stage2, scaler, imputer, config
        )
        print(f"  Classification: [{time.time()-t0:.1f}s]")

        # ── Process each frame ──
        print("\n  Step C: Detecting objects per frame...")
        t0 = time.time()
        scene_bboxes = process_scene(scene_dir, voxel_labels, voxel_size, verbose=verbose)
        all_bboxes.extend(scene_bboxes)
        print(f"  Detection: {len(scene_bboxes)} objects [{time.time()-t0:.1f}s]")

    # ── Build output CSV ──
    print(f"\n{'=' * 70}")
    print("WRITING OUTPUT")
    print("=" * 70)

    if not all_bboxes:
        print("  WARNING: No detections produced!")
        return pd.DataFrame()

    df = pd.DataFrame(all_bboxes)
    df = df.rename(columns={
        'center_x': 'bbox_center_x', 'center_y': 'bbox_center_y',
        'center_z': 'bbox_center_z', 'width': 'bbox_width',
        'length': 'bbox_length', 'height': 'bbox_height', 'yaw': 'bbox_yaw',
    })

    # Competition format columns
    csv_cols = [
        'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
        'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
        'bbox_width', 'bbox_length', 'bbox_height', 'bbox_yaw',
        'class_ID', 'class_label',
    ]
    output_df = df[[c for c in csv_cols if c in df.columns]]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    elapsed = time.time() - t_start

    print(f"\n  Total detections: {len(output_df)}")
    print(f"  Class breakdown:")
    print(output_df['class_label'].value_counts().to_string())
    print(f"\n  Saved: {output_path}")
    print(f"  Total time: {elapsed:.1f}s")

    return output_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Run full inference pipeline')
    parser.add_argument('--npz-root', type=str, required=True,
                        help='Root directory with scene_*/frame_*.npz')
    parser.add_argument('--model-dir', type=str, default='models_v2/',
                        help='Directory with trained models')
    parser.add_argument('--output-csv', type=str, default='predictions.csv',
                        help='Output CSV path')
    parser.add_argument('--voxel-size', type=float, default=0.5,
                        help='Voxel size in meters (must match training)')
    args = parser.parse_args()

    run_inference(
        args.npz_root, args.model_dir, args.output_csv,
        voxel_size=args.voxel_size,
    )


if __name__ == '__main__':
    main()
