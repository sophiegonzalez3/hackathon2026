"""
leave_one_out_eval.py — Leave-One-Scene-Out Cross-Validation
=============================================================
This is your most important validation script before competition day.

For each scene:
  1. Train on the other 9 scenes (consolidate + ground removal + voxel classifier)
  2. Run full inference pipeline on the held-out scene (no RGB!)
  3. Compare predictions against GT bboxes

This directly simulates the competition's Scene B (unknown environment).

Usage:
    python leave_one_out_eval.py \
        --npz-root processed/ \
        --gt-csv gt_runs/gt_bboxes_run_05_merge_clean.csv \
        --output-dir loso_eval/ \
        --voxel-size 0.5

    # Quick test with just 2 folds instead of 10:
    python leave_one_out_eval.py \
        --npz-root processed/ \
        --gt-csv gt_runs/gt_bboxes_run_05_merge_clean.csv \
        --output-dir loso_eval/ \
        --max-folds 2
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score

from consolidate_scene import consolidate_scene, local_to_world, VoxelGrid
from ground_removal import remove_ground_voxels
from train_voxel_classifier import (
    compute_neighborhood_features, ALL_FEATURES, VOXEL_FEATURES, RANDOM_STATE
)


CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']


# ─────────────────────────────────────────────────────────────
# TRAINING (on N-1 scenes)
# ─────────────────────────────────────────────────────────────
def train_on_scenes(scene_dirs, voxel_size, verbose=True):
    """
    Consolidate multiple scenes, remove ground, train both classifiers.

    Returns:
        clf_stage1, clf_stage2, scaler, imputer, feature_names, ground_info
    """
    all_voxel_dfs = []

    for scene_dir in scene_dirs:
        if verbose:
            print(f"    Consolidating {scene_dir.name}...")

        grid = consolidate_scene(
            scene_dir, voxel_size=voxel_size,
            with_labels=True, verbose=False,
        )
        vdf = grid.to_dataframe(with_labels=True)
        vdf['scene'] = scene_dir.name
        all_voxel_dfs.append(vdf)

    df = pd.concat(all_voxel_dfs, ignore_index=True)
    if verbose:
        print(f"    Total training voxels: {len(df):,}")

    # ── Ground removal (using labels as sanity check) ──
    n_before = len(df)
    df_no_ground, ground_heights = remove_ground_voxels(
        df, method='tile', distance_threshold=1.0,
        min_height_above_ground=1.5, verbose=verbose,
    )

    # ── Neighborhood features ──
    df_no_ground = compute_neighborhood_features(df_no_ground, radius=2)

    # ── Prepare features ──
    feature_names = ALL_FEATURES
    X_raw = df_no_ground[feature_names].values

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_imputed)

    # ── Stage 1: Obstacle vs Background ──
    y_binary = df_no_ground['is_obstacle'].astype(int).values
    n_obs = y_binary.sum()
    n_bg = (1 - y_binary).sum()

    # Balance
    max_bg = min(n_bg, n_obs * 5)
    bg_idx = np.where(y_binary == 0)[0]
    np.random.seed(RANDOM_STATE)
    bg_sample = np.random.choice(bg_idx, size=min(max_bg, len(bg_idx)), replace=False)
    obs_idx = np.where(y_binary == 1)[0]
    train_idx = np.concatenate([obs_idx, bg_sample])

    clf_stage1 = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    clf_stage1.fit(X[train_idx], y_binary[train_idx])

    if verbose:
        print(f"    Stage 1 trained: {n_obs:,} obstacle / {max_bg:,} background samples")

    # ── Stage 2: Obstacle class ──
    obstacle_mask = df_no_ground['is_obstacle'].values
    X_obs = X[obstacle_mask]
    y_class = df_no_ground.loc[obstacle_mask, 'class_id'].astype(int).values

    clf_stage2 = GradientBoostingClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    clf_stage2.fit(X_obs, y_class)

    if verbose:
        print(f"    Stage 2 trained on {len(y_class):,} obstacle voxels")

    return clf_stage1, clf_stage2, scaler, imputer, feature_names, ground_heights


# ─────────────────────────────────────────────────────────────
# INFERENCE (on held-out scene)
# ─────────────────────────────────────────────────────────────
def infer_on_scene(scene_dir, clf_stage1, clf_stage2, scaler, imputer,
                   feature_names, voxel_size, verbose=True):
    """
    Run full inference on one scene WITHOUT using RGB.

    Returns:
        list of bbox dicts
    """
    from inference_pipeline import (
        detect_objects_in_frame, backproject_labels,
    )

    # ── Step A: Consolidate scene (NO labels) ──
    grid = consolidate_scene(
        scene_dir, voxel_size=voxel_size,
        with_labels=False, verbose=False,
    )
    vdf = grid.to_dataframe(with_labels=False)

    if verbose:
        print(f"    Consolidated: {len(vdf):,} voxels")

    # ── Step B: Remove ground ──
    vdf_clean, _ = remove_ground_voxels(
        vdf, method='tile', distance_threshold=1.0,
        min_height_above_ground=1.5, verbose=verbose,
    )

    # ── Step C: Compute features ──
    vdf_clean = compute_neighborhood_features(vdf_clean, radius=2)

    for col in feature_names:
        if col not in vdf_clean.columns:
            vdf_clean[col] = 0.0

    X_raw = vdf_clean[feature_names].values
    X_imputed = imputer.transform(X_raw)
    X = scaler.transform(X_imputed)

    # ── Step D: Classify voxels ──
    y_stage1 = clf_stage1.predict(X)
    obstacle_mask = y_stage1 == 1

    y_class = np.full(len(vdf_clean), -1, dtype=int)
    if obstacle_mask.any():
        y_class[obstacle_mask] = clf_stage2.predict(X[obstacle_mask])

    if verbose:
        n_obs = (y_class >= 0).sum()
        print(f"    Classified: {n_obs:,} obstacle voxels")
        for cid, name in enumerate(CLASS_NAMES):
            print(f"      {name}: {(y_class == cid).sum():,}")

    # ── Step E: Build voxel label lookup ──
    voxel_labels = {}
    for i, (_, row) in enumerate(vdf_clean.iterrows()):
        key = (int(row['vx']), int(row['vy']), int(row['vz']))
        voxel_labels[key] = int(y_class[i])

    # ── Step F: Per-frame detection ──
    npz_files = sorted(scene_dir.glob('frame_*.npz'))
    all_bboxes = []

    for frame_idx, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        xyz_local = data['xyz']
        ego_pose = data['ego_pose']

        point_labels = backproject_labels(xyz_local, ego_pose, voxel_labels, voxel_size)
        bboxes = detect_objects_in_frame(xyz_local, point_labels)

        for b in bboxes:
            b['scene'] = scene_dir.name
            b['pose_index'] = frame_idx
            b['frame_file'] = str(npz_path)
            b['ego_x'] = ego_pose[0]
            b['ego_y'] = ego_pose[1]
            b['ego_z'] = ego_pose[2]
            b['ego_yaw'] = ego_pose[3]

        all_bboxes.extend(bboxes)

    if verbose:
        print(f"    Detected {len(all_bboxes)} objects across {len(npz_files)} frames")

    return all_bboxes


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def compute_iou_3d_aabb(box_a, box_b):
    """Axis-aligned 3D IoU between two bboxes."""
    def corners(b, prefix='bbox_'):
        cx = b[f'{prefix}center_x']
        cy = b[f'{prefix}center_y']
        cz = b[f'{prefix}center_z']
        w = b[f'{prefix}width'] / 2
        l = b[f'{prefix}length'] / 2
        h = b[f'{prefix}height'] / 2
        return (cx-w, cy-l, cz-h, cx+w, cy+l, cz+h)

    a = corners(box_a)
    b = corners(box_b)

    x1, y1, z1 = max(a[0],b[0]), max(a[1],b[1]), max(a[2],b[2])
    x2, y2, z2 = min(a[3],b[3]), min(a[4],b[4]), min(a[5],b[5])

    if x2<=x1 or y2<=y1 or z2<=z1:
        return 0.0

    inter = (x2-x1)*(y2-y1)*(z2-z1)
    vol_a = (a[3]-a[0])*(a[4]-a[1])*(a[5]-a[2])
    vol_b = (b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2])
    return inter / (vol_a + vol_b - inter + 1e-10)


def evaluate_scene(pred_bboxes, gt_df, scene_name, iou_threshold=0.5):
    """Evaluate predictions for one scene against GT."""
    # Convert predictions to DataFrame
    if not pred_bboxes:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_df), 'per_class': {}}

    pred_df = pd.DataFrame(pred_bboxes)
    pred_df = pred_df.rename(columns={
        'center_x': 'bbox_center_x', 'center_y': 'bbox_center_y',
        'center_z': 'bbox_center_z', 'width': 'bbox_width',
        'length': 'bbox_length', 'height': 'bbox_height', 'yaw': 'bbox_yaw',
    })

    scene_gt = gt_df[gt_df['scene'] == scene_name]

    results = {'tp': 0, 'fp': 0, 'fn': 0, 'per_class': {}}
    for cls in CLASS_NAMES:
        results['per_class'][cls] = {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}

    # Match per frame
    for (scene, pose), gt_frame in scene_gt.groupby(['scene', 'pose_index']):
        pred_frame = pred_df[
            (pred_df.get('scene', '') == scene) &
            (pred_df.get('pose_index', -1) == pose)
        ] if 'scene' in pred_df.columns else pd.DataFrame()

        gt_matched = set()
        pred_matched = set()

        for pi, pred_row in pred_frame.iterrows():
            best_iou = 0
            best_gi = None
            for gi, gt_row in gt_frame.iterrows():
                if gi in gt_matched:
                    continue
                if gt_row['class_label'] != pred_row.get('class_label', ''):
                    continue
                iou = compute_iou_3d_aabb(gt_row, pred_row)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi is not None:
                cls = gt_frame.loc[best_gi, 'class_label']
                results['per_class'][cls]['tp'] += 1
                results['per_class'][cls]['ious'].append(best_iou)
                results['tp'] += 1
                gt_matched.add(best_gi)
                pred_matched.add(pi)

        for pi, pred_row in pred_frame.iterrows():
            if pi not in pred_matched:
                cls = pred_row.get('class_label', 'Unknown')
                if cls in results['per_class']:
                    results['per_class'][cls]['fp'] += 1
                results['fp'] += 1

        for gi, gt_row in gt_frame.iterrows():
            if gi not in gt_matched:
                cls = gt_row['class_label']
                results['per_class'][cls]['fn'] += 1
                results['fn'] += 1

    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Leave-One-Scene-Out evaluation')
    parser.add_argument('--npz-root', type=str, default='processed/')
    parser.add_argument('--gt-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='loso_eval/')
    parser.add_argument('--voxel-size', type=float, default=0.5)
    parser.add_argument('--max-folds', type=int, default=10,
                        help='Max number of folds to run (for quick testing)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find scenes
    npz_root = Path(args.npz_root)
    scene_dirs = sorted([d for d in npz_root.iterdir()
                         if d.is_dir() and d.name.startswith('scene_')])

    if not scene_dirs:
        raise FileNotFoundError(f"No scene_* dirs in {npz_root}")

    n_folds = min(len(scene_dirs), args.max_folds)
    print(f"Leave-One-Scene-Out: {n_folds} folds from {len(scene_dirs)} scenes")

    # Load GT
    gt_df = pd.read_csv(args.gt_csv)
    print(f"Ground truth: {len(gt_df)} bboxes")

    all_results = []

    for fold_idx in range(n_folds):
        test_scene = scene_dirs[fold_idx]
        train_scenes = [d for i, d in enumerate(scene_dirs) if i != fold_idx]

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{n_folds}: Test on {test_scene.name}")
        print(f"  Training on: {[d.name for d in train_scenes]}")
        print("=" * 70)

        t0 = time.time()

        # ── Train ──
        print("\n  TRAINING:")
        clf1, clf2, scaler, imputer, feat_names, ground_h = train_on_scenes(
            train_scenes, args.voxel_size, verbose=True
        )

        # ── Inference on held-out scene ──
        print(f"\n  INFERENCE on {test_scene.name}:")
        pred_bboxes = infer_on_scene(
            test_scene, clf1, clf2, scaler, imputer, feat_names,
            args.voxel_size, verbose=True
        )

        # ── Evaluate ──
        scene_results = evaluate_scene(pred_bboxes, gt_df, test_scene.name)
        scene_results['scene'] = test_scene.name
        scene_results['n_predictions'] = len(pred_bboxes)
        scene_results['time'] = time.time() - t0
        all_results.append(scene_results)

        # Print per-scene results
        tp = scene_results['tp']
        fp = scene_results['fp']
        fn = scene_results['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*p*r / (p+r) if (p+r) > 0 else 0

        print(f"\n  RESULTS for {test_scene.name}:")
        print(f"    TP={tp}  FP={fp}  FN={fn}")
        print(f"    Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
        print(f"    Time: {scene_results['time']:.1f}s")

        for cls in CLASS_NAMES:
            cr = scene_results['per_class'][cls]
            cls_tp, cls_fp, cls_fn = cr['tp'], cr['fp'], cr['fn']
            cls_p = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
            cls_r = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0
            cls_f1 = 2*cls_p*cls_r / (cls_p+cls_r) if (cls_p+cls_r) > 0 else 0
            avg_iou = np.mean(cr['ious']) if cr['ious'] else 0
            print(f"      {cls:20s}: P={cls_p:.3f} R={cls_r:.3f} F1={cls_f1:.3f} "
                  f"meanIoU={avg_iou:.3f} (TP={cls_tp} FP={cls_fp} FN={cls_fn})")

    # ── Aggregate results ──
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (Leave-One-Scene-Out)")
    print("=" * 70)

    total_tp = sum(r['tp'] for r in all_results)
    total_fp = sum(r['fp'] for r in all_results)
    total_fn = sum(r['fn'] for r in all_results)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2*overall_p*overall_r / (overall_p+overall_r) if (overall_p+overall_r) > 0 else 0

    print(f"\n  Overall: TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision={overall_p:.3f}  Recall={overall_r:.3f}  F1={overall_f1:.3f}")

    for cls in CLASS_NAMES:
        cls_tp = sum(r['per_class'][cls]['tp'] for r in all_results)
        cls_fp = sum(r['per_class'][cls]['fp'] for r in all_results)
        cls_fn = sum(r['per_class'][cls]['fn'] for r in all_results)
        cls_p = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
        cls_r = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0
        cls_f1 = 2*cls_p*cls_r / (cls_p+cls_r) if (cls_p+cls_r) > 0 else 0

        all_ious = []
        for r in all_results:
            all_ious.extend(r['per_class'][cls]['ious'])
        avg_iou = np.mean(all_ious) if all_ious else 0

        print(f"  {cls:20s}: P={cls_p:.3f} R={cls_r:.3f} F1={cls_f1:.3f} "
              f"meanIoU={avg_iou:.3f}")

    # Save summary
    summary = {
        'n_folds': n_folds,
        'voxel_size': args.voxel_size,
        'overall_precision': overall_p,
        'overall_recall': overall_r,
        'overall_f1': overall_f1,
        'per_scene': [{
            'scene': r['scene'],
            'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'],
            'n_predictions': r['n_predictions'],
            'time': r['time'],
        } for r in all_results],
    }
    with open(output_dir / 'loso_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {output_dir / 'loso_results.json'}")


if __name__ == '__main__':
    main()
