"""
self_evaluate.py — Run the full pipeline on training data and compare with GT
==============================================================================
This is your sanity check before the competition eval day.

Usage:
    python self_evaluate.py \
        --npz-root processed/ \
        --model-dir models_v2/ \
        --gt-csv gt_runs/gt_bboxes_run_05_merge_clean.csv \
        --voxel-size 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from inference_pipeline import run_inference


CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']


def compute_iou_3d_aabb(box_a, box_b):
    """
    Compute axis-aligned IoU between two boxes.
    Each box: dict with bbox_center_x/y/z, bbox_width/length/height.
    Simplified (ignores yaw) — gives a lower bound on true IoU.
    """
    def get_corners(b):
        cx, cy, cz = b['bbox_center_x'], b['bbox_center_y'], b['bbox_center_z']
        w, l, h = b['bbox_width'] / 2, b['bbox_length'] / 2, b['bbox_height'] / 2
        return (cx - w, cy - l, cz - h, cx + w, cy + l, cz + h)

    a = get_corners(box_a)
    b = get_corners(box_b)

    # Intersection
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    z1 = max(a[2], b[2])
    x2 = min(a[3], b[3])
    y2 = min(a[4], b[4])
    z2 = min(a[5], b[5])

    if x2 <= x1 or y2 <= y1 or z2 <= z1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1) * (z2 - z1)

    vol_a = (a[3]-a[0]) * (a[4]-a[1]) * (a[5]-a[2])
    vol_b = (b[3]-b[0]) * (b[4]-b[1]) * (b[5]-b[2])

    union = vol_a + vol_b - inter
    return inter / (union + 1e-10)


def match_predictions(gt_df, pred_df, iou_threshold=0.5):
    """
    Match predictions to ground truth using greedy IoU matching.

    Returns matched pairs, false positives, and false negatives per frame.
    """
    # Group by frame
    gt_groups = gt_df.groupby(['scene', 'pose_index'])
    pred_groups = pred_df.groupby(['scene', 'pose_index']) if 'scene' in pred_df.columns \
        else pred_df.groupby(['ego_x', 'ego_y', 'ego_z', 'ego_yaw'])

    all_tp = []  # (gt_row, pred_row, iou)
    all_fp = []  # pred_rows with no match
    all_fn = []  # gt_rows with no match

    for frame_key, gt_frame in gt_groups:
        scene, pose = frame_key

        # Find matching pred frame
        pred_frame = None
        if 'scene' in pred_df.columns:
            mask = (pred_df['scene'] == scene) & (pred_df['pose_index'] == pose)
            pred_frame = pred_df[mask]
        
        if pred_frame is None or len(pred_frame) == 0:
            all_fn.extend(gt_frame.to_dict('records'))
            continue

        gt_matched = set()
        pred_matched = set()

        # Compute IoU matrix
        for pi, pred_row in pred_frame.iterrows():
            best_iou = 0
            best_gi = None
            for gi, gt_row in gt_frame.iterrows():
                if gi in gt_matched:
                    continue
                if gt_row['class_label'] != pred_row['class_label']:
                    continue
                iou = compute_iou_3d_aabb(gt_row, pred_row)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi is not None:
                all_tp.append({
                    'gt_class': gt_frame.loc[best_gi, 'class_label'],
                    'pred_class': pred_row['class_label'],
                    'iou': best_iou,
                })
                gt_matched.add(best_gi)
                pred_matched.add(pi)

        # Unmatched predictions = false positives
        for pi, pred_row in pred_frame.iterrows():
            if pi not in pred_matched:
                all_fp.append({'pred_class': pred_row['class_label']})

        # Unmatched GTs = false negatives
        for gi, gt_row in gt_frame.iterrows():
            if gi not in gt_matched:
                all_fn.append({'gt_class': gt_row['class_label']})

    return all_tp, all_fp, all_fn


def print_evaluation(all_tp, all_fp, all_fn):
    """Print AP-style evaluation metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (IoU >= 0.5)")
    print("=" * 70)

    for cls in CLASS_NAMES:
        tp = sum(1 for t in all_tp if t['gt_class'] == cls)
        fp = sum(1 for f in all_fp if f['pred_class'] == cls)
        fn = sum(1 for f in all_fn if f.get('gt_class') == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        avg_iou = np.mean([t['iou'] for t in all_tp if t['gt_class'] == cls]) \
            if any(t['gt_class'] == cls for t in all_tp) else 0

        print(f"\n  {cls}:")
        print(f"    TP={tp}  FP={fp}  FN={fn}")
        print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
        print(f"    Mean IoU (matched): {avg_iou:.3f}")

    # Overall
    total_tp = len(all_tp)
    total_fp = len(all_fp)
    total_fn = len(all_fn)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) \
        if (overall_p + overall_r) > 0 else 0
    mean_iou = np.mean([t['iou'] for t in all_tp]) if all_tp else 0

    print(f"\n  OVERALL:")
    print(f"    TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"    Precision={overall_p:.3f}  Recall={overall_r:.3f}  F1={overall_f1:.3f}")
    print(f"    Mean IoU (matched): {mean_iou:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Self-evaluate on training data')
    parser.add_argument('--npz-root', type=str, default='processed/')
    parser.add_argument('--model-dir', type=str, default='models_v2/')
    parser.add_argument('--gt-csv', type=str, required=True,
                        help='Ground truth bbox CSV')
    parser.add_argument('--voxel-size', type=float, default=0.5)
    parser.add_argument('--output-csv', type=str, default='self_eval_predictions.csv')
    args = parser.parse_args()

    # Run inference on training data
    pred_df = run_inference(
        args.npz_root, args.model_dir, args.output_csv,
        voxel_size=args.voxel_size,
    )

    # Load GT
    gt_df = pd.read_csv(args.gt_csv)
    print(f"\nGround truth: {len(gt_df)} bboxes")
    print(f"Predictions:  {len(pred_df)} bboxes")

    # Match and evaluate
    all_tp, all_fp, all_fn = match_predictions(gt_df, pred_df)
    print_evaluation(all_tp, all_fp, all_fn)


if __name__ == '__main__':
    main()
