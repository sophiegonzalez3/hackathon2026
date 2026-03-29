"""
Clustering Parameter Tuner & Analyzer
======================================
Helps find optimal HDBSCAN parameters and analyzes clustering quality
against ground truth.

Usage:
    # Quick test on a few frames
    python cluster_tuning.py --cleaned-dir cleaned/ --gt-csv bboxes_train.csv --n-frames 10

    # Grid search over parameters
    python cluster_tuning.py --cleaned-dir cleaned/ --gt-csv bboxes_train.csv --grid-search

    # Analyze specific parameter set
    python cluster_tuning.py --cleaned-dir cleaned/ --gt-csv bboxes_train.csv \
        --min-cluster-size 30 --min-samples 5 --analyze
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    import hdbscan
except ImportError:
    print("pip install hdbscan")
    exit(1)

# Import from main script
from cluster_detection import (
    cluster_frame_v2, fit_obb_minarea, extract_bbox_features,
    CLASS_NAMES, CLASS_RGB, DEFAULT_CONFIG
)


# ═══════════════════════════════════════════════════════════════════════════════
# IOU COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def bbox_to_corners(center, length, width, height, yaw):
    """Convert bbox params to 8 corner points."""
    # Half dimensions
    l, w, h = length/2, width/2, height/2
    
    # Corners in local frame
    corners_local = np.array([
        [-l, -w, -h], [-l, -w, +h], [-l, +w, -h], [-l, +w, +h],
        [+l, -w, -h], [+l, -w, +h], [+l, +w, -h], [+l, +w, +h],
    ])
    
    # Rotation matrix (around Z)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # Transform to world
    corners_world = corners_local @ R.T + center
    return corners_world


def iou_3d_approx(box1: dict, box2: dict) -> float:
    """
    Approximate 3D IoU using axis-aligned bounding boxes after alignment.
    For exact OBB IoU, you'd need convex hull intersection - this is faster.
    """
    # Get centers and dimensions
    c1 = np.array([box1['bbox_center_x'], box1['bbox_center_y'], box1['bbox_center_z']])
    c2 = np.array([box2['bbox_center_x'], box2['bbox_center_y'], box2['bbox_center_z']])
    
    # Use max of length/width as "size" in XY (orientation-agnostic)
    size1_xy = max(box1['bbox_length'], box1['bbox_width'])
    size2_xy = max(box2['bbox_length'], box2['bbox_width'])
    
    h1, h2 = box1['bbox_height'], box2['bbox_height']
    
    # Approximate as cylinders / axis-aligned
    # XY distance
    dist_xy = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    
    # XY overlap (circular approximation)
    r1, r2 = size1_xy/2, size2_xy/2
    if dist_xy >= r1 + r2:
        xy_overlap = 0
    elif dist_xy <= abs(r1 - r2):
        xy_overlap = np.pi * min(r1, r2)**2
    else:
        # Lens-shaped overlap
        d = dist_xy
        part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2)/(2*d*r1))
        part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2)/(2*d*r2))
        part3 = 0.5 * np.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
        xy_overlap = part1 + part2 - part3
    
    # Z overlap
    z1_min, z1_max = c1[2] - h1/2, c1[2] + h1/2
    z2_min, z2_max = c2[2] - h2/2, c2[2] + h2/2
    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    
    # Volumes
    v1 = np.pi * r1**2 * h1
    v2 = np.pi * r2**2 * h2
    intersection = xy_overlap * z_overlap
    union = v1 + v2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def iou_3d_box(box1: dict, box2: dict) -> float:
    """
    Simple axis-aligned 3D IoU (ignoring rotation).
    Use this for a faster approximation.
    """
    c1 = np.array([box1['bbox_center_x'], box1['bbox_center_y'], box1['bbox_center_z']])
    c2 = np.array([box2['bbox_center_x'], box2['bbox_center_y'], box2['bbox_center_z']])
    
    # Half-sizes (use max of l/w for rotation-invariant comparison)
    s1 = np.array([max(box1['bbox_length'], box1['bbox_width'])/2,
                   min(box1['bbox_length'], box1['bbox_width'])/2,
                   box1['bbox_height']/2])
    s2 = np.array([max(box2['bbox_length'], box2['bbox_width'])/2,
                   min(box2['bbox_length'], box2['bbox_width'])/2,
                   box2['bbox_height']/2])
    
    # AABB intersection
    min1, max1 = c1 - s1, c1 + s1
    min2, max2 = c2 - s2, c2 + s2
    
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(0, inter_max - inter_min)
    
    inter_vol = inter_size[0] * inter_size[1] * inter_size[2]
    vol1 = (2*s1[0]) * (2*s1[1]) * (2*s1[2])
    vol2 = (2*s2[0]) * (2*s2[1]) * (2*s2[2])
    
    union = vol1 + vol2 - inter_vol
    
    if union <= 0:
        return 0.0
    
    return inter_vol / union


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def match_detections_to_gt(
    detections: List[dict],
    gt_boxes: List[dict],
    iou_threshold: float = 0.3,
) -> Tuple[int, int, int, float]:
    """
    Match detections to GT using Hungarian algorithm.
    
    Returns:
        (TP, FP, FN, mean_iou)
    """
    if len(detections) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, 0.0
    
    if len(detections) == 0:
        return 0, 0, len(gt_boxes), 0.0
    
    if len(gt_boxes) == 0:
        return 0, len(detections), 0, 0.0
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(detections), len(gt_boxes)))
    for i, det in enumerate(detections):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = iou_3d_box(det, gt)
    
    # Hungarian matching (maximize IoU = minimize -IoU)
    det_idx, gt_idx = linear_sum_assignment(-iou_matrix)
    
    # Count matches above threshold
    tp = 0
    matched_ious = []
    matched_det = set()
    matched_gt = set()
    
    for d, g in zip(det_idx, gt_idx):
        if iou_matrix[d, g] >= iou_threshold:
            tp += 1
            matched_det.add(d)
            matched_gt.add(g)
            matched_ious.append(iou_matrix[d, g])
    
    fp = len(detections) - tp
    fn = len(gt_boxes) - tp
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    
    return tp, fp, fn, mean_iou


def evaluate_clustering(
    cleaned_dir: Path,
    gt_csv: Path,
    config: dict,
    n_frames: int = None,
) -> dict:
    """
    Evaluate clustering against ground truth.
    """
    cleaned_dir = Path(cleaned_dir)
    
    # Load GT
    gt_df = pd.read_csv(gt_csv)
    
    # Find frames
    npz_files = sorted(cleaned_dir.glob("**/*.npz"))
    if n_frames:
        npz_files = npz_files[:n_frames]
    
    # Aggregate stats
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []
    per_class = {c: {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []} for c in CLASS_NAMES}
    
    total_clusters = 0
    total_gt_objects = 0
    
    for npz_path in tqdm(npz_files, desc="Evaluating", leave=False):
        scene = npz_path.parent.name
        pose_index = int(npz_path.stem.split('_')[1])
        
        # Load frame
        data = np.load(npz_path)
        xyz = data['xyz'].astype(np.float32)
        rgb = data.get('rgb', None)
        
        # Get GT for this frame
        frame_gt = gt_df[(gt_df['scene'] == scene) & (gt_df['pose_index'] == pose_index)]
        gt_boxes = frame_gt.to_dict('records')
        total_gt_objects += len(gt_boxes)
        
        # Cluster
        detections = cluster_frame(xyz, rgb, config)
        total_clusters += len(detections)
        
        # Match overall
        tp, fp, fn, mean_iou = match_detections_to_gt(detections, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if mean_iou > 0:
            all_ious.append(mean_iou)
        
        # Per-class matching (based on GT class)
        for cls in CLASS_NAMES:
            cls_gt = [g for g in gt_boxes if g['class_label'] == cls]
            cls_det = [d for d in detections if d.get('gt_class') == cls]
            
            ctp, cfp, cfn, ciou = match_detections_to_gt(cls_det, cls_gt)
            per_class[cls]['tp'] += ctp
            per_class[cls]['fp'] += cfp
            per_class[cls]['fn'] += cfn
            if ciou > 0:
                per_class[cls]['ious'].append(ciou)
    
    # Compute metrics
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    results = {
        'config': config,
        'n_frames': len(npz_files),
        'total_clusters': total_clusters,
        'total_gt_objects': total_gt_objects,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': np.mean(all_ious) if all_ious else 0,
        'per_class': {},
    }
    
    for cls in CLASS_NAMES:
        c = per_class[cls]
        cp = c['tp'] / max(c['tp'] + c['fp'], 1)
        cr = c['tp'] / max(c['tp'] + c['fn'], 1)
        cf1 = 2 * cp * cr / max(cp + cr, 1e-6)
        results['per_class'][cls] = {
            'precision': cp,
            'recall': cr,
            'f1': cf1,
            'mean_iou': np.mean(c['ious']) if c['ious'] else 0,
        }
    
    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("CLUSTERING EVALUATION RESULTS")
    print("="*60)
    
    cfg = results['config']
    print(f"\nConfig: min_cluster_size={cfg['min_cluster_size']}, "
          f"min_samples={cfg['min_samples']}, "
          f"epsilon={cfg.get('cluster_selection_epsilon', 0)}")
    
    print(f"\nFrames: {results['n_frames']}")
    print(f"Total clusters: {results['total_clusters']} "
          f"({results['total_clusters']/results['n_frames']:.1f} per frame)")
    print(f"Total GT objects: {results['total_gt_objects']}")
    
    print(f"\n{'Metric':<15} {'Value':>10}")
    print("-"*30)
    print(f"{'TP':<15} {results['tp']:>10}")
    print(f"{'FP':<15} {results['fp']:>10}")
    print(f"{'FN':<15} {results['fn']:>10}")
    print(f"{'Precision':<15} {results['precision']:>10.3f}")
    print(f"{'Recall':<15} {results['recall']:>10.3f}")
    print(f"{'F1':<15} {results['f1']:>10.3f}")
    print(f"{'Mean IoU':<15} {results['mean_iou']:>10.3f}")
    
    print(f"\n{'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'mIoU':>8}")
    print("-"*50)
    for cls in CLASS_NAMES:
        c = results['per_class'][cls]
        print(f"{cls:<15} {c['precision']:>8.3f} {c['recall']:>8.3f} "
              f"{c['f1']:>8.3f} {c['mean_iou']:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def grid_search(
    cleaned_dir: Path,
    gt_csv: Path,
    n_frames: int = 20,
) -> pd.DataFrame:
    """
    Grid search over clustering parameters.
    """
    # Parameter grid
    min_cluster_sizes = [20, 30, 50, 75, 100]
    min_samples_list = [5, 10, 15, 20]
    epsilons = [0.0, 0.3, 0.5, 1.0]
    
    results = []
    total = len(min_cluster_sizes) * len(min_samples_list) * len(epsilons)
    
    print(f"Grid search: {total} configurations")
    print(f"Testing on {n_frames} frames each\n")
    
    pbar = tqdm(total=total, desc="Grid search")
    
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            for eps in epsilons:
                config = {
                    'min_cluster_size': mcs,
                    'min_samples': ms,
                    'cluster_selection_epsilon': eps,
                    'metric': 'euclidean',
                    'min_points_bbox': 20,
                    'max_cluster_points': 50000,
                    'min_bbox_volume': 0.1,
                    'max_bbox_volume': 50000,
                }
                
                try:
                    res = evaluate_clustering(cleaned_dir, gt_csv, config, n_frames)
                    results.append({
                        'min_cluster_size': mcs,
                        'min_samples': ms,
                        'epsilon': eps,
                        'f1': res['f1'],
                        'precision': res['precision'],
                        'recall': res['recall'],
                        'mean_iou': res['mean_iou'],
                        'n_clusters': res['total_clusters'],
                        # Per-class
                        'f1_antenna': res['per_class']['Antenna']['f1'],
                        'f1_cable': res['per_class']['Cable']['f1'],
                        'f1_pole': res['per_class']['Electric Pole']['f1'],
                        'f1_turbine': res['per_class']['Wind Turbine']['f1'],
                    })
                except Exception as e:
                    print(f"Error with config {config}: {e}")
                
                pbar.update(1)
    
    pbar.close()
    
    df = pd.DataFrame(results)
    df = df.sort_values('f1', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS BY F1")
    print("="*60)
    print(df.head(10).to_string(index=False))
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_clustering(
    cleaned_dir: Path,
    gt_csv: Path,
    config: dict,
    n_frames: int = 10,
):
    """
    Detailed analysis of clustering behavior.
    """
    cleaned_dir = Path(cleaned_dir)
    gt_df = pd.read_csv(gt_csv)
    npz_files = sorted(cleaned_dir.glob("**/*.npz"))[:n_frames]
    
    print("\n" + "="*60)
    print("DETAILED CLUSTERING ANALYSIS")
    print("="*60)
    
    # Collect stats
    cluster_sizes = []
    gt_sizes = []
    class_cluster_stats = {c: {'sizes': [], 'heights': [], 'elongations': []} 
                           for c in CLASS_NAMES}
    
    for npz_path in tqdm(npz_files, desc="Analyzing"):
        scene = npz_path.parent.name
        pose_index = int(npz_path.stem.split('_')[1])
        
        data = np.load(npz_path)
        xyz = data['xyz'].astype(np.float32)
        rgb = data.get('rgb', None)
        
        # Get GT
        frame_gt = gt_df[(gt_df['scene'] == scene) & (gt_df['pose_index'] == pose_index)]
        for _, row in frame_gt.iterrows():
            gt_sizes.append(row['bbox_length'] * row['bbox_width'] * row['bbox_height'])
        
        # Cluster
        detections = cluster_frame(xyz, rgb, config)
        
        for det in detections:
            cluster_sizes.append(det['n_points'])
            
            cls = det.get('gt_class')
            if cls and cls in class_cluster_stats:
                stats = class_cluster_stats[cls]
                stats['sizes'].append(det['n_points'])
                stats['heights'].append(det['bbox_height'])
                stats['elongations'].append(det['elongation'])
    
    # Print analysis
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-"*50)
    print(f"{'Avg cluster size (points)':<30} {np.mean(cluster_sizes):>15.1f}")
    print(f"{'Median cluster size':<30} {np.median(cluster_sizes):>15.1f}")
    print(f"{'Max cluster size':<30} {max(cluster_sizes):>15}")
    print(f"{'Min cluster size':<30} {min(cluster_sizes):>15}")
    
    print(f"\n{'Class':<15} {'Count':>8} {'Avg Size':>10} {'Avg Height':>12} {'Avg Elong':>12}")
    print("-"*60)
    
    for cls in CLASS_NAMES:
        stats = class_cluster_stats[cls]
        n = len(stats['sizes'])
        if n > 0:
            print(f"{cls:<15} {n:>8} {np.mean(stats['sizes']):>10.1f} "
                  f"{np.mean(stats['heights']):>12.2f} {np.mean(stats['elongations']):>12.2f}")
        else:
            print(f"{cls:<15} {0:>8} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
    
    # Cable-specific analysis
    if class_cluster_stats['Cable']['elongations']:
        cable_elong = class_cluster_stats['Cable']['elongations']
        print(f"\nCable elongation distribution:")
        print(f"  Min: {min(cable_elong):.2f}")
        print(f"  25%: {np.percentile(cable_elong, 25):.2f}")
        print(f"  50%: {np.median(cable_elong):.2f}")
        print(f"  75%: {np.percentile(cable_elong, 75):.2f}")
        print(f"  Max: {max(cable_elong):.2f}")
    else:
        print("\nNo Cable clusters detected - cables may be fragmenting!")
        print("Try: lower min_cluster_size, or use line detection instead")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tune and analyze clustering")
    parser.add_argument('--cleaned-dir', type=str, default='cleaned/')
    parser.add_argument('--gt-csv', type=str, default='bboxes_train.csv')
    parser.add_argument('--n-frames', type=int, default=20)
    
    parser.add_argument('--grid-search', action='store_true',
                        help='Run grid search over parameters')
    parser.add_argument('--analyze', action='store_true',
                        help='Run detailed analysis')
    
    # Specific params to test
    parser.add_argument('--min-cluster-size', type=int, default=50)
    parser.add_argument('--min-samples', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.grid_search:
        df = grid_search(args.cleaned_dir, args.gt_csv, args.n_frames)
        df.to_csv('grid_search_results.csv', index=False)
        print("\nSaved to grid_search_results.csv")
    else:
        config = {
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples,
            'cluster_selection_epsilon': args.epsilon,
            'metric': 'euclidean',
            'min_points_bbox': 20,
            'max_cluster_points': 50000,
            'min_bbox_volume': 0.1,
            'max_bbox_volume': 50000,
        }
        
        results = evaluate_clustering(
            args.cleaned_dir, args.gt_csv, config, args.n_frames
        )
        print_results(results)
        
        if args.analyze:
            analyze_clustering(
                args.cleaned_dir, args.gt_csv, config, args.n_frames
            )


if __name__ == '__main__':
    main()
