#!/usr/bin/env python3
"""
Merge prediction bboxes of the same class that are close together.
Same philosophy as ground truth generation: if two bboxes of the same class
have centers within merge_distance, merge them keeping the biggest dimensions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Dict


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE DISTANCES (same as GT config)
# ═══════════════════════════════════════════════════════════════════════════════

MERGE_DISTANCES: Dict[str, float] = {
    'Antenna': 30.0,
    'Cable': 0.5,           # cables can be close (parallel lines)
    'Electric Pole': 15.0,
    'Wind Turbine': 50.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE MERGE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def merge_bboxes_union_find(centers: np.ndarray, merge_distance: float) -> np.ndarray:
    """
    Union-find to group bboxes whose centers are within merge_distance.
    Returns array of group labels.
    """
    n = len(centers)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    
    # Compute pairwise distances
    dists = cdist(centers, centers)
    
    # Union-Find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Merge nearby bboxes
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < merge_distance:
                union(i, j)
    
    # Get group labels
    groups = np.array([find(i) for i in range(n)])
    
    # Renumber to consecutive integers
    unique_groups = np.unique(groups)
    group_map = {g: idx for idx, g in enumerate(unique_groups)}
    return np.array([group_map[g] for g in groups])


def merge_group(df_group: pd.DataFrame) -> pd.Series:
    """
    Merge a group of bboxes into one, keeping biggest dimensions.
    """
    # Center = centroid of all centers
    cx = df_group['bbox_center_x'].mean()
    cy = df_group['bbox_center_y'].mean()
    cz = df_group['bbox_center_z'].mean()
    
    # Dimensions = max of all
    w = df_group['bbox_width'].max()
    l = df_group['bbox_length'].max()
    h = df_group['bbox_height'].max()
    
    # Yaw = from the bbox with highest confidence (or first)
    if 'confidence' in df_group.columns:
        best_idx = df_group['confidence'].idxmax()
    else:
        best_idx = df_group.index[0]
    yaw = df_group.loc[best_idx, 'bbox_yaw']
    
    # Keep max confidence
    conf = df_group['confidence'].max() if 'confidence' in df_group.columns else 1.0
    
    # Copy first row and update
    result = df_group.iloc[0].copy()
    result['bbox_center_x'] = cx
    result['bbox_center_y'] = cy
    result['bbox_center_z'] = cz
    result['bbox_width'] = w
    result['bbox_length'] = l
    result['bbox_height'] = h
    result['bbox_yaw'] = yaw
    if 'confidence' in result.index:
        result['confidence'] = conf
    
    return result


def merge_frame_class(df: pd.DataFrame, merge_distance: float) -> pd.DataFrame:
    """
    Merge bboxes within a single frame and class.
    """
    if len(df) <= 1:
        return df
    
    # Get centers
    centers = df[['bbox_center_x', 'bbox_center_y', 'bbox_center_z']].values
    
    # Find groups
    groups = merge_bboxes_union_find(centers, merge_distance)
    
    # Merge each group
    merged_rows = []
    for g in np.unique(groups):
        group_df = df[groups == g]
        merged_rows.append(merge_group(group_df))
    
    return pd.DataFrame(merged_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def merge_predictions(
    input_csv: str,
    output_csv: str,
    merge_distances: Dict[str, float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge predictions: group by frame + class, merge nearby bboxes.
    """
    if merge_distances is None:
        merge_distances = MERGE_DISTANCES
    
    df = pd.read_csv(input_csv)
    
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  MERGE PREDICTIONS")
        print(f"{'═'*60}")
        print(f"  Input: {len(df)} detections")
        print(f"\n  Merge distances:")
        for cls, dist in merge_distances.items():
            print(f"    {cls}: {dist}m")
    
    # Group by frame (ego pose) and class
    # Frame is identified by (scene, pose_index) or (ego_x, ego_y, ego_z, ego_yaw)
    if 'scene' in df.columns and 'pose_index' in df.columns:
        frame_cols = ['scene', 'pose_index']
    else:
        frame_cols = ['ego_x', 'ego_y', 'ego_z', 'ego_yaw']
    
    merged_dfs = []
    
    for group_key, group_df in df.groupby(frame_cols + ['class_label']):
        class_label = group_key[-1]  # last element is class_label
        merge_dist = merge_distances.get(class_label, 10.0)
        merged = merge_frame_class(group_df.copy(), merge_dist)
        merged_dfs.append(merged)
    
    if merged_dfs:
        result = pd.concat(merged_dfs, ignore_index=True)
    else:
        result = df.iloc[:0].copy()
    
    if verbose:
        print(f"\n  Output: {len(result)} detections")
        print(f"  Reduced by: {len(df) - len(result)} ({100*(len(df)-len(result))/max(len(df),1):.1f}%)")
        print(f"\n  Class breakdown:")
        for cls in ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']:
            before = len(df[df['class_label'] == cls])
            after = len(result[result['class_label'] == cls])
            print(f"    {cls}: {before} → {after}")
    
    # Save
    result.to_csv(output_csv, index=False)
    if verbose:
        print(f"\n✓ Saved to {output_csv}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge nearby prediction bboxes')
    parser.add_argument('input_csv', help='Input predictions CSV')
    parser.add_argument('output_csv', nargs='?', default=None, help='Output CSV (default: input_merged.csv)')
    parser.add_argument('--antenna-dist', type=float, default=30.0, help='Merge distance for Antenna')
    parser.add_argument('--cable-dist', type=float, default=0.5, help='Merge distance for Cable')
    parser.add_argument('--pole-dist', type=float, default=15.0, help='Merge distance for Electric Pole')
    parser.add_argument('--turbine-dist', type=float, default=50.0, help='Merge distance for Wind Turbine')
    
    args = parser.parse_args()
    
    # Output path
    if args.output_csv is None:
        inp = Path(args.input_csv)
        args.output_csv = str(inp.with_name(inp.stem + '_merged' + inp.suffix))
    
    # Custom merge distances
    merge_distances = {
        'Antenna': args.antenna_dist,
        'Cable': args.cable_dist,
        'Electric Pole': args.pole_dist,
        'Wind Turbine': args.turbine_dist,
    }
    
    merge_predictions(args.input_csv, args.output_csv, merge_distances)