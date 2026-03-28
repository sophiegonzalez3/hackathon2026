"""
Post-processing to improve predictions WITHOUT retraining.
Apply this to your predictions.csv before evaluation.
"""

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Tune these based on your data
# ═══════════════════════════════════════════════════════════════════════════════

# Per-class confidence thresholds (raise them!)
CLASS_THRESHOLDS = {
    'Antenna':       0.40,  # Was implicitly 0.20, your mean is 0.466
    'Cable':         0.28,  # Aggressive filter - mean is 0.246, most are noise
    'Electric Pole': 0.35,  # Mean is 0.402
    'Wind Turbine':  0.25,  # Mean is 0.300
}

# Per-class max detections per frame (prevents explosion)
CLASS_MAX_PER_FRAME = {
    'Antenna':       10,
    'Cable':         30,   # Cables are fragmented, allow more but not 50+
    'Electric Pole': 8,
    'Wind Turbine':  10,
}

# Dimension sanity filters (remove obviously wrong boxes)
# Format: (min_val, max_val) for each dimension
DIM_FILTERS = {
    'Antenna': {
        'bbox_width':  (2, 25),
        'bbox_length': (2, 25),
        'bbox_height': (10, 80),
    },
    'Cable': {
        'bbox_width':  (5, 100),   # Allow wide range since GT is messy
        'bbox_length': (0.3, 15),
        'bbox_height': (0.3, 15),
    },
    'Electric Pole': {
        'bbox_width':  (2, 40),
        'bbox_length': (2, 25),
        'bbox_height': (15, 80),
    },
    'Wind Turbine': {
        'bbox_width':  (15, 150),
        'bbox_length': (3, 50),
        'bbox_height': (20, 200),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_class_thresholds(df):
    """Filter by per-class confidence thresholds."""
    masks = []
    for cls, thresh in CLASS_THRESHOLDS.items():
        mask = (df['class_label'] == cls) & (df['confidence'] >= thresh)
        masks.append(mask)
    
    keep_mask = pd.concat([pd.Series(m) for m in masks], axis=1).any(axis=1)
    filtered = df[keep_mask].copy()
    
    print(f"Confidence filter: {len(df)} → {len(filtered)} ({len(df) - len(filtered)} removed)")
    return filtered


def apply_dimension_filters(df):
    """Remove boxes with unrealistic dimensions."""
    keep_mask = pd.Series([True] * len(df), index=df.index)
    
    for cls, filters in DIM_FILTERS.items():
        cls_mask = df['class_label'] == cls
        for dim, (min_val, max_val) in filters.items():
            dim_ok = (df[dim] >= min_val) & (df[dim] <= max_val)
            keep_mask &= ~cls_mask | dim_ok  # Keep if not this class OR dim is OK
    
    filtered = df[keep_mask].copy()
    print(f"Dimension filter: {len(df)} → {len(filtered)} ({len(df) - len(filtered)} removed)")
    return filtered


def apply_per_frame_limits(df):
    """Limit detections per frame per class (keep highest confidence)."""
    results = []
    
    # Group by frame
    frame_cols = ['ego_x', 'ego_y', 'ego_z', 'ego_yaw']
    
    for frame_id, frame_df in df.groupby(frame_cols):
        frame_results = []
        for cls, max_det in CLASS_MAX_PER_FRAME.items():
            cls_df = frame_df[frame_df['class_label'] == cls]
            if len(cls_df) > max_det:
                cls_df = cls_df.nlargest(max_det, 'confidence')
            frame_results.append(cls_df)
        results.append(pd.concat(frame_results, ignore_index=True))
    
    filtered = pd.concat(results, ignore_index=True)
    print(f"Per-frame limit: {len(df)} → {len(filtered)} ({len(df) - len(filtered)} removed)")
    return filtered


def simple_3d_nms(df, iou_threshold=0.3):
    """
    Simple 3D NMS within each frame and class.
    Removes overlapping boxes, keeping higher confidence ones.
    """
    def compute_iou_3d(box_a, box_b):
        def get_corners(b):
            cx, cy, cz = b['bbox_center_x'], b['bbox_center_y'], b['bbox_center_z']
            w, l, h = b['bbox_width']/2, b['bbox_length']/2, b['bbox_height']/2
            return (cx-w, cy-l, cz-h, cx+w, cy+l, cz+h)
        
        a, b = get_corners(box_a), get_corners(box_b)
        x1, y1, z1 = max(a[0],b[0]), max(a[1],b[1]), max(a[2],b[2])
        x2, y2, z2 = min(a[3],b[3]), min(a[4],b[4]), min(a[5],b[5])
        
        if x2 <= x1 or y2 <= y1 or z2 <= z1:
            return 0.0
        
        inter = (x2-x1) * (y2-y1) * (z2-z1)
        vol_a = (a[3]-a[0]) * (a[4]-a[1]) * (a[5]-a[2])
        vol_b = (b[3]-b[0]) * (b[4]-b[1]) * (b[5]-b[2])
        return inter / (vol_a + vol_b - inter + 1e-10)
    
    frame_cols = ['ego_x', 'ego_y', 'ego_z', 'ego_yaw']
    results = []
    
    for frame_id, frame_df in df.groupby(frame_cols):
        for cls in frame_df['class_label'].unique():
            cls_df = frame_df[frame_df['class_label'] == cls].copy()
            cls_df = cls_df.sort_values('confidence', ascending=False)
            
            keep_indices = []
            rows = cls_df.to_dict('records')
            
            for i, row in enumerate(rows):
                keep = True
                for j in keep_indices:
                    if compute_iou_3d(row, rows[j]) > iou_threshold:
                        keep = False
                        break
                if keep:
                    keep_indices.append(i)
            
            results.append(cls_df.iloc[keep_indices])
    
    filtered = pd.concat(results, ignore_index=True)
    print(f"3D NMS (IoU>{iou_threshold}): {len(df)} → {len(filtered)} ({len(df) - len(filtered)} removed)")
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def postprocess(input_csv, output_csv):
    """Apply all post-processing steps."""
    print(f"\n{'═'*60}")
    print(f"  POST-PROCESSING: {input_csv}")
    print(f"{'═'*60}\n")
    
    df = pd.read_csv(input_csv)
    print(f"Input: {len(df)} detections\n")
    
    # Print initial stats
    print("Before post-processing:")
    print(df['class_label'].value_counts().to_string())
    print()
    
    # Apply filters in order
    df = apply_class_thresholds(df)
    df = apply_dimension_filters(df)
    df = simple_3d_nms(df, iou_threshold=0.3)
    df = apply_per_frame_limits(df)
    
    print(f"\nFinal: {len(df)} detections")
    print("\nAfter post-processing:")
    print(df['class_label'].value_counts().to_string())
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to {output_csv}")
    
    # Print confidence stats
    print("\nConfidence stats after filtering:")
    for cls in ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']:
        cls_df = df[df['class_label'] == cls]
        if len(cls_df) > 0:
            print(f"  {cls}: mean={cls_df['confidence'].mean():.3f}, count={len(cls_df)}")


if __name__ == '__main__':
    import sys
    
    input_csv = sys.argv[1] if len(sys.argv) > 1 else 'predictions.csv'
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'predictions_filtered.csv'
    
    postprocess(input_csv, output_csv)
