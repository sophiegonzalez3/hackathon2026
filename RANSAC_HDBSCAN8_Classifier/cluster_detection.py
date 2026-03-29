"""
Improved Cluster Detection with Geometric Filtering
=====================================================
Addresses the over-clustering problem by:
1. Filtering clusters based on object-like geometric properties
2. Special handling for cables (high elongation, merge fragments)
3. Height-based filtering to remove remaining ground clusters

Usage:
    python cluster_detection_v2.py --cleaned-dir cleaned/ --gt-csv bboxes_train.csv \
        --out-dir cluster_detections_v2/
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    import hdbscan
except ImportError:
    print("pip install hdbscan")
    exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']
CLASS_RGB = {
    (38, 23, 180): 'Antenna',
    (177, 132, 47): 'Cable',
    (129, 81, 97): 'Electric Pole',
    (66, 132, 9): 'Wind Turbine',
}


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDING BOX FITTING
# ═══════════════════════════════════════════════════════════════════════════════

def fit_obb_pca(points: np.ndarray) -> dict:
    """Fit oriented bounding box using PCA."""
    if len(points) < 4:
        return None
    
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    try:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
    except:
        eigenvectors = np.eye(3)
    
    projected = centered @ eigenvectors
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extents = maxs - mins
    
    principal_xy = eigenvectors[:2, 0]
    yaw = np.arctan2(principal_xy[1], principal_xy[0])
    
    length = extents[0]
    width = extents[1]
    height = extents[2]
    
    if width > length:
        length, width = width, length
        yaw += np.pi / 2
    
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    bbox_center_local = (mins + maxs) / 2
    bbox_center = centroid + eigenvectors @ bbox_center_local
    
    return {
        'center': bbox_center,
        'length': float(length),
        'width': float(width),
        'height': float(height),
        'yaw': float(yaw),
        'volume': float(length * width * height),
    }


def fit_obb_minarea(points: np.ndarray) -> dict:
    """Fit minimum-area oriented bounding box using rotating calipers."""
    if len(points) < 4:
        return None
    
    xy = points[:, :2]
    
    try:
        hull = ConvexHull(xy)
        hull_points = xy[hull.vertices]
    except:
        return fit_obb_pca(points)
    
    n = len(hull_points)
    min_area = float('inf')
    best_rect = None
    
    for i in range(n):
        edge = hull_points[(i + 1) % n] - hull_points[i]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue
        edge = edge / edge_len
        
        perp = np.array([-edge[1], edge[0]])
        
        proj_edge = hull_points @ edge
        proj_perp = hull_points @ perp
        
        min_e, max_e = proj_edge.min(), proj_edge.max()
        min_p, max_p = proj_perp.min(), proj_perp.max()
        
        area = (max_e - min_e) * (max_p - min_p)
        
        if area < min_area:
            min_area = area
            best_rect = {
                'edge': edge,
                'perp': perp,
                'min_e': min_e, 'max_e': max_e,
                'min_p': min_p, 'max_p': max_p,
            }
    
    if best_rect is None:
        return fit_obb_pca(points)
    
    length_xy = best_rect['max_e'] - best_rect['min_e']
    width_xy = best_rect['max_p'] - best_rect['min_p']
    
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    height = z_max - z_min
    
    center_e = (best_rect['min_e'] + best_rect['max_e']) / 2
    center_p = (best_rect['min_p'] + best_rect['max_p']) / 2
    center_xy = center_e * best_rect['edge'] + center_p * best_rect['perp']
    center_z = (z_min + z_max) / 2
    center = np.array([center_xy[0], center_xy[1], center_z])
    
    yaw = np.arctan2(best_rect['edge'][1], best_rect['edge'][0])
    
    if width_xy > length_xy:
        length_xy, width_xy = width_xy, length_xy
        yaw += np.pi / 2
    
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    
    return {
        'center': center,
        'length': float(length_xy),
        'width': float(width_xy),
        'height': float(height),
        'yaw': float(yaw),
        'volume': float(length_xy * width_xy * height),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_bbox_features(points: np.ndarray, bbox: dict) -> dict:
    """Extract geometric features from a cluster for classification."""
    n_points = len(points)
    
    length = bbox['length']
    width = bbox['width']
    height = bbox['height']
    volume = bbox['volume']
    
    aspect_lw = length / max(width, 0.01)
    aspect_lh = length / max(height, 0.01)
    aspect_wh = width / max(height, 0.01)
    
    elongation = length / max(min(width, height), 0.01)
    flatness = min(width, height) / max(length, 0.01)
    
    density = n_points / max(volume, 0.01)
    
    z = points[:, 2]
    z_min = z.min()
    z_max = z.max()
    z_mean = z.mean()
    z_std = z.std()
    
    z_above_ground = z_min
    vertical_ratio = height / max(length, width, 0.01)
    xy_extent = np.sqrt(length**2 + width**2)
    
    return {
        'bbox_length': length,
        'bbox_width': width,
        'bbox_height': height,
        'bbox_volume': volume,
        'aspect_lw': aspect_lw,
        'aspect_lh': aspect_lh,
        'aspect_wh': aspect_wh,
        'elongation': elongation,
        'flatness': flatness,
        'n_points': n_points,
        'density': density,
        'z_min': z_min,
        'z_max': z_max,
        'z_mean': z_mean,
        'z_std': z_std,
        'z_above_ground': z_above_ground,
        'vertical_ratio': vertical_ratio,
        'xy_extent': xy_extent,
    }


def get_dominant_class(rgb: np.ndarray) -> Optional[str]:
    """Get the most common GT class in a cluster based on RGB labels."""
    class_counts = {}
    
    for r, g, b in rgb:
        key = (int(r), int(g), int(b))
        if key in CLASS_RGB:
            cls = CLASS_RGB[key]
            class_counts[cls] = class_counts.get(cls, 0) + 1
    
    if not class_counts:
        return None
    
    return max(class_counts, key=class_counts.get)


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC FILTERS - Key to reducing FP
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricFilter:
    """
    Filter clusters based on geometric properties.
    Objects have distinct signatures vs terrain noise.
    """
    
    # Min/max dimensions per class (meters) - tune these!
    CLASS_CONSTRAINTS = {
        'Antenna': {
            'min_height': 5.0,
            'max_height': 100.0,
            'min_elongation': 1.0,
            'max_elongation': 20.0,
            'vertical_ratio_min': 0.5,  # Taller than wide
        },
        'Cable': {
            'min_height': 0.1,
            'max_height': 15.0,
            'min_elongation': 5.0,  # KEY: cables are elongated
            'max_elongation': 2000.0,
            'min_length': 5.0,  # At least 5m long
        },
        'Electric Pole': {
            'min_height': 3.0,
            'max_height': 50.0,
            'min_elongation': 1.0,
            'max_elongation': 15.0,
            'vertical_ratio_min': 0.3,
        },
        'Wind Turbine': {
            'min_height': 10.0,
            'max_height': 200.0,
            'min_volume': 50.0,  # Big objects
        },
    }
    
    # General object constraints (not terrain)
    OBJECT_CONSTRAINTS = {
        'min_height_above_local': 1.0,  # At least 1m above local ground
        'min_z_range': 1.0,  # Points span at least 1m vertically
        'max_flatness': 0.8,  # Not too flat (ground patches are flat)
        'min_points': 30,
    }
    
    @classmethod
    def is_object_like(cls, cluster_points: np.ndarray, bbox: dict) -> Tuple[bool, str]:
        """
        Check if cluster has object-like properties vs terrain noise.
        Returns (is_valid, reason)
        """
        constraints = cls.OBJECT_CONSTRAINTS
        
        # Height span check
        z = cluster_points[:, 2]
        z_range = z.max() - z.min()
        if z_range < constraints['min_z_range']:
            return False, f"z_range={z_range:.1f} < {constraints['min_z_range']}"
        
        # Not too flat (ground patches)
        flatness = min(bbox['width'], bbox['height']) / max(bbox['length'], 0.01)
        if flatness > constraints['max_flatness'] and bbox['height'] < 2.0:
            return False, f"too_flat (flatness={flatness:.2f}, height={bbox['height']:.1f})"
        
        # Point count
        if len(cluster_points) < constraints['min_points']:
            return False, f"too_few_points ({len(cluster_points)})"
        
        return True, "ok"
    
    @classmethod
    def classify_by_geometry(cls, bbox: dict, features: dict) -> Tuple[Optional[str], float]:
        """
        Predict likely class based on geometric features alone.
        Returns (predicted_class, confidence)
        """
        scores = {}
        
        for class_name, constraints in cls.CLASS_CONSTRAINTS.items():
            score = 1.0
            
            # Height check
            if 'min_height' in constraints:
                if bbox['height'] < constraints['min_height']:
                    score *= 0.1
            if 'max_height' in constraints:
                if bbox['height'] > constraints['max_height']:
                    score *= 0.1
            
            # Elongation check (KEY for cables)
            elongation = features.get('elongation', 1.0)
            if 'min_elongation' in constraints:
                if elongation < constraints['min_elongation']:
                    score *= 0.1
                elif class_name == 'Cable' and elongation > 10:
                    score *= 2.0  # Bonus for cable-like elongation
            if 'max_elongation' in constraints:
                if elongation > constraints['max_elongation']:
                    score *= 0.1
            
            # Vertical ratio
            if 'vertical_ratio_min' in constraints:
                vr = features.get('vertical_ratio', 0)
                if vr < constraints['vertical_ratio_min']:
                    score *= 0.3
            
            # Volume check
            if 'min_volume' in constraints:
                if bbox['volume'] < constraints['min_volume']:
                    score *= 0.1
            
            # Length check
            if 'min_length' in constraints:
                if bbox['length'] < constraints['min_length']:
                    score *= 0.2
            
            scores[class_name] = score
        
        if not scores:
            return None, 0.0
        
        best_class = max(scores, key=scores.get)
        confidence = scores[best_class] / (sum(scores.values()) + 0.001)
        
        return best_class, confidence


# ═══════════════════════════════════════════════════════════════════════════════
# CABLE-SPECIFIC DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cables_by_elongation(
    xyz: np.ndarray,
    clusters: np.ndarray,
    min_elongation: float = 10.0,
    merge_distance: float = 5.0,
) -> List[dict]:
    """
    Special cable detection: find highly elongated clusters and merge nearby ones.
    
    Cables fragment because they're thin - we merge fragments that are:
    1. Both highly elongated
    2. Roughly collinear
    3. Close together
    """
    cable_candidates = []
    
    # Find elongated clusters
    unique_labels = set(clusters)
    unique_labels.discard(-1)
    
    elongated_clusters = []
    
    for label in unique_labels:
        mask = clusters == label
        points = xyz[mask]
        
        if len(points) < 20:
            continue
        
        bbox = fit_obb_minarea(points)
        if bbox is None:
            continue
        
        features = extract_bbox_features(points, bbox)
        elongation = features['elongation']
        
        if elongation >= min_elongation:
            elongated_clusters.append({
                'label': label,
                'points': points,
                'bbox': bbox,
                'features': features,
                'center': bbox['center'],
                'direction': get_principal_direction(points),
            })
    
    if not elongated_clusters:
        return []
    
    # Merge nearby collinear clusters
    merged = merge_cable_fragments(elongated_clusters, merge_distance)
    
    # Build final detections
    for group in merged:
        all_points = np.vstack([c['points'] for c in group])
        bbox = fit_obb_minarea(all_points)
        if bbox is None:
            continue
        
        features = extract_bbox_features(all_points, bbox)
        
        cable_candidates.append({
            'cluster_id': group[0]['label'],  # Use first cluster's ID
            'n_points': len(all_points),
            'n_fragments': len(group),
            'bbox_center_x': float(bbox['center'][0]),
            'bbox_center_y': float(bbox['center'][1]),
            'bbox_center_z': float(bbox['center'][2]),
            'bbox_length': bbox['length'],
            'bbox_width': bbox['width'],
            'bbox_height': bbox['height'],
            'bbox_yaw': bbox['yaw'],
            'predicted_class': 'Cable',
            'geo_confidence': 0.8,  # High confidence for elongated objects
            **features,
        })
    
    return cable_candidates


def get_principal_direction(points: np.ndarray) -> np.ndarray:
    """Get the principal direction of a point cloud (first eigenvector)."""
    centered = points - points.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        return Vt[0]  # First principal component
    except:
        return np.array([1, 0, 0])


def merge_cable_fragments(
    clusters: List[dict],
    max_distance: float = 5.0,
    max_angle: float = 30.0,  # degrees
) -> List[List[dict]]:
    """
    Merge cable fragments that are nearby and roughly collinear.
    """
    if len(clusters) <= 1:
        return [[c] for c in clusters]
    
    n = len(clusters)
    centers = np.array([c['center'] for c in clusters])
    directions = np.array([c['direction'] for c in clusters])
    
    # Build adjacency based on distance and angle
    adj = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(i+1, n):
            # Distance check
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > max_distance:
                continue
            
            # Angle check (directions should be similar)
            cos_angle = abs(np.dot(directions[i], directions[j]))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            if angle_deg > max_angle:
                continue
            
            adj[i, j] = adj[j, i] = True
    
    # Connected components
    visited = [False] * n
    groups = []
    
    for start in range(n):
        if visited[start]:
            continue
        
        # BFS
        group = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            group.append(clusters[node])
            
            for neighbor in range(n):
                if adj[node, neighbor] and not visited[neighbor]:
                    queue.append(neighbor)
        
        groups.append(group)
    
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE V2
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_frame_v2(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    config: dict = None,
) -> List[dict]:
    """
    Improved clustering with filtering and cable-specific handling.
    """
    config = config or {}
    
    min_cluster_size = config.get('min_cluster_size', 30)
    min_samples = config.get('min_samples', 5)
    epsilon = config.get('cluster_selection_epsilon', 0.5)
    
    if len(xyz) < min_cluster_size:
        return []
    
    # Step 1: Height-based pre-filtering
    # Remove points very close to the local minimum (remaining ground)
    z = xyz[:, 2]
    z_min_local = np.percentile(z, 5)  # 5th percentile as "ground"
    height_above_ground = z - z_min_local
    
    # Keep points at least 0.5m above local ground
    height_mask = height_above_ground > 0.5
    xyz_filtered = xyz[height_mask]
    rgb_filtered = rgb[height_mask] if rgb is not None else None
    
    if len(xyz_filtered) < min_cluster_size:
        return []
    
    # Step 2: HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean',
        core_dist_n_jobs=-1,
    )
    
    labels = clusterer.fit_predict(xyz_filtered)
    
    # Step 3: Process clusters with geometric filtering
    detections = []
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    cable_labels = set()  # Track which clusters we classify as cables
    
    for label in unique_labels:
        mask = labels == label
        cluster_points = xyz_filtered[mask]
        cluster_rgb = rgb_filtered[mask] if rgb_filtered is not None else None
        
        n_points = len(cluster_points)
        if n_points < 20:
            continue
        
        # Fit bbox
        bbox = fit_obb_minarea(cluster_points)
        if bbox is None:
            continue
        
        # Extract features
        features = extract_bbox_features(cluster_points, bbox)
        
        # Geometric filtering
        is_valid, reason = GeometricFilter.is_object_like(cluster_points, bbox)
        if not is_valid:
            continue
        
        # Geometric classification
        geo_class, geo_conf = GeometricFilter.classify_by_geometry(bbox, features)
        
        # Track cables for merging
        if geo_class == 'Cable' and features['elongation'] > 10:
            cable_labels.add(label)
        
        # GT class from RGB (for analysis)
        gt_class = None
        if cluster_rgb is not None:
            gt_class = get_dominant_class(cluster_rgb)
        
        detection = {
            'cluster_id': int(label),
            'n_points': n_points,
            'bbox_center_x': float(bbox['center'][0]),
            'bbox_center_y': float(bbox['center'][1]),
            'bbox_center_z': float(bbox['center'][2]),
            'bbox_length': bbox['length'],
            'bbox_width': bbox['width'],
            'bbox_height': bbox['height'],
            'bbox_yaw': bbox['yaw'],
            'predicted_class': geo_class,
            'geo_confidence': geo_conf,
            'gt_class': gt_class,
            **features,
        }
        
        detections.append(detection)
    
    # Step 4: Cable merging
    # Re-run cable detection with fragment merging
    cable_detections = detect_cables_by_elongation(
        xyz_filtered, labels,
        min_elongation=config.get('cable_min_elongation', 8.0),
        merge_distance=config.get('cable_merge_distance', 10.0),
    )
    
    # Replace individual cable detections with merged ones
    non_cable_detections = [d for d in detections if d.get('predicted_class') != 'Cable']
    
    # Add merged cables
    all_detections = non_cable_detections + cable_detections
    
    return all_detections


def process_dataset_v2(
    cleaned_dir: Path,
    out_dir: Path,
    gt_csv: Optional[Path] = None,
    config: dict = None,
    max_frames: Optional[int] = None,
) -> pd.DataFrame:
    """Process all frames with improved clustering."""
    
    config = config or {
        'min_cluster_size': 30,
        'min_samples': 5,
        'cluster_selection_epsilon': 0.5,
        'cable_min_elongation': 8.0,
        'cable_merge_distance': 10.0,
    }
    
    cleaned_dir = Path(cleaned_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = sorted(cleaned_dir.glob("**/*.npz"))
    if max_frames:
        npz_files = npz_files[:max_frames]
    
    print(f"Processing {len(npz_files)} frames")
    print(f"Config: {config}")
    
    all_detections = []
    
    for npz_path in tqdm(npz_files, desc="Processing"):
        scene = npz_path.parent.name
        pose_index = int(npz_path.stem.split('_')[1])
        
        try:
            data = np.load(npz_path)
            xyz = data['xyz'].astype(np.float32)
            rgb = data.get('rgb', None)
            if rgb is not None:
                rgb = rgb.astype(np.uint8)
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
        
        detections = cluster_frame_v2(xyz, rgb, config)
        
        for det in detections:
            det['scene'] = scene
            det['pose_index'] = pose_index
        
        all_detections.extend(detections)
    
    df = pd.DataFrame(all_detections)
    
    if len(df) == 0:
        print("No detections!")
        return df
    
    # Stats
    print(f"\nTotal detections: {len(df)}")
    print(f"Per frame: {len(df) / len(npz_files):.1f}")
    
    print("\nPredicted class distribution:")
    print(df['predicted_class'].value_counts())
    
    if 'gt_class' in df.columns:
        print("\nGT class distribution (in detected clusters):")
        print(df['gt_class'].value_counts())
    
    # Save
    df.to_csv(out_dir / "detections_v2.csv", index=False)
    print(f"\nSaved to {out_dir / 'detections_v2.csv'}")
    
    # Evaluation if GT provided
    if gt_csv:
        evaluate_v2(df, gt_csv, len(npz_files))
    
    return df


def iou_3d_box(box1: dict, box2: dict) -> float:
    """Simple axis-aligned 3D IoU (ignoring rotation)."""
    c1 = np.array([box1['bbox_center_x'], box1['bbox_center_y'], box1['bbox_center_z']])
    c2 = np.array([box2['bbox_center_x'], box2['bbox_center_y'], box2['bbox_center_z']])
    
    s1 = np.array([max(box1['bbox_length'], box1['bbox_width'])/2,
                   min(box1['bbox_length'], box1['bbox_width'])/2,
                   box1['bbox_height']/2])
    s2 = np.array([max(box2['bbox_length'], box2['bbox_width'])/2,
                   min(box2['bbox_length'], box2['bbox_width'])/2,
                   box2['bbox_height']/2])
    
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


def match_detections_to_gt(
    detections: List[dict],
    gt_boxes: List[dict],
    iou_threshold: float = 0.3,
) -> Tuple[int, int, int, float]:
    """Match detections to GT using Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment
    
    if len(detections) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, 0.0
    if len(detections) == 0:
        return 0, 0, len(gt_boxes), 0.0
    if len(gt_boxes) == 0:
        return 0, len(detections), 0, 0.0
    
    iou_matrix = np.zeros((len(detections), len(gt_boxes)))
    for i, det in enumerate(detections):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = iou_3d_box(det, gt)
    
    det_idx, gt_idx = linear_sum_assignment(-iou_matrix)
    
    tp = 0
    matched_ious = []
    
    for d, g in zip(det_idx, gt_idx):
        if iou_matrix[d, g] >= iou_threshold:
            tp += 1
            matched_ious.append(iou_matrix[d, g])
    
    fp = len(detections) - tp
    fn = len(gt_boxes) - tp
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    
    return tp, fp, fn, mean_iou


def evaluate_v2(detections_df: pd.DataFrame, gt_csv: Path, n_frames: int):
    """Quick evaluation of v2 detections."""
    
    gt_df = pd.read_csv(gt_csv)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    per_class = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CLASS_NAMES}
    
    for scene in detections_df['scene'].unique():
        for pose in detections_df[detections_df['scene'] == scene]['pose_index'].unique():
            # Get detections for this frame
            frame_det = detections_df[
                (detections_df['scene'] == scene) & 
                (detections_df['pose_index'] == pose)
            ].to_dict('records')
            
            # Get GT
            frame_gt = gt_df[
                (gt_df['scene'] == scene) & 
                (gt_df['pose_index'] == pose)
            ].to_dict('records')
            
            tp, fp, fn, _ = match_detections_to_gt(frame_det, frame_gt, iou_threshold=0.25)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Per-class (by predicted class)
            for cls in CLASS_NAMES:
                cls_det = [d for d in frame_det if d.get('predicted_class') == cls]
                cls_gt = [g for g in frame_gt if g.get('class_label') == cls]
                ctp, cfp, cfn, _ = match_detections_to_gt(cls_det, cls_gt, iou_threshold=0.2)
                per_class[cls]['tp'] += ctp
                per_class[cls]['fp'] += cfp
                per_class[cls]['fn'] += cfn
    
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    print("\n" + "="*60)
    print("V2 EVALUATION RESULTS")
    print("="*60)
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    
    print(f"\n{'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-"*45)
    for cls in CLASS_NAMES:
        c = per_class[cls]
        cp = c['tp'] / max(c['tp'] + c['fp'], 1)
        cr = c['tp'] / max(c['tp'] + c['fn'], 1)
        cf1 = 2 * cp * cr / max(cp + cr, 1e-6)
        print(f"{cls:<15} {cp:>8.3f} {cr:>8.3f} {cf1:>8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaned-dir', type=str, default='cleaned/')
    parser.add_argument('--out-dir', type=str, default='cluster_detections_v2/')
    parser.add_argument('--gt-csv', type=str, default=None)
    parser.add_argument('--max-frames', type=int, default=None)
    
    # Clustering params
    parser.add_argument('--min-cluster-size', type=int, default=30)
    parser.add_argument('--min-samples', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.5)
    
    # Cable params
    parser.add_argument('--cable-min-elongation', type=float, default=8.0)
    parser.add_argument('--cable-merge-distance', type=float, default=10.0)
    
    args = parser.parse_args()
    
    config = {
        'min_cluster_size': args.min_cluster_size,
        'min_samples': args.min_samples,
        'cluster_selection_epsilon': args.epsilon,
        'cable_min_elongation': args.cable_min_elongation,
        'cable_merge_distance': args.cable_merge_distance,
    }
    
    process_dataset_v2(
        args.cleaned_dir,
        args.out_dir,
        args.gt_csv,
        config,
        args.max_frames,
    )


if __name__ == '__main__':
    main()