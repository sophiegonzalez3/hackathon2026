"""
PointPillars + CenterHead — Utilities
======================================
Pillarization, Gaussian target generation, CenterNet-style decoding & NMS.
"""

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Pillarization (numpy, done in dataset __getitem__)
# ═══════════════════════════════════════════════════════════════════════════════

def pillarize(points, cfg):
    """
    Convert raw points (N, 4) → pillar tensors for the model.
    Fully vectorized — no Python loops over pillars.

    Args:
        points: (N, 4) — [x, y, z, reflectivity]
        cfg: Config object

    Returns:
        pillar_features:    (P, max_pts, 9) float32
        num_points:         (P,) int32
        pillar_coords:      (P, 2) int32 — [grid_x_idx, grid_y_idx]
    """
    pcr = cfg.point_cloud_range
    x_min, y_min, z_min = pcr[0], pcr[1], pcr[2]
    x_max, y_max, z_max = pcr[3], pcr[4], pcr[5]
    px, py = cfg.pillar_x, cfg.pillar_y
    max_pts = cfg.max_points_per_pillar
    max_pillars = cfg.max_pillars

    # ── Clip to range ─────────────────────────────────────────────────────
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]

    if len(points) == 0:
        return (
            np.zeros((1, max_pts, 9), dtype=np.float32),
            np.zeros((1,), dtype=np.int32),
            np.zeros((1, 2), dtype=np.int32),
        )

    # ── Compute pillar grid indices ───────────────────────────────────────
    xi = np.floor((points[:, 0] - x_min) / px).astype(np.int32)
    yi = np.floor((points[:, 1] - y_min) / py).astype(np.int32)
    xi = np.clip(xi, 0, cfg.grid_x - 1)
    yi = np.clip(yi, 0, cfg.grid_y - 1)

    flat_idx = xi * cfg.grid_y + yi

    # ── Sort points by pillar index (groups them together) ────────────────
    sort_order = np.argsort(flat_idx, kind='mergesort')
    flat_sorted = flat_idx[sort_order]
    points_sorted = points[sort_order]

    # ── Find unique pillars and their boundaries ─────────────────────────
    unique_flat, starts, counts = np.unique(
        flat_sorted, return_index=True, return_counts=True
    )
    num_pillars = len(unique_flat)

    # ── Subsample pillars if too many ─────────────────────────────────────
    if num_pillars > max_pillars:
        keep = np.random.choice(num_pillars, max_pillars, replace=False)
        keep.sort()
        unique_flat = unique_flat[keep]
        starts = starts[keep]
        counts = counts[keep]
        num_pillars = max_pillars

    # ── Pillar grid coordinates ───────────────────────────────────────────
    gx = unique_flat // cfg.grid_y
    gy = unique_flat % cfg.grid_y
    pillar_coords = np.stack([gx, gy], axis=1).astype(np.int32)

    # Pillar centers in world coordinates
    center_x = x_min + (gx + 0.5) * px
    center_y = y_min + (gy + 0.5) * py

    # ── Clamp per-pillar point count ──────────────────────────────────────
    clamped_counts = np.minimum(counts, max_pts)
    num_points_per_pillar = clamped_counts.astype(np.int32)

    # ── Build flat arrays of (pillar_idx, position_in_pillar, point_idx) ──
    total_selected = int(clamped_counts.sum())
    pillar_ids = np.repeat(np.arange(num_pillars), clamped_counts)

    # Position within each pillar: [0,1,..,c0-1, 0,1,..,c1-1, ...]
    # Using cumsum trick — zero Python loops
    cumsum_clamped = np.cumsum(clamped_counts)
    cumsum_shifted = np.empty(num_pillars, dtype=np.int64)
    cumsum_shifted[0] = 0
    cumsum_shifted[1:] = cumsum_clamped[:-1]
    repeated_shifts = np.repeat(cumsum_shifted, clamped_counts)
    pos_in_pillar = (np.arange(total_selected) - repeated_shifts).astype(np.int32)

    # Global point indices in points_sorted: [s0+0, s0+1, .., s1+0, s1+1, ..]
    repeated_starts = np.repeat(starts, clamped_counts)
    global_pt_idx = (repeated_starts + pos_in_pillar).astype(np.int64)

    # ── Gather selected points ────────────────────────────────────────────
    sel_pts = points_sorted[global_pt_idx]  # (total_selected, 4)

    # ── Compute per-pillar mean XYZ (vectorized) ─────────────────────────
    # Sum xyz per pillar, then divide by count
    pillar_sum_xyz = np.zeros((num_pillars, 3), dtype=np.float64)
    np.add.at(pillar_sum_xyz, pillar_ids, sel_pts[:, :3])
    pillar_mean_xyz = pillar_sum_xyz / np.maximum(clamped_counts, 1)[:, None]

    # Broadcast mean back to each point
    mean_per_point = pillar_mean_xyz[pillar_ids]  # (total_selected, 3)

    # Broadcast center back to each point
    cx_per_point = center_x[pillar_ids]  # (total_selected,)
    cy_per_point = center_y[pillar_ids]  # (total_selected,)

    # ── Build the 9-feature vector for each selected point ───────────────
    features_9 = np.zeros((total_selected, 9), dtype=np.float32)
    features_9[:, :4] = sel_pts                                # x, y, z, r
    features_9[:, 4] = sel_pts[:, 0] - mean_per_point[:, 0]   # x_c
    features_9[:, 5] = sel_pts[:, 1] - mean_per_point[:, 1]   # y_c
    features_9[:, 6] = sel_pts[:, 2] - mean_per_point[:, 2]   # z_c
    features_9[:, 7] = sel_pts[:, 0] - cx_per_point            # x_p
    features_9[:, 8] = sel_pts[:, 1] - cy_per_point            # y_p

    # ── Scatter into (P, max_pts, 9) output array ─────────────────────────
    pillar_features = np.zeros((num_pillars, max_pts, 9), dtype=np.float32)
    pillar_features[pillar_ids, pos_in_pillar] = features_9

    return pillar_features, num_points_per_pillar, pillar_coords


# ═══════════════════════════════════════════════════════════════════════════════
# CenterNet Gaussian target generation
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_2d(shape, sigma=1.0):
    """Generate 2D Gaussian kernel."""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """Draw a Gaussian on the heatmap at center (x, y) with given radius."""
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape

    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    if left + right <= 0 or top + bottom <= 0:
        return

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if masked_heatmap.shape[0] > 0 and masked_heatmap.shape[1] > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Compute Gaussian radius from object BEV size.
    From CenterNet: radius such that IoU of a box shifted by radius
    with the GT box is >= min_overlap.
    """
    width, height = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def generate_targets(gt_boxes, cfg):
    """
    Generate CenterHead targets for one frame.

    Args:
        gt_boxes: list of dicts with keys:
            class_id, bbox_center_x/y/z, bbox_width/length/height, bbox_yaw

    Returns:
        heatmap:    (num_classes, hm_H, hm_W) float32
        offset:     (2, hm_H, hm_W) float32  — sub-pixel xy offset
        z_target:   (1, hm_H, hm_W) float32  — z center
        dim_target: (3, hm_H, hm_W) float32  — log(w), log(l), log(h)
        rot_target: (2, hm_H, hm_W) float32  — sin(yaw), cos(yaw)
        reg_mask:   (hm_H, hm_W) float32      — 1 at GT center, 0 elsewhere
        indices:    (max_objs, 2) int32        — [hm_y, hm_x] of GT centers
        num_objs:   int                         — number of GT objects
    """
    pcr = cfg.point_cloud_range
    hm_h, hm_w = cfg.heatmap_h, cfg.heatmap_w
    stride = cfg.head_stride
    px, py = cfg.pillar_x, cfg.pillar_y

    # Effective resolution of the heatmap in meters
    res_x = px * stride  # meters per heatmap pixel
    res_y = py * stride

    heatmap = np.zeros((cfg.num_classes, hm_h, hm_w), dtype=np.float32)
    offset = np.zeros((2, hm_h, hm_w), dtype=np.float32)
    z_target = np.zeros((1, hm_h, hm_w), dtype=np.float32)
    dim_target = np.zeros((3, hm_h, hm_w), dtype=np.float32)
    rot_target = np.zeros((2, hm_h, hm_w), dtype=np.float32)
    reg_mask = np.zeros((hm_h, hm_w), dtype=np.float32)

    max_objs = 64
    indices = np.zeros((max_objs, 2), dtype=np.int32)
    num_objs = 0

    for box in gt_boxes:
        cx = box['bbox_center_x']
        cy = box['bbox_center_y']
        cz = box['bbox_center_z']
        w = box['bbox_width']
        l = box['bbox_length']
        h = box['bbox_height']
        yaw = box['bbox_yaw']
        cid = box['class_id']
        cls_name = cfg.class_names[cid]

        # Convert center to heatmap coordinates
        hm_x = (cx - pcr[0]) / res_x
        hm_y = (cy - pcr[1]) / res_y

        hm_x_int = int(np.floor(hm_x))
        hm_y_int = int(np.floor(hm_y))

        # Skip if outside heatmap
        if hm_x_int < 0 or hm_x_int >= hm_w or hm_y_int < 0 or hm_y_int >= hm_h:
            continue

        # Gaussian radius from BEV object size (in heatmap pixels)
        bev_w = w / res_x  # width in heatmap pixels
        bev_l = l / res_y  # length in heatmap pixels
        radius = max(
            int(gaussian_radius((bev_w, bev_l), min_overlap=0.7)),
            cfg.gaussian_min_radius.get(cls_name, 2)
        )

        # Draw Gaussian on class-specific heatmap
        draw_gaussian(heatmap[cid], center=(hm_x_int, hm_y_int), radius=radius)

        # Regression targets at the integer center location
        offset[0, hm_y_int, hm_x_int] = hm_x - hm_x_int  # fractional x offset
        offset[1, hm_y_int, hm_x_int] = hm_y - hm_y_int  # fractional y offset
        z_target[0, hm_y_int, hm_x_int] = cz
        dim_target[0, hm_y_int, hm_x_int] = np.log(max(w, 0.1))
        dim_target[1, hm_y_int, hm_x_int] = np.log(max(l, 0.1))
        dim_target[2, hm_y_int, hm_x_int] = np.log(max(h, 0.1))
        rot_target[0, hm_y_int, hm_x_int] = np.sin(yaw)
        rot_target[1, hm_y_int, hm_x_int] = np.cos(yaw)

        reg_mask[hm_y_int, hm_x_int] = 1.0

        if num_objs < max_objs:
            indices[num_objs] = [hm_y_int, hm_x_int]
            num_objs += 1

    return heatmap, offset, z_target, dim_target, rot_target, reg_mask, indices, num_objs


# ═══════════════════════════════════════════════════════════════════════════════
# Inference: decode predictions → 3D boxes
# ═══════════════════════════════════════════════════════════════════════════════

def nms_heatmap(heatmap, kernel=3):
    """Simple max-pool NMS on heatmap."""
    import torch
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(
        heatmap, kernel_size=kernel, stride=1, padding=pad
    )
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_predictions(preds, cfg):
    """
    Decode CenterHead outputs → list of 3D bounding boxes.

    Args:
        preds: dict with 'heatmap', 'offset', 'z', 'dim', 'rot'
               All tensors of shape (B, C, H, W)

    Returns:
        list of (per-batch) dicts:
          boxes: (K, 7) — [cx, cy, cz, w, l, h, yaw]
          scores: (K,)
          labels: (K,) int
    """
    pcr = cfg.point_cloud_range
    stride = cfg.head_stride
    res_x = cfg.pillar_x * stride
    res_y = cfg.pillar_y * stride

    heatmap = preds['heatmap']  # (B, C, H, W)
    B = heatmap.shape[0]

    # NMS on heatmap
    heatmap = nms_heatmap(heatmap, kernel=cfg.nms_kernel)

    results = []

    for b in range(B):
        hm = heatmap[b]  # (C, H, W)

        # Flatten and get top-K
        C, H, W = hm.shape
        hm_flat = hm.reshape(C, -1)  # (C, H*W)

        # Per-class top-K, then merge
        all_scores = []
        all_indices = []
        all_labels = []

        for c in range(C):
            scores_c = hm_flat[c]
            mask = scores_c > cfg.score_threshold
            if mask.sum() == 0:
                continue
            valid_scores = scores_c[mask]
            valid_indices = torch.where(mask)[0]
            all_scores.append(valid_scores)
            all_indices.append(valid_indices)
            all_labels.append(torch.full_like(valid_indices, c))

        if len(all_scores) == 0:
            results.append({
                'boxes': torch.zeros((0, 7), device=hm.device),
                'scores': torch.zeros((0,), device=hm.device),
                'labels': torch.zeros((0,), dtype=torch.long, device=hm.device),
            })
            continue

        all_scores = torch.cat(all_scores)
        all_indices = torch.cat(all_indices)
        all_labels = torch.cat(all_labels)

        # Keep top max_detections
        if len(all_scores) > cfg.max_detections:
            topk = torch.topk(all_scores, cfg.max_detections)
            all_scores = topk.values
            sel = topk.indices
            all_indices = all_indices[sel]
            all_labels = all_labels[sel]

        # Convert flat indices to (y, x) coordinates
        ys = (all_indices // W).float()
        xs = (all_indices % W).float()

        # Gather regression values
        offset_b = preds['offset'][b]   # (2, H, W)
        z_b = preds['z'][b]             # (1, H, W)
        dim_b = preds['dim'][b]         # (3, H, W)
        rot_b = preds['rot'][b]         # (2, H, W)

        y_idx = ys.long()
        x_idx = xs.long()

        off_x = offset_b[0, y_idx, x_idx]
        off_y = offset_b[1, y_idx, x_idx]
        z_val = z_b[0, y_idx, x_idx]
        dim_vals = torch.stack([dim_b[i, y_idx, x_idx] for i in range(3)], dim=1)  # (K, 3)
        rot_vals = torch.stack([rot_b[i, y_idx, x_idx] for i in range(2)], dim=1)  # (K, 2)

        # Decode to world coordinates
        cx = (xs + off_x) * res_x + pcr[0]
        cy = (ys + off_y) * res_y + pcr[1]
        cz = z_val

        w = torch.exp(dim_vals[:, 0]).clamp(max=500)
        l = torch.exp(dim_vals[:, 1]).clamp(max=500)
        h = torch.exp(dim_vals[:, 2]).clamp(max=500)

        yaw = torch.atan2(rot_vals[:, 0], rot_vals[:, 1])

        boxes = torch.stack([cx, cy, cz, w, l, h, yaw], dim=1)  # (K, 7)

        results.append({
            'boxes': boxes,
            'scores': all_scores,
            'labels': all_labels,
        })

    return results
