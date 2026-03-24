"""
Airbus Hackathon 2026 — LiDAR Notebook Helpers (Plotly)
========================================================
Usage in a Jupyter notebook:

    import lidar_notebook as ln

    # List all frames
    df, poses = ln.load("data/scene_01.h5")

    # Pick a frame and get local XYZ + class labels
    frame, xyz = ln.get_frame(df, poses, pose_index=3)

    # Interactive plots (display in notebook)
    ln.plot_bev(xyz, frame)
    ln.plot_3d(xyz, frame)

    # Save to file instead (PNG needs kaleido: pip install kaleido)
    ln.plot_bev(xyz, frame, save="plots/bev.png")
    ln.plot_3d(xyz, frame, save="plots/3d.html")   # .html keeps interactivity

    # Do both at once
    ln.plot_bev(xyz, frame, save="plots/bev.png", show=True)

    # Batch-save every view for a frame
    ln.save_all(xyz, frame, out_dir="plots", prefix="pose3")
    ln.save_all(xyz, frame, out_dir="plots", prefix="pose3", fmt="html")  # interactive
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import lidar_utils

# ─── Class definitions ────────────────────────────────────────────────────────
CLASS_MAP = {
    (38, 23, 180):  {"id": 0, "label": "Antenna",       "color": "#2617B4"},
    (177, 132, 47): {"id": 1, "label": "Cable",         "color": "#B1842F"},
    (129, 81, 97):  {"id": 2, "label": "Electric Pole", "color": "#815161"},
    (66, 132, 9):   {"id": 3, "label": "Wind Turbine",  "color": "#428409"},
}

CLASS_COLORS = {
    "Background":    "#888888",
    "Antenna":       "#2617B4",
    "Cable":         "#B1842F",
    "Electric Pole": "#815161",
    "Wind Turbine":  "#428409",
}


def _assign_classes(df):
    """Add class_id and class_label columns from RGB ground truth."""
    df = df.copy()
    df["class_id"] = -1
    df["class_label"] = "Background"
    for (r, g, b), info in CLASS_MAP.items():
        mask = (df["r"] == r) & (df["g"] == g) & (df["b"] == b)
        df.loc[mask, "class_id"] = info["id"]
        df.loc[mask, "class_label"] = info["label"]
    return df


# ─── Output helper ────────────────────────────────────────────────────────────
def _output(fig, save=None, show=None):
    """Show and/or save a Plotly figure.

    Parameters
    ----------
    fig  : plotly Figure
    save : str or None — file path (.png, .jpg, .svg, .pdf, .html)
           PNG/JPG/SVG/PDF require: pip install kaleido
           .html preserves full interactivity (no extra deps)
    show : bool or None
           None (default) → show only when save is NOT set
           True  → always show (even when saving)
           False → never show
    """
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".html":
            fig.write_html(str(path))
        else:
            fig.write_image(str(path), scale=2)
        print(f"  Saved → {path}")

    do_show = show if show is not None else (save is None)
    if do_show:
        fig.show()


# ─── Loading helpers ──────────────────────────────────────────────────────────
def load(file_path, dataset_name="lidar_points"):
    """Load an HDF5 file → (full_df, poses_df).

    Returns
    -------
    df : pd.DataFrame   — all points
    poses : pd.DataFrame — unique poses with pose_index & num_points (or None)
    """
    df = lidar_utils.load_h5_data(file_path, dataset_name)
    print(f"Loaded {len(df):,} points  |  columns: {list(df.columns)}")
    poses = lidar_utils.get_unique_poses(df)
    if poses is not None:
        print(f"Found {len(poses)} unique frames (poses)")
        display(poses[["pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points"]])
    return df, poses


def get_frame(df, poses, pose_index=0, drop_invalid=True):
    """Filter to a single frame → (frame_df, xyz).

    Parameters
    ----------
    df         : full dataframe from load()
    poses      : poses dataframe from load()
    pose_index : which frame to pick
    drop_invalid : remove distance_cm == 0 beams

    Returns
    -------
    frame : pd.DataFrame with class_id / class_label columns
    xyz   : np.ndarray (N, 3) in local Cartesian meters
    """
    if poses is not None:
        selected = poses.iloc[pose_index]
        frame = lidar_utils.filter_by_pose(df, selected)
    else:
        frame = df.copy()

    if drop_invalid:
        frame = frame[frame["distance_cm"] > 0].reset_index(drop=True)

    xyz = lidar_utils.spherical_to_local_cartesian(frame)

    if {"r", "g", "b"}.issubset(frame.columns):
        frame = _assign_classes(frame)

    print(f"Pose #{pose_index}  →  {len(frame):,} valid points")
    return frame, xyz


# ─── Summary ──────────────────────────────────────────────────────────────────
def summary(frame, xyz):
    """Print quick stats about a frame."""
    print(f"Points: {len(frame):,}")
    print(f"X: [{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}] m")
    print(f"Y: [{xyz[:,1].min():.1f}, {xyz[:,1].max():.1f}] m")
    print(f"Z: [{xyz[:,2].min():.1f}, {xyz[:,2].max():.1f}] m")
    if "class_label" in frame.columns:
        print("\nClass breakdown:")
        print(frame["class_label"].value_counts().to_string())


# ─── Plotly helpers ───────────────────────────────────────────────────────────
def _subsample(xyz, frame, max_points=200_000):
    """Randomly subsample if too many points (keeps Plotly responsive)."""
    if len(frame) <= max_points:
        return xyz, frame
    idx = np.random.choice(len(frame), max_points, replace=False)
    idx.sort()
    return xyz[idx], frame.iloc[idx].reset_index(drop=True)


def plot_bev(xyz, frame, max_points=200_000, point_size=1.5,
             title="Bird's Eye View", save=None, show=None):
    """Interactive top-down (X-Y) scatter colored by class."""
    xyz_s, fr = _subsample(xyz, frame, max_points)

    fig = go.Figure()

    bg = fr["class_label"] == "Background"
    if bg.any():
        fig.add_trace(go.Scattergl(
            x=xyz_s[bg, 0], y=xyz_s[bg, 1],
            mode="markers",
            marker=dict(size=point_size, color=CLASS_COLORS["Background"], opacity=0.2),
            name=f"Background ({bg.sum():,})",
        ))

    for label, color in CLASS_COLORS.items():
        if label == "Background":
            continue
        mask = fr["class_label"] == label
        if mask.any():
            fig.add_trace(go.Scattergl(
                x=xyz_s[mask, 0], y=xyz_s[mask, 1],
                mode="markers",
                marker=dict(size=point_size + 1, color=color),
                name=f"{label} ({mask.sum():,})",
            ))

    fig.update_layout(
        title=title, xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x"), width=800, height=800,
        template="plotly_dark",
    )
    _output(fig, save, show)


def plot_side(xyz, frame, max_points=200_000, point_size=1.5,
              title="Side View (X-Z)", save=None, show=None):
    """Interactive side view (X-Z) colored by class."""
    xyz_s, fr = _subsample(xyz, frame, max_points)

    fig = go.Figure()

    bg = fr["class_label"] == "Background"
    if bg.any():
        fig.add_trace(go.Scattergl(
            x=xyz_s[bg, 0], y=xyz_s[bg, 2],
            mode="markers",
            marker=dict(size=point_size, color=CLASS_COLORS["Background"], opacity=0.2),
            name=f"Background ({bg.sum():,})",
        ))

    for label, color in CLASS_COLORS.items():
        if label == "Background":
            continue
        mask = fr["class_label"] == label
        if mask.any():
            fig.add_trace(go.Scattergl(
                x=xyz_s[mask, 0], y=xyz_s[mask, 2],
                mode="markers",
                marker=dict(size=point_size + 1, color=color),
                name=f"{label} ({mask.sum():,})",
            ))

    fig.update_layout(
        title=title, xaxis_title="X (m)", yaxis_title="Z (m)",
        width=1000, height=400, template="plotly_dark",
    )
    _output(fig, save, show)


def plot_bev_reflectivity(xyz, frame, max_points=200_000, point_size=1.5,
                          title="BEV — Reflectivity", save=None, show=None):
    """BEV colored by reflectivity (what you'll have at eval time)."""
    xyz_s, fr = _subsample(xyz, frame, max_points)

    fig = go.Figure(go.Scattergl(
        x=xyz_s[:, 0], y=xyz_s[:, 1],
        mode="markers",
        marker=dict(
            size=point_size,
            color=fr["reflectivity"],
            colorscale="Turbo",
            colorbar=dict(title="Reflectivity"),
            opacity=0.6,
        ),
        hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Refl: %{marker.color}<extra></extra>",
    ))

    fig.update_layout(
        title=title, xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x"), width=800, height=800,
        template="plotly_dark",
    )
    _output(fig, save, show)


def plot_3d(xyz, frame, max_points=100_000, point_size=1,
            title="3D Point Cloud", save=None, show=None):
    """Full interactive 3D scatter colored by class."""
    xyz_s, fr = _subsample(xyz, frame, max_points)

    fig = go.Figure()

    bg = fr["class_label"] == "Background"
    if bg.any():
        fig.add_trace(go.Scatter3d(
            x=xyz_s[bg, 0], y=xyz_s[bg, 1], z=xyz_s[bg, 2],
            mode="markers",
            marker=dict(size=point_size, color=CLASS_COLORS["Background"], opacity=0.15),
            name=f"Background ({bg.sum():,})",
        ))

    for label, color in CLASS_COLORS.items():
        if label == "Background":
            continue
        mask = fr["class_label"] == label
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=xyz_s[mask, 0], y=xyz_s[mask, 1], z=xyz_s[mask, 2],
                mode="markers",
                marker=dict(size=point_size + 0.5, color=color),
                name=f"{label} ({mask.sum():,})",
            ))

    fig.update_layout(
        title=title, width=900, height=700,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        template="plotly_dark",
    )
    _output(fig, save, show)


def plot_3d_reflectivity(xyz, frame, max_points=100_000, point_size=1,
                         title="3D — Reflectivity", save=None, show=None):
    """Full 3D scatter colored by reflectivity."""
    xyz_s, fr = _subsample(xyz, frame, max_points)

    fig = go.Figure(go.Scatter3d(
        x=xyz_s[:, 0], y=xyz_s[:, 1], z=xyz_s[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color=fr["reflectivity"],
            colorscale="Turbo",
            colorbar=dict(title="Reflectivity"),
            opacity=0.6,
        ),
        hovertemplate="X:%{x:.1f} Y:%{y:.1f} Z:%{z:.1f}<br>Refl:%{marker.color}<extra></extra>",
    ))

    fig.update_layout(
        title=title, width=900, height=700,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        template="plotly_dark",
    )
    _output(fig, save, show)


def plot_class_dist(frame, title="Class Distribution", save=None, show=None):
    """Horizontal bar chart of point counts per class."""
    counts = frame["class_label"].value_counts()
    labels = ["Background", "Antenna", "Cable", "Electric Pole", "Wind Turbine"]
    vals = [counts.get(l, 0) for l in labels]
    colors = [CLASS_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker_color=colors,
        text=[f"{v:,}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title="Number of Points",
        width=700, height=350, template="plotly_dark",
    )
    _output(fig, save, show)


def plot_obstacles_only(xyz, frame, max_points=200_000, point_size=2,
                        title="Obstacles Only (no background)", save=None, show=None):
    """3D scatter with ONLY obstacle classes — much easier to see structure."""
    mask = frame["class_id"] >= 0
    xyz_obs = xyz[mask]
    fr_obs = frame[mask].reset_index(drop=True)

    if len(fr_obs) == 0:
        print("No obstacles in this frame.")
        return

    xyz_s, fr = _subsample(xyz_obs, fr_obs, max_points)

    fig = go.Figure()
    for label, color in CLASS_COLORS.items():
        if label == "Background":
            continue
        m = fr["class_label"] == label
        if m.any():
            fig.add_trace(go.Scatter3d(
                x=xyz_s[m, 0], y=xyz_s[m, 1], z=xyz_s[m, 2],
                mode="markers",
                marker=dict(size=point_size, color=color),
                name=f"{label} ({m.sum():,})",
            ))

    fig.update_layout(
        title=title, width=900, height=700,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        template="plotly_dark",
    )
    _output(fig, save, show)


# ─── Batch save ───────────────────────────────────────────────────────────────
def save_all(xyz, frame, out_dir="plots", prefix="frame", fmt="png"):
    """Save every plot to files in one call.

    Parameters
    ----------
    out_dir : output directory (created automatically)
    prefix  : filename prefix, e.g. "pose3"
    fmt     : "png", "svg", "pdf", or "html"
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ext = fmt if fmt.startswith(".") else f".{fmt}"

    plot_bev(xyz, frame,              save=str(d / f"{prefix}_bev_class{ext}"))
    plot_side(xyz, frame,             save=str(d / f"{prefix}_side_class{ext}"))
    plot_bev_reflectivity(xyz, frame, save=str(d / f"{prefix}_bev_refl{ext}"))
    plot_3d(xyz, frame,               save=str(d / f"{prefix}_3d_class{ext}"))
    plot_3d_reflectivity(xyz, frame,  save=str(d / f"{prefix}_3d_refl{ext}"))
    plot_class_dist(frame,            save=str(d / f"{prefix}_class_dist{ext}"))
    plot_obstacles_only(xyz, frame,   save=str(d / f"{prefix}_obstacles{ext}"))

    print(f"\nAll 7 plots saved to {d}/")





############################ Function VIZU POST preprocessing ################

"""
Plotly-based helpers to visually inspect preprocessed .npz frames
and the ground-truth bounding boxes produced by generate_gt_bboxes.py.

Usage in a Jupyter notebook:

    

    # ── Single-frame inspection (point cloud + bboxes) ──────────────
    ln.plot_bev_bboxes("scene_1", 12)          # Bird's Eye View
    ln.plot_side_bboxes("scene_1", 12)         # Side View (X-Z)
    ln.plot_3d_bboxes("scene_1", 12)           # Full 3D
    ln.plot_obstacles_bboxes("scene_1", 12)    # Obstacles only (no bg)

    # ── Dataset-wide quality checks ─────────────────────────────────
    ln.plot_bbox_sizes()                # Violin of bbox dims per class
    ln.plot_objects_per_frame()         # Heatmap: #objects per frame
    ln.plot_class_balance()             # Stacked bar per scene
    ln.qa_report()                     # Print text summary of issues

    # ── Customise paths if needed ───────────────────────────────────
    ln.PROCESSED_DIR = "my_processed/"
    ln.BBOXES_CSV    = "my_gt_bboxes.csv"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — change these if your paths differ
# ═══════════════════════════════════════════════════════════════════════════════
PROCESSED_DIR = "processed/"
BBOXES_CSV = "gt_bboxes.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
CLASS_COLORS = {
    "Background":    "#888888",
    "Antenna":       "#2617B4",
    "Cable":         "#B1842F",
    "Electric Pole": "#815161",
    "Wind Turbine":  "#428409",
}
CLASS_ID_TO_LABEL = {0: "Antenna", 1: "Cable", 2: "Electric Pole", 3: "Wind Turbine"}
CLASS_RGB = {
    (38, 23, 180):  0,
    (177, 132, 47): 1,
    (129, 81, 97):  2,
    (66, 132, 9):   3,
}

# Brighter bbox edge colors (stand out against dark background)
BBOX_COLORS = {
    "Antenna":       "#5B4FFF",
    "Cable":         "#FFD060",
    "Electric Pole": "#FF6B8A",
    "Wind Turbine":  "#40FF40",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════════
def load_frame(scene, pose_index, processed_dir=None):
    """Load a preprocessed .npz frame.

    Returns
    -------
    xyz          : (N, 3) float32
    reflectivity : (N,)   uint8
    rgb          : (N, 3) uint8
    ego_pose     : (4,)   float64
    class_ids    : (N,)   int32  — -1=Background, 0-3=obstacle
    class_labels : list[str]     — per-point label strings
    """
    d = processed_dir or PROCESSED_DIR
    path = Path(d) / scene / f"frame_{pose_index:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frame not found: {path}")
    data = np.load(path)
    xyz = data["xyz"]
    reflectivity = data["reflectivity"]
    rgb = data["rgb"]
    ego_pose = data["ego_pose"]

    # Assign classes
    class_ids = np.full(len(rgb), -1, dtype=np.int32)
    for (r, g, b), cid in CLASS_RGB.items():
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        class_ids[mask] = cid
    class_labels = ["Background" if c == -1 else CLASS_ID_TO_LABEL[c] for c in class_ids]

    return xyz, reflectivity, rgb, ego_pose, class_ids, class_labels


def load_bboxes(scene=None, pose_index=None, bboxes_csv=None):
    """Load bounding boxes, optionally filtered by scene and/or pose.

    Returns pd.DataFrame with columns:
        bbox_center_x/y/z, bbox_width/length/height, bbox_yaw,
        class_ID, class_label, num_points, scene, pose_index, ...
    """
    csv = bboxes_csv or BBOXES_CSV
    df = pd.read_csv(csv)
    if scene is not None:
        df = df[df["scene"] == scene]
    if pose_index is not None:
        df = df[df["pose_index"] == pose_index]
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _bbox_corners_2d(row):
    """Compute 4 BEV corners of an oriented bbox → (5, 2) array (closed loop)."""
    cx, cy = row["bbox_center_x"], row["bbox_center_y"]
    w, l = row["bbox_width"], row["bbox_length"]
    yaw = row["bbox_yaw"]

    # Half extents in local frame
    dx, dy = w / 2, l / 2
    corners_local = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy],
        [-dx, -dy],  # close the rectangle
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_world = (R @ corners_local.T).T + np.array([cx, cy])
    return corners_world


def _bbox_wireframe_3d(row):
    """Compute 12-edge wireframe of a 3D oriented bbox.

    Returns (xs, ys, zs) lists with None separators for Plotly line traces.
    """
    cx = row["bbox_center_x"]
    cy = row["bbox_center_y"]
    cz = row["bbox_center_z"]
    w, l, h = row["bbox_width"], row["bbox_length"], row["bbox_height"]
    yaw = row["bbox_yaw"]

    dx, dy, dz = w / 2, l / 2, h / 2

    # 8 corners in local frame
    corners = np.array([
        [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz],
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = (R @ corners.T).T + np.array([cx, cy, cz])

    # 12 edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]

    return xs, ys, zs


def _subsample(xyz, class_labels, max_points=200_000):
    """Randomly subsample points to keep Plotly responsive."""
    if len(xyz) <= max_points:
        return xyz, class_labels
    idx = np.random.choice(len(xyz), max_points, replace=False)
    idx.sort()
    return xyz[idx], [class_labels[i] for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Single-frame plots with bounding boxes
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bev_bboxes(scene, pose_index, max_points=200_000, point_size=1.5,
                    processed_dir=None, bboxes_csv=None, bg_opacity=0.15):
    """Bird's Eye View with oriented bounding-box rectangles overlaid."""

    xyz, _, _, ego, cids, clabels = load_frame(scene, pose_index, processed_dir)
    bboxes = load_bboxes(scene, pose_index, bboxes_csv)
    xyz_s, cl_s = _subsample(xyz, clabels, max_points)
    cl_arr = np.array(cl_s)

    fig = go.Figure()

    # Points — background
    bg = cl_arr == "Background"
    if bg.any():
        fig.add_trace(go.Scattergl(
            x=xyz_s[bg, 0], y=xyz_s[bg, 1], mode="markers",
            marker=dict(size=point_size, color=CLASS_COLORS["Background"], opacity=bg_opacity),
            name=f"Background ({bg.sum():,})",
        ))

    # Points — obstacles
    for label, color in CLASS_COLORS.items():
        if label == "Background":
            continue
        m = cl_arr == label
        if m.any():
            fig.add_trace(go.Scattergl(
                x=xyz_s[m, 0], y=xyz_s[m, 1], mode="markers",
                marker=dict(size=point_size + 1, color=color),
                name=f"{label} ({m.sum():,})",
            ))

    # Bounding boxes
    for _, row in bboxes.iterrows():
        corners = _bbox_corners_2d(row)
        color = BBOX_COLORS.get(row["class_label"], "#FFFFFF")
        fig.add_trace(go.Scattergl(
            x=corners[:, 0], y=corners[:, 1], mode="lines",
            line=dict(color=color, width=2),
            name=f"bbox {row['class_label']} ({row['num_points']} pts)",
            showlegend=False,
            hovertext=(f"{row['class_label']}<br>"
                       f"size: {row['bbox_width']:.1f}×{row['bbox_length']:.1f}×{row['bbox_height']:.1f}m<br>"
                       f"pts: {row['num_points']}"),
            hoverinfo="text",
        ))

    n_obj = len(bboxes)
    fig.update_layout(
        title=f"BEV — {scene} pose {pose_index}  ({n_obj} objects)",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x"), width=900, height=900,
        template="plotly_dark",
    )
    fig.show()


def plot_side_bboxes(scene, pose_index, max_points=200_000, point_size=1.5,
                     processed_dir=None, bboxes_csv=None, bg_opacity=0.15):
    """Side view (X-Z) with bounding-box rectangles overlaid."""

    xyz, _, _, ego, cids, clabels = load_frame(scene, pose_index, processed_dir)
    bboxes = load_bboxes(scene, pose_index, bboxes_csv)
    xyz_s, cl_s = _subsample(xyz, clabels, max_points)
    cl_arr = np.array(cl_s)

    fig = go.Figure()

    # Points
    bg = cl_arr == "Background"
    if bg.any():
        fig.add_trace(go.Scattergl(
            x=xyz_s[bg, 0], y=xyz_s[bg, 2], mode="markers",
            marker=dict(size=point_size, color=CLASS_COLORS["Background"], opacity=bg_opacity),
            name=f"Background ({bg.sum():,})",
        ))
    for label, color in CLASS_COLORS.items():
        if label == "Background":
            continue
        m = cl_arr == label
        if m.any():
            fig.add_trace(go.Scattergl(
                x=xyz_s[m, 0], y=xyz_s[m, 2], mode="markers",
                marker=dict(size=point_size + 1, color=color),
                name=f"{label} ({m.sum():,})",
            ))

    # Bboxes as X-Z rectangles (axis-aligned projection)
    for _, row in bboxes.iterrows():
        cx, cz = row["bbox_center_x"], row["bbox_center_z"]
        # Project: use width along X, height along Z
        hw = row["bbox_width"] / 2
        hh = row["bbox_height"] / 2
        rx = [cx - hw, cx + hw, cx + hw, cx - hw, cx - hw]
        rz = [cz - hh, cz - hh, cz + hh, cz + hh, cz - hh]
        color = BBOX_COLORS.get(row["class_label"], "#FFFFFF")
        fig.add_trace(go.Scattergl(
            x=rx, y=rz, mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hovertext=f"{row['class_label']} — {row['bbox_width']:.1f}×{row['bbox_height']:.1f}m",
            hoverinfo="text",
        ))

    fig.update_layout(
        title=f"Side View — {scene} pose {pose_index}",
        xaxis_title="X (m)", yaxis_title="Z (m)",
        width=1000, height=450, template="plotly_dark",
    )
    fig.show()


def plot_3d_bboxes(scene, pose_index, max_points=100_000, point_size=1,
                   processed_dir=None, bboxes_csv=None, bg_opacity=0.10):
    """Full 3D scatter with wireframe bounding boxes."""

    xyz, _, _, ego, cids, clabels = load_frame(scene, pose_index, processed_dir)
    bboxes = load_bboxes(scene, pose_index, bboxes_csv)
    xyz_s, cl_s = _subsample(xyz, clabels, max_points)
    cl_arr = np.array(cl_s)

    fig = go.Figure()

    # Background
    bg = cl_arr == "Background"
    if bg.any():
        fig.add_trace(go.Scatter3d(
            x=xyz_s[bg, 0], y=xyz_s[bg, 1], z=xyz_s[bg, 2], mode="markers",
            marker=dict(size=point_size, color=CLASS_COLORS["Background"], opacity=bg_opacity),
            name=f"Background ({bg.sum():,})",
        ))

    # Obstacles
    for label, color in CLASS_COLORS.items():
        if label == "Background":
            continue
        m = cl_arr == label
        if m.any():
            fig.add_trace(go.Scatter3d(
                x=xyz_s[m, 0], y=xyz_s[m, 1], z=xyz_s[m, 2], mode="markers",
                marker=dict(size=point_size + 0.5, color=color),
                name=f"{label} ({m.sum():,})",
            ))

    # 3D wireframe boxes
    for _, row in bboxes.iterrows():
        xs, ys, zs = _bbox_wireframe_3d(row)
        color = BBOX_COLORS.get(row["class_label"], "#FFFFFF")
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=3),
            name=f"bbox {row['class_label']}",
            showlegend=False,
            hovertext=f"{row['class_label']} — {row['bbox_width']:.1f}×{row['bbox_length']:.1f}×{row['bbox_height']:.1f}m",
            hoverinfo="text",
        ))

    fig.update_layout(
        title=f"3D — {scene} pose {pose_index}  ({len(bboxes)} objects)",
        width=950, height=750,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                   aspectmode="data"),
        template="plotly_dark",
    )
    fig.show()


def plot_obstacles_bboxes(scene, pose_index, max_points=200_000, point_size=2,
                          processed_dir=None, bboxes_csv=None):
    """3D scatter of obstacle points only (no background) with wireframe boxes.
    Much cleaner view to check bbox fit quality."""

    xyz, _, _, ego, cids, clabels = load_frame(scene, pose_index, processed_dir)
    bboxes = load_bboxes(scene, pose_index, bboxes_csv)

    # Keep only obstacle points
    cids_arr = np.array(cids)
    obs_mask = cids_arr >= 0
    xyz_obs = xyz[obs_mask]
    cl_obs = [clabels[i] for i in np.where(obs_mask)[0]]

    if len(xyz_obs) == 0:
        print(f"No obstacle points in {scene} pose {pose_index}.")
        return

    xyz_s, cl_s = _subsample(xyz_obs, cl_obs, max_points)
    cl_arr = np.array(cl_s)

    fig = go.Figure()

    for label, color in CLASS_COLORS.items():
        if label == "Background":
            continue
        m = cl_arr == label
        if m.any():
            fig.add_trace(go.Scatter3d(
                x=xyz_s[m, 0], y=xyz_s[m, 1], z=xyz_s[m, 2], mode="markers",
                marker=dict(size=point_size, color=color),
                name=f"{label} ({m.sum():,})",
            ))

    for _, row in bboxes.iterrows():
        xs, ys, zs = _bbox_wireframe_3d(row)
        color = BBOX_COLORS.get(row["class_label"], "#FFFFFF")
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=3),
            name=f"bbox {row['class_label']}",
            showlegend=False,
        ))

    fig.update_layout(
        title=f"Obstacles + Bboxes — {scene} pose {pose_index}  ({len(bboxes)} objects)",
        width=950, height=750,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                   aspectmode="data"),
        template="plotly_dark",
    )
    fig.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset-wide quality plots
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bbox_sizes(bboxes_csv=None):
    """Violin + strip plot of bounding-box dimensions per class.
    Great for spotting outliers or misclustered objects."""

    df = load_bboxes(bboxes_csv=bboxes_csv)

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Width (m)", "Length (m)", "Height (m)"])
    dims = ["bbox_width", "bbox_length", "bbox_height"]

    for col_idx, dim in enumerate(dims, 1):
        for label in ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]:
            subset = df[df["class_label"] == label]
            fig.add_trace(go.Violin(
                y=subset[dim], name=label,
                marker_color=BBOX_COLORS[label],
                box_visible=True, meanline_visible=True,
                legendgroup=label, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(
        title="Bounding-Box Size Distributions by Class",
        width=1200, height=500, template="plotly_dark",
        violinmode="group",
    )
    fig.show()


def plot_objects_per_frame(bboxes_csv=None):
    """Heatmap: number of detected objects per (scene, pose_index).
    Helps spot frames with 0 detections or suspiciously many."""

    df = load_bboxes(bboxes_csv=bboxes_csv)
    counts = df.groupby(["scene", "pose_index"]).size().reset_index(name="n_objects")

    scenes = sorted(counts["scene"].unique())
    max_pose = int(counts["pose_index"].max()) + 1

    z = np.zeros((len(scenes), max_pose))
    for _, row in counts.iterrows():
        si = scenes.index(row["scene"])
        z[si, int(row["pose_index"])] = row["n_objects"]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(max_pose)),
        y=scenes,
        colorscale="YlOrRd",
        colorbar=dict(title="# objects"),
        hovertemplate="scene: %{y}<br>pose: %{x}<br>objects: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Objects per Frame",
        xaxis_title="Pose Index", yaxis_title="Scene",
        width=1100, height=400, template="plotly_dark",
    )
    fig.show()


def plot_class_balance(bboxes_csv=None):
    """Stacked bar chart: object count per class per scene."""

    df = load_bboxes(bboxes_csv=bboxes_csv)
    pivot = df.groupby(["scene", "class_label"]).size().unstack(fill_value=0)

    fig = go.Figure()
    for label in ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]:
        if label in pivot.columns:
            fig.add_trace(go.Bar(
                x=pivot.index, y=pivot[label], name=label,
                marker_color=BBOX_COLORS[label],
            ))

    fig.update_layout(
        barmode="stack",
        title="Object Count per Class per Scene",
        xaxis_title="Scene", yaxis_title="# Objects",
        width=900, height=450, template="plotly_dark",
    )
    fig.show()


def plot_points_per_class(bboxes_csv=None):
    """Box plot of num_points per bbox, grouped by class.
    Useful to check if clusters are too small / too large."""

    df = load_bboxes(bboxes_csv=bboxes_csv)

    fig = go.Figure()
    for label in ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]:
        subset = df[df["class_label"] == label]
        fig.add_trace(go.Box(
            y=subset["num_points"], name=label,
            marker_color=BBOX_COLORS[label],
            boxmean=True,
        ))

    fig.update_layout(
        title="Points per Bounding Box by Class",
        yaxis_title="num_points", yaxis_type="log",
        width=800, height=450, template="plotly_dark",
    )
    fig.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Text-based QA report
# ═══════════════════════════════════════════════════════════════════════════════
def qa_report(processed_dir=None, bboxes_csv=None):
    """Print a text QA summary flagging potential issues."""

    d = processed_dir or PROCESSED_DIR
    csv = bboxes_csv or BBOXES_CSV

    print("=" * 70)
    print("  QA REPORT — Preprocessing & Bounding Boxes")
    print("=" * 70)

    # ── 1. Check .npz completeness ──
    print("\n── NPZ Completeness ──")
    from collections import defaultdict
    import glob as _glob

    npz_counts = defaultdict(int)
    for p in sorted(_glob.glob(str(Path(d) / "scene_*" / "frame_*.npz"))):
        scene = Path(p).parent.name
        npz_counts[scene] += 1

    all_ok = True
    for scene in sorted(npz_counts):
        n = npz_counts[scene]
        status = "✓" if n == 100 else f"⚠ MISSING {100 - n} frames"
        if n != 100:
            all_ok = False
        print(f"  {scene:15s}: {n:3d} / 100  {status}")
    if all_ok:
        print("  All scenes complete ✓")

    # ── 2. Bbox stats ──
    print("\n── Bounding Box Summary ──")
    try:
        df = pd.read_csv(csv)
    except FileNotFoundError:
        print(f"  ⚠ {csv} not found — run generate_gt_bboxes.py first")
        return

    n_scenes = df["scene"].nunique()
    n_frames = df.groupby(["scene", "pose_index"]).ngroups
    print(f"  Total objects : {len(df):,}")
    print(f"  Scenes        : {n_scenes}")
    print(f"  Frames w/ obj : {n_frames}")
    print(f"\n  Per class:")
    for label in ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]:
        sub = df[df["class_label"] == label]
        if len(sub) > 0:
            print(f"    {label:15s}: {len(sub):5d} objects  |  "
                  f"median size: {sub['bbox_width'].median():.1f} × "
                  f"{sub['bbox_length'].median():.1f} × "
                  f"{sub['bbox_height'].median():.1f} m  |  "
                  f"median pts: {sub['num_points'].median():.0f}")

    # ── 3. Flag outliers ──
    print("\n── Potential Issues ──")
    issues = 0

    # Tiny boxes
    tiny = df[(df["bbox_width"] < 0.5) & (df["bbox_length"] < 0.5) & (df["bbox_height"] < 0.5)]
    if len(tiny) > 0:
        issues += 1
        print(f"  ⚠ {len(tiny)} very small bboxes (all dims < 0.5m):")
        for _, r in tiny.head(5).iterrows():
            print(f"      {r['scene']} pose {r['pose_index']} — {r['class_label']}  "
                  f"({r['bbox_width']:.2f}×{r['bbox_length']:.2f}×{r['bbox_height']:.2f}m, "
                  f"{r['num_points']} pts)")
        if len(tiny) > 5:
            print(f"      ... and {len(tiny) - 5} more")

    # Huge boxes
    huge = df[(df["bbox_width"] > 200) | (df["bbox_length"] > 200) | (df["bbox_height"] > 200)]
    if len(huge) > 0:
        issues += 1
        print(f"  ⚠ {len(huge)} very large bboxes (any dim > 200m):")
        for _, r in huge.head(5).iterrows():
            print(f"      {r['scene']} pose {r['pose_index']} — {r['class_label']}  "
                  f"({r['bbox_width']:.1f}×{r['bbox_length']:.1f}×{r['bbox_height']:.1f}m, "
                  f"{r['num_points']} pts)")
        if len(huge) > 5:
            print(f"      ... and {len(huge) - 5} more")

    # Low-point boxes
    low_pts = df[df["num_points"] < 20]
    if len(low_pts) > 0:
        issues += 1
        print(f"  ⚠ {len(low_pts)} bboxes with < 20 points (possibly noise)")

    # Frames with many objects (clustering might be wrong)
    per_frame = df.groupby(["scene", "pose_index"]).size()
    crowded = per_frame[per_frame > 30]
    if len(crowded) > 0:
        issues += 1
        print(f"  ⚠ {len(crowded)} frames with > 30 objects (check clustering):")
        for (sc, pi), cnt in crowded.head(5).items():
            print(f"      {sc} pose {pi}: {cnt} objects")

    # Frames with 0 objects (might be expected for some scenes)
    manifest_path = Path(d) / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        all_frames = set(zip(manifest["scene"], manifest["pose_index"]))
        bbox_frames = set(zip(df["scene"], df["pose_index"]))
        empty = all_frames - bbox_frames
        if len(empty) > 0:
            issues += 1
            print(f"  ⚠ {len(empty)} frames with 0 detected objects")
            for sc, pi in sorted(empty)[:5]:
                print(f"      {sc} pose {pi}")
            if len(empty) > 5:
                print(f"      ... and {len(empty) - 5} more")

    if issues == 0:
        print("  No issues found ✓")

    print("\n" + "=" * 70)



######################## Compare different run of bbox process #################

    """
Side-by-side comparison of bounding boxes from two different runs.

Drop this into lidar_notebook.py (after plot_side_bboxes) or import it
directly in your notebook:

    from compare_side_bboxes import compare_side_bboxes

    compare_side_bboxes(
        "scene_1", 12,
        bboxes_csv_a="gt_bboxes_v1.csv",
        bboxes_csv_b="gt_bboxes_v2.csv",
        label_a="Baseline (eps=3)",
        label_b="Tuned (eps=5, merged)",
    )
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# ─── Copy these from lidar_notebook.py (or import them) ──────────────────────
PROCESSED_DIR = "processed/"
BBOXES_CSV = "gt_bboxes.csv"

CLASS_COLORS = {
    "Background":    "#888888",
    "Antenna":       "#2617B4",
    "Cable":         "#B1842F",
    "Electric Pole": "#815161",
    "Wind Turbine":  "#428409",
}
CLASS_ID_TO_LABEL = {0: "Antenna", 1: "Cable", 2: "Electric Pole", 3: "Wind Turbine"}
CLASS_RGB = {
    (38, 23, 180):  0,
    (177, 132, 47): 1,
    (129, 81, 97):  2,
    (66, 132, 9):   3,
}
BBOX_COLORS = {
    "Antenna":       "#5B4FFF",
    "Cable":         "#FFD060",
    "Electric Pole": "#FF6B8A",
    "Wind Turbine":  "#40FF40",
}


# ─── Loaders (same as lidar_notebook.py) ─────────────────────────────────────
def _load_frame(scene, pose_index, processed_dir=None):
    d = processed_dir or PROCESSED_DIR
    path = Path(d) / scene / f"frame_{pose_index:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frame not found: {path}")
    data = np.load(path)
    xyz = data["xyz"]
    rgb = data["rgb"]
    class_ids = np.full(len(rgb), -1, dtype=np.int32)
    for (r, g, b), cid in CLASS_RGB.items():
        mask = (rgb[:, 0] == r) & (rgb[:, 1] == g) & (rgb[:, 2] == b)
        class_ids[mask] = cid
    class_labels = ["Background" if c == -1 else CLASS_ID_TO_LABEL[c] for c in class_ids]
    return xyz, class_ids, class_labels


def _load_bboxes(scene, pose_index, bboxes_csv):
    df = pd.read_csv(bboxes_csv)
    df = df[(df["scene"] == scene) & (df["pose_index"] == pose_index)]
    return df.reset_index(drop=True)


def _subsample(xyz, class_labels, max_points):
    if len(xyz) <= max_points:
        return xyz, class_labels
    idx = np.random.choice(len(xyz), max_points, replace=False)
    idx.sort()
    return xyz[idx], [class_labels[i] for i in idx]


# ─── Main comparison function ────────────────────────────────────────────────
def compare_side_bboxes(
    scene,
    pose_index,
    bboxes_csv_a,
    bboxes_csv_b,
    label_a="Run A",
    label_b="Run B",
    max_points=200_000,
    point_size=1.5,
    processed_dir=None,
    bg_opacity=0.15,
):
    """Side-by-side comparison of two bbox runs on the same frame (X-Z view).

    Parameters
    ----------
    scene        : str   — scene name, e.g. "scene_1"
    pose_index   : int   — frame index
    bboxes_csv_a : str   — path to first bbox CSV
    bboxes_csv_b : str   — path to second bbox CSV
    label_a      : str   — display name for first run
    label_b      : str   — display name for second run
    max_points   : int   — subsample limit for Plotly performance
    point_size   : float — marker size
    processed_dir: str   — path to preprocessed .npz directory
    bg_opacity   : float — opacity for background points
    """

    # ── Load shared point cloud ──
    xyz, cids, clabels = _load_frame(scene, pose_index, processed_dir)

    # ── Load both bbox sets ──
    bboxes_a = _load_bboxes(scene, pose_index, bboxes_csv_a)
    bboxes_b = _load_bboxes(scene, pose_index, bboxes_csv_b)

    # ── Subsample (same seed → identical points on both panels) ──
    rng_state = np.random.get_state()
    np.random.seed(42)
    xyz_s, cl_s = _subsample(xyz, clabels, max_points)
    np.random.set_state(rng_state)
    cl_arr = np.array(cl_s)

    # ── Build subplots ──
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{label_a}  ({len(bboxes_a)} objects)",
            f"{label_b}  ({len(bboxes_b)} objects)",
        ],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    # ── Helper: add point traces to a subplot column ──
    def _add_points(col, show_legend):
        bg = cl_arr == "Background"
        if bg.any():
            fig.add_trace(
                go.Scattergl(
                    x=xyz_s[bg, 0], y=xyz_s[bg, 2], mode="markers",
                    marker=dict(size=point_size, color=CLASS_COLORS["Background"],
                                opacity=bg_opacity),
                    name=f"Background ({bg.sum():,})",
                    showlegend=show_legend,
                    legendgroup="Background",
                ),
                row=1, col=col,
            )
        for label, color in CLASS_COLORS.items():
            if label == "Background":
                continue
            m = cl_arr == label
            if m.any():
                fig.add_trace(
                    go.Scattergl(
                        x=xyz_s[m, 0], y=xyz_s[m, 2], mode="markers",
                        marker=dict(size=point_size + 1, color=color),
                        name=f"{label} ({m.sum():,})",
                        showlegend=show_legend,
                        legendgroup=label,
                    ),
                    row=1, col=col,
                )

    # ── Helper: add bbox rectangles to a subplot column ──
    def _add_bboxes(bboxes_df, col, show_legend):
        for _, row in bboxes_df.iterrows():
            cx, cz = row["bbox_center_x"], row["bbox_center_z"]
            hw = row["bbox_width"] / 2
            hh = row["bbox_height"] / 2
            rx = [cx - hw, cx + hw, cx + hw, cx - hw, cx - hw]
            rz = [cz - hh, cz - hh, cz + hh, cz + hh, cz - hh]
            clabel = row["class_label"]
            color = BBOX_COLORS.get(clabel, "#FFFFFF")
            fig.add_trace(
                go.Scattergl(
                    x=rx, y=rz, mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertext=(
                        f"{clabel}<br>"
                        f"size: {row['bbox_width']:.1f}×{row['bbox_height']:.1f}m<br>"
                        f"pts: {row.get('num_points', '?')}"
                    ),
                    hoverinfo="text",
                ),
                row=1, col=col,
            )

    # ── Populate both panels ──
    _add_points(col=1, show_legend=True)
    _add_bboxes(bboxes_a, col=1, show_legend=False)

    _add_points(col=2, show_legend=False)
    _add_bboxes(bboxes_b, col=2, show_legend=False)

    # ── Shared axis ranges so both views are perfectly aligned ──
    x_all = xyz_s[:, 0]
    z_all = xyz_s[:, 2]
    x_pad = (x_all.max() - x_all.min()) * 0.02
    z_pad = (z_all.max() - z_all.min()) * 0.02
    xrange = [x_all.min() - x_pad, x_all.max() + x_pad]
    zrange = [z_all.min() - z_pad, z_all.max() + z_pad]

    fig.update_xaxes(range=xrange, title_text="X (m)", row=1, col=1)
    fig.update_xaxes(range=xrange, title_text="X (m)", row=1, col=2)
    fig.update_yaxes(range=zrange, title_text="Z (m)", row=1, col=1)
    fig.update_yaxes(range=zrange, row=1, col=2)

    # ── Summary line ──
    n_a, n_b = len(bboxes_a), len(bboxes_b)
    diff = n_b - n_a
    diff_str = f"+{diff}" if diff >= 0 else str(diff)

    fig.update_layout(
        title=(
            f"Side View Comparison — {scene} pose {pose_index}    "
            f"({label_a}: {n_a} obj  vs  {label_b}: {n_b} obj  [{diff_str}])"
        ),
        width=1800,
        height=500,
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )
    fig.show()

    # ── Print per-class delta table ──
    _print_delta_table(bboxes_a, bboxes_b, label_a, label_b)


def _print_delta_table(bboxes_a, bboxes_b, label_a, label_b):
    """Print a compact per-class comparison table."""
    classes = ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]
    print(f"\n{'Class':17s} | {label_a:>8s} | {label_b:>8s} | {'Delta':>6s}")
    print("-" * 50)
    total_a, total_b = 0, 0
    for cls in classes:
        ca = len(bboxes_a[bboxes_a["class_label"] == cls])
        cb = len(bboxes_b[bboxes_b["class_label"] == cls])
        d = cb - ca
        ds = f"+{d}" if d >= 0 else str(d)
        print(f"{cls:17s} | {ca:8d} | {cb:8d} | {ds:>6s}")
        total_a += ca
        total_b += cb
    d = total_b - total_a
    ds = f"+{d}" if d >= 0 else str(d)
    print("-" * 50)
    print(f"{'TOTAL':17s} | {total_a:8d} | {total_b:8d} | {ds:>6s}")


# ─── Batch comparison across many frames ─────────────────────────────────────
def compare_side_bboxes_batch(
    bboxes_csv_a,
    bboxes_csv_b,
    label_a="Run A",
    label_b="Run B",
    n_frames=5,
    processed_dir=None,
    **kwargs,
):
    """Run the comparison on the N frames with the largest delta in object count.

    Useful to quickly find where the two runs differ most.
    """
    df_a = pd.read_csv(bboxes_csv_a)
    df_b = pd.read_csv(bboxes_csv_b)

    # Count objects per frame in each run
    counts_a = df_a.groupby(["scene", "pose_index"]).size().rename("n_a")
    counts_b = df_b.groupby(["scene", "pose_index"]).size().rename("n_b")
    merged = pd.concat([counts_a, counts_b], axis=1).fillna(0).astype(int)
    merged["abs_delta"] = (merged["n_b"] - merged["n_a"]).abs()
    merged = merged.sort_values("abs_delta", ascending=False)

    print(f"Top {n_frames} frames by object-count difference:\n")
    for i, ((scene, pose), row) in enumerate(merged.head(n_frames).iterrows()):
        d = row["n_b"] - row["n_a"]
        ds = f"+{d}" if d >= 0 else str(d)
        print(f"  {i+1}. {scene} pose {pose:3d}  —  "
              f"{label_a}: {row['n_a']}  vs  {label_b}: {row['n_b']}  ({ds})")
        compare_side_bboxes(
            scene, pose,
            bboxes_csv_a=bboxes_csv_a,
            bboxes_csv_b=bboxes_csv_b,
            label_a=label_a,
            label_b=label_b,
            processed_dir=processed_dir,
            **kwargs,
        )