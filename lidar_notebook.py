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
