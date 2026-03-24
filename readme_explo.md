# LiDAR Notebook — Quick Start Guide

> **Audience:** Junior team members new to the project.
> **Prerequisite:** Python 3.10+, Jupyter Notebook or JupyterLab.

---

## 1. Setup

Make sure you have the dependencies installed:

```bash
pip install numpy pandas h5py plotly
pip install kaleido   # only needed if you want to save plots as PNG/SVG/PDF
```

Place `lidar_notebook.py` and `lidar_utils.py` in the same folder as your notebook (or in your Python path).

---

## 2. Basic Workflow

Every analysis follows three steps: **load → pick a frame → plot**.

```python
import lidar_notebook as ln

# Step 1 — Load an HDF5 file
df, poses = ln.load("trainingData/scene_1.h5")
# This prints the total number of points and lists all available frames.

# Step 2 — Pick a frame by its pose index
frame, xyz = ln.get_frame(df, poses, pose_index=3)
# frame = DataFrame with all point fields + class labels
# xyz   = NumPy array (N, 3) of X, Y, Z coordinates in meters

# Step 3 — Plot!
ln.plot_bev(xyz, frame)
```

### Quick stats

```python
ln.summary(frame, xyz)
```

Prints spatial bounds (min/max X, Y, Z in meters) and the number of points per class. Use this to sanity-check a frame before plotting.

---

## 3. Available Plots

### `plot_bev` — Bird's Eye View (top-down, X-Y)

```python
ln.plot_bev(xyz, frame)
```

**What it shows:** The scene from above, like a map. Each point is projected onto the horizontal plane.

**Why it matters:** You see where obstacles are relative to the sensor: how far, in which direction, and how they relate to each other (e.g. cables running between poles). This is the view you'll use most when designing bounding boxes, since `bbox_center_x` and `bbox_center_y` live in this plane.

---

### `plot_side` — Side / Elevation View (X-Z)

```python
ln.plot_side(xyz, frame)
```

**What it shows:** The scene from the side — horizontal distance (X) vs height (Z).

**Why it matters:** Reveals how tall things are. Wind turbines and antennas are tall and narrow, poles are medium, cables hang at specific heights between poles. Essential for setting `bbox_height` and `bbox_center_z`.

---

### `plot_3d` — Full 3D Scatter

```python
ln.plot_3d(xyz, frame)
```

**What it shows:** All points in 3D space. You can rotate, zoom, and hover with your mouse.

**Why it matters:** The complete picture. Cables sag and curve in 3D in ways you can't see in 2D projections. Use this to build intuition about the true geometry of each obstacle class.

> **Tip:** This view caps at 100k points by default to keep the browser responsive. Pass `max_points=200_000` if you need more.

---

### `plot_obstacles_only` — 3D Without Background

```python
ln.plot_obstacles_only(xyz, frame)
```

**What it shows:** Same as `plot_3d` but with all background/terrain points removed.

**Why it matters:** Background points massively outnumber obstacles (often 99%+), making them hard to see. This view isolates the obstacle structure — very useful when inspecting clustering results or checking if your bounding boxes make sense.

---

### `plot_bev_reflectivity` — BEV Colored by Reflectivity

```python
ln.plot_bev_reflectivity(xyz, frame)
```

**What it shows:** Top-down view colored by laser return intensity instead of class.

**Why it matters:** On the **evaluation set**, you won't have RGB class labels (they're all set to 128). Reflectivity + geometry is all your model gets as input. This view simulates what inference looks like — check whether you can visually distinguish obstacles from terrain using reflectivity alone.

---

### `plot_3d_reflectivity` — 3D Colored by Reflectivity

```python
ln.plot_3d_reflectivity(xyz, frame)
```

Same idea as above but in full 3D. Useful for spotting reflectivity patterns on vertical structures (poles, turbine towers).

---

### `plot_class_dist` — Class Distribution Bar Chart

```python
ln.plot_class_dist(frame)
```

**What it shows:** Horizontal bar chart counting how many points belong to each class.

**Why it matters:** Shows class imbalance at a glance. Background dominates, cables are the rarest. This tells you whether you need class weighting, focal loss, or oversampling during training.

---

## 4. Saving Plots

Every plot function accepts `save=` and `show=` parameters:

```python
# Display only (default)
ln.plot_bev(xyz, frame)

# Save only (no display)
ln.plot_bev(xyz, frame, save="plots/bev.png")

# Save AND display
ln.plot_bev(xyz, frame, save="plots/bev.png", show=True)

# Save as interactive HTML (no kaleido needed, keeps zoom/hover)
ln.plot_3d(xyz, frame, save="plots/3d_view.html")
```

### Batch save all plots at once

```python
ln.save_all(xyz, frame, out_dir="plots", prefix="pose3")             # PNG
ln.save_all(xyz, frame, out_dir="plots", prefix="pose3", fmt="html") # interactive HTML
ln.save_all(xyz, frame, out_dir="plots", prefix="pose3", fmt="svg")  # vector
```

This creates 7 files in the output directory covering every view.

---

## 5. Class Reference

These are the 4 obstacle classes you need to detect:

| Class ID | Label            | RGB (ground truth) | Color on plots |
|:--------:|:-----------------|:------------------:|:--------------:|
| 0        | Antenna          | (38, 23, 180)      | Blue           |
| 1        | Cable            | (177, 132, 47)     | Gold           |
| 2        | Electric Pole    | (129, 81, 97)      | Mauve          |
| 3        | Wind Turbine     | (66, 132, 9)       | Green          |
| —        | Background       | anything else       | Grey           |

---

## 6. Common Patterns

### Loop through several frames

```python
df, poses = ln.load("data/scene_01.h5")

for i in range(min(5, len(poses))):
    frame, xyz = ln.get_frame(df, poses, pose_index=i)
    ln.save_all(xyz, frame, out_dir="plots", prefix=f"pose{i}")
```

### Check a frame quickly without plotting

```python
frame, xyz = ln.get_frame(df, poses, pose_index=7)
ln.summary(frame, xyz)
```

### Only look at obstacles

```python
obs_mask = frame["class_id"] >= 0
obs_xyz = xyz[obs_mask]
obs_frame = frame[obs_mask].reset_index(drop=True)
print(f"This frame has {len(obs_frame)} obstacle points")
```

---

## 7. Troubleshooting

| Problem | Fix |
|:--------|:----|
| `ModuleNotFoundError: plotly` | `pip install plotly` |
| Saving PNG fails | `pip install kaleido` |
| Plot is slow / browser lags | Lower `max_points` (e.g. `max_points=50_000`) |
| No obstacles visible | Check `ln.summary()` — the frame might have zero obstacle points |
| `display()` not found | You're not in Jupyter — use `print(poses)` instead |