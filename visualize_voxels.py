"""
visualize_voxels.py — Visualize consolidated voxel scene maps
==============================================================
Produces:
  1. 2D projection plots (top-down BEV + side views) as PNG
  2. 3D interactive scatter plot as HTML (opens in browser)

Usage:
    # Visualize one consolidated scene (training, with labels)
    python visualize_voxels.py --voxel-csv consolidated/scene_1_voxels.csv --with-labels

    # Visualize without labels (inference result)
    python visualize_voxels.py --voxel-csv consolidated/scene_1_voxels.csv

    # Directly consolidate and visualize a scene
    python visualize_voxels.py --npz-dir processed/scene_1 --voxel-size 0.5 --with-labels
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']
CLASS_COLORS = {
    -1: '#AAAAAA',  # Background (grey)
    0: '#2617B4',   # Antenna (blue)
    1: '#B18430',   # Cable (brown/gold)
    2: '#815161',   # Electric Pole (mauve)
    3: '#428409',   # Wind Turbine (green)
}
CLASS_COLORS_RGB = {
    -1: (0.67, 0.67, 0.67),
    0: (0.15, 0.09, 0.71),
    1: (0.69, 0.52, 0.19),
    2: (0.51, 0.32, 0.38),
    3: (0.26, 0.52, 0.04),
}


def load_or_consolidate(args):
    """Either load a pre-computed voxel CSV or consolidate from .npz."""
    if args.voxel_csv:
        print(f"Loading voxel data from {args.voxel_csv}")
        df = pd.read_csv(args.voxel_csv)
        scene_name = Path(args.voxel_csv).stem.replace('_voxels', '')
        return df, scene_name

    elif args.npz_dir:
        from consolidate_scene import consolidate_scene
        print(f"Consolidating scene from {args.npz_dir}")
        grid = consolidate_scene(
            args.npz_dir,
            voxel_size=args.voxel_size,
            with_labels=args.with_labels,
            verbose=True,
        )
        df = grid.to_dataframe(with_labels=args.with_labels)
        scene_name = Path(args.npz_dir).name
        return df, scene_name

    else:
        raise ValueError("Provide either --voxel-csv or --npz-dir")


def plot_projections(df, scene_name, with_labels, output_dir):
    """
    Create 2D projection plots: BEV (top-down), XZ (front), YZ (side).
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(f'Scene Consolidation: {scene_name}\n'
                 f'{len(df):,} occupied voxels',
                 fontsize=16, fontweight='bold')

    # Determine coloring
    if with_labels and 'class_id' in df.columns:
        # Color by class
        obstacle = df[df.get('is_obstacle', df['class_id'] >= 0) == True]
        background = df[df.get('is_obstacle', df['class_id'] >= 0) == False]
        color_mode = 'class'
    else:
        obstacle = df
        background = pd.DataFrame()
        color_mode = 'hits'

    # ── Plot 1: BEV (top-down, X-Y) ──
    ax = axes[0, 0]
    if color_mode == 'class':
        # Background first (grey, small)
        if len(background) > 0:
            bg_sample = background.sample(min(len(background), 50000), random_state=42)
            ax.scatter(bg_sample['world_x'], bg_sample['world_y'],
                       c='#DDDDDD', s=0.5, alpha=0.3, rasterized=True)
        # Obstacles on top
        for cid, name in enumerate(CLASS_NAMES):
            cls_df = obstacle[obstacle['class_id'] == cid]
            if len(cls_df) > 0:
                ax.scatter(cls_df['world_x'], cls_df['world_y'],
                           c=CLASS_COLORS[cid], s=3, alpha=0.7, label=name)
        ax.legend(fontsize=9, markerscale=5)
    else:
        sc = ax.scatter(df['world_x'], df['world_y'],
                        c=np.log1p(df['hit_count']), cmap='viridis',
                        s=1, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label='log(hit_count)')
    ax.set_xlabel('World X (m)')
    ax.set_ylabel('World Y (m)')
    ax.set_title('Bird\'s Eye View (X-Y)')
    ax.set_aspect('equal')

    # ── Plot 2: Side view (X-Z) ──
    ax = axes[0, 1]
    if color_mode == 'class':
        if len(background) > 0:
            bg_sample = background.sample(min(len(background), 50000), random_state=42)
            ax.scatter(bg_sample['world_x'], bg_sample['world_z'],
                       c='#DDDDDD', s=0.5, alpha=0.3, rasterized=True)
        for cid, name in enumerate(CLASS_NAMES):
            cls_df = obstacle[obstacle['class_id'] == cid]
            if len(cls_df) > 0:
                ax.scatter(cls_df['world_x'], cls_df['world_z'],
                           c=CLASS_COLORS[cid], s=3, alpha=0.7, label=name)
        ax.legend(fontsize=9, markerscale=5)
    else:
        sc = ax.scatter(df['world_x'], df['world_z'],
                        c=np.log1p(df['hit_count']), cmap='viridis',
                        s=1, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label='log(hit_count)')
    ax.set_xlabel('World X (m)')
    ax.set_ylabel('World Z (m)')
    ax.set_title('Side View (X-Z)')

    # ── Plot 3: Side view (Y-Z) ──
    ax = axes[1, 0]
    if color_mode == 'class':
        if len(background) > 0:
            bg_sample = background.sample(min(len(background), 50000), random_state=42)
            ax.scatter(bg_sample['world_y'], bg_sample['world_z'],
                       c='#DDDDDD', s=0.5, alpha=0.3, rasterized=True)
        for cid, name in enumerate(CLASS_NAMES):
            cls_df = obstacle[obstacle['class_id'] == cid]
            if len(cls_df) > 0:
                ax.scatter(cls_df['world_y'], cls_df['world_z'],
                           c=CLASS_COLORS[cid], s=3, alpha=0.7, label=name)
        ax.legend(fontsize=9, markerscale=5)
    else:
        sc = ax.scatter(df['world_y'], df['world_z'],
                        c=np.log1p(df['hit_count']), cmap='viridis',
                        s=1, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label='log(hit_count)')
    ax.set_xlabel('World Y (m)')
    ax.set_ylabel('World Z (m)')
    ax.set_title('Side View (Y-Z)')

    # ── Plot 4: Statistics ──
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"Scene: {scene_name}\n"
    stats_text += f"Total voxels: {len(df):,}\n"
    stats_text += f"Voxel size: {df['world_x'].diff().dropna().abs().median():.2f}m (approx)\n\n"

    stats_text += f"World extent:\n"
    stats_text += f"  X: [{df['world_x'].min():.0f}, {df['world_x'].max():.0f}]m\n"
    stats_text += f"  Y: [{df['world_y'].min():.0f}, {df['world_y'].max():.0f}]m\n"
    stats_text += f"  Z: [{df['world_z'].min():.0f}, {df['world_z'].max():.0f}]m\n\n"

    stats_text += f"Hit count: median={df['hit_count'].median():.0f}, "
    stats_text += f"max={df['hit_count'].max():.0f}\n"
    stats_text += f"Frame count: median={df['frame_count'].median():.0f}, "
    stats_text += f"max={df['frame_count'].max():.0f}\n\n"

    if with_labels and 'is_obstacle' in df.columns:
        n_obs = df['is_obstacle'].sum()
        n_bg = (~df['is_obstacle']).sum()
        stats_text += f"Obstacle voxels: {n_obs:,} ({100*n_obs/len(df):.1f}%)\n"
        stats_text += f"Background voxels: {n_bg:,} ({100*n_bg/len(df):.1f}%)\n\n"

        obs_df = df[df['is_obstacle']]
        for cid, name in enumerate(CLASS_NAMES):
            count = (obs_df['class_id'] == cid).sum()
            stats_text += f"  {name}: {count:,} voxels\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(output_dir) / f'{scene_name}_voxel_map.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved projection plots: {out_path}")
    return out_path


def plot_feature_distributions(df, scene_name, output_dir):
    """
    Plot feature distributions for obstacle vs background.
    Helps verify that the features are actually discriminative.
    """
    if 'is_obstacle' not in df.columns:
        print("  No labels — skipping feature distribution plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Feature Distributions: Obstacle vs Background — {scene_name}',
                 fontsize=14, fontweight='bold')

    features = [
        ('hit_count', 'Hit Count (log scale)', True),
        ('frame_count', 'Frame Count', False),
        ('ref_mean', 'Mean Reflectivity', False),
        ('z_mean', 'Mean Z Height (m)', False),
        ('z_range', 'Z Range within Voxel (m)', False),
        ('observation_ratio', 'Observation Ratio', False),
    ]

    for idx, (feat, title, use_log) in enumerate(features):
        ax = axes.flat[idx]
        bg = df[~df['is_obstacle']][feat].dropna()
        obs = df[df['is_obstacle']][feat].dropna()

        if use_log:
            bg = np.log1p(bg)
            obs = np.log1p(obs)

        ax.hist(bg, bins=50, alpha=0.5, density=True, color='grey', label='Background')
        ax.hist(obs, bins=50, alpha=0.5, density=True, color='red', label='Obstacle')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(output_dir) / f'{scene_name}_feature_distributions.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved feature distributions: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize consolidated voxel scene')
    parser.add_argument('--voxel-csv', type=str, default=None,
                        help='Pre-computed voxel CSV file')
    parser.add_argument('--npz-dir', type=str, default=None,
                        help='Scene directory with frame_*.npz (consolidates on the fly)')
    parser.add_argument('--voxel-size', type=float, default=0.5)
    parser.add_argument('--with-labels', action='store_true',
                        help='Use RGB labels (training data only)')
    parser.add_argument('--output-dir', type=str, default='plots/',
                        help='Output directory for plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, scene_name = load_or_consolidate(args)

    print(f"\n  Voxel DataFrame: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    plot_projections(df, scene_name, args.with_labels, output_dir)
    plot_feature_distributions(df, scene_name, output_dir)


if __name__ == '__main__':
    main()
