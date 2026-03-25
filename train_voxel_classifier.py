"""
train_voxel_classifier.py — Train voxel-level classifiers
==========================================================
Two-stage classification on consolidated voxel features:

  Stage 1: Obstacle vs Background (binary)
  Stage 2: Obstacle class (Antenna / Cable / Pole / Turbine)

Uses the consolidated voxel .parquet files from consolidate_scene.py.

Usage:
    python train_voxel_classifier.py \
        --voxel-dir consolidated/ \
        --output-dir models_v2/
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']
RANDOM_STATE = 42

# Features computed from the voxel grid
VOXEL_FEATURES = [
    'hit_count',
    'frame_count',
    'observation_ratio',
    'ref_mean',
    'ref_std',
    'z_min',
    'z_max',
    'z_range',
    'z_mean',
    'z_std',
    'elev_mean',
]


# ─────────────────────────────────────────────────────────────
# NEIGHBORHOOD FEATURES (computed from surrounding voxels)
# ─────────────────────────────────────────────────────────────
def compute_neighborhood_features(df, radius=2):
    """
    For each voxel, compute features from its local neighborhood.
    This captures context — a voxel that's part of a tall column (pole)
    looks different from one that's part of a flat sheet (ground).

    Args:
        df: DataFrame with voxel data (must have vx, vy, vz columns)
        radius: neighborhood radius in voxel units

    Returns:
        DataFrame with additional neighborhood columns
    """
    print(f"  Computing neighborhood features (radius={radius})...")
    t0 = time.time()

    # Build spatial index as a dict for fast lookup
    voxel_lookup = {}
    for idx, row in df.iterrows():
        key = (int(row['vx']), int(row['vy']), int(row['vz']))
        voxel_lookup[key] = idx

    n = len(df)
    neigh_count = np.zeros(n)
    neigh_z_span = np.zeros(n)
    neigh_hit_sum = np.zeros(n)
    neigh_ref_mean = np.zeros(n)
    neigh_vertical_extent = np.zeros(n)

    # For each voxel, scan its column (same vx, vy, varying vz)
    # This is much faster than a full 3D neighborhood scan
    column_lookup = {}
    for idx, row in df.iterrows():
        col_key = (int(row['vx']), int(row['vy']))
        if col_key not in column_lookup:
            column_lookup[col_key] = []
        column_lookup[col_key].append((int(row['vz']), idx))

    for col_key, voxels_in_col in column_lookup.items():
        z_vals = [v[0] for v in voxels_in_col]
        z_span = max(z_vals) - min(z_vals) + 1 if z_vals else 0
        col_hit_sum = sum(df.loc[v[1], 'hit_count'] for v in voxels_in_col)
        col_ref_mean = np.mean([df.loc[v[1], 'ref_mean'] for v in voxels_in_col])

        for vz, idx in voxels_in_col:
            neigh_count[idx] = len(voxels_in_col)
            neigh_z_span[idx] = z_span
            neigh_hit_sum[idx] = col_hit_sum
            neigh_ref_mean[idx] = col_ref_mean
            neigh_vertical_extent[idx] = z_span * df.loc[idx, 'world_z']  # crude height proxy

    df = df.copy()
    df['neigh_column_count'] = neigh_count
    df['neigh_column_z_span'] = neigh_z_span
    df['neigh_column_hit_sum'] = neigh_hit_sum
    df['neigh_column_ref_mean'] = neigh_ref_mean

    # Height above minimum z in column (ground-relative height proxy)
    col_z_mins = {}
    for col_key, voxels_in_col in column_lookup.items():
        col_z_mins[col_key] = min(v[0] for v in voxels_in_col)

    height_above_col_min = np.zeros(n)
    for idx, row in df.iterrows():
        col_key = (int(row['vx']), int(row['vy']))
        voxel_size = row.get('world_z', 0) / (row['vz'] + 1e-10) if row['vz'] != 0 else 0.5
        height_above_col_min[idx] = (row['vz'] - col_z_mins[col_key])

    df['height_above_column_base'] = height_above_col_min

    elapsed = time.time() - t0
    print(f"    Done [{elapsed:.1f}s]")

    return df


# ─────────────────────────────────────────────────────────────
# FEATURE LIST
# ─────────────────────────────────────────────────────────────
ALL_FEATURES = VOXEL_FEATURES + [
    'neigh_column_count',
    'neigh_column_z_span',
    'neigh_column_hit_sum',
    'neigh_column_ref_mean',
    'height_above_column_base',
]


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_voxel_data(voxel_dir):
    """Load all scene voxel parquets and concatenate."""
    voxel_dir = Path(voxel_dir)
    parquet_files = sorted(voxel_dir.glob('*_voxels.csv'))

    if not parquet_files:
        raise FileNotFoundError(f"No *_voxels.csv files in {voxel_dir}")

    dfs = []
    for f in parquet_files:
        df = pd.read_csv(f)
        print(f"  Loaded {f.name}: {len(df):,} voxels")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total: {len(combined):,} voxels from {len(parquet_files)} scenes")
    return combined


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
def train_stage1(X, y_binary, feature_names):
    """Train obstacle vs background classifier."""
    print("\n" + "=" * 70)
    print("STAGE 1: Obstacle vs Background")
    print("=" * 70)

    # Balance: subsample background to at most 5x obstacle count
    obstacle_mask = y_binary == 1
    n_obs = obstacle_mask.sum()
    n_bg = (~obstacle_mask).sum()
    print(f"  Obstacle voxels: {n_obs:,}")
    print(f"  Background voxels: {n_bg:,}")

    max_bg = min(n_bg, n_obs * 5)
    bg_indices = np.where(~obstacle_mask)[0]
    np.random.seed(RANDOM_STATE)
    bg_sample = np.random.choice(bg_indices, size=max_bg, replace=False)
    obs_indices = np.where(obstacle_mask)[0]
    train_indices = np.concatenate([obs_indices, bg_sample])
    np.random.shuffle(train_indices)

    X_bal = X[train_indices]
    y_bal = y_binary[train_indices]
    print(f"  Balanced training set: {len(y_bal):,} "
          f"({(y_bal == 1).sum():,} obstacle, {(y_bal == 0).sum():,} background)")

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(clf, X_bal, y_bal, cv=cv)

    f1 = f1_score(y_bal, y_pred, average='binary')
    print(f"\n  5-Fold CV Results:")
    print(classification_report(y_bal, y_pred,
                                target_names=['Background', 'Obstacle'], digits=3))

    # Train on full balanced set
    clf.fit(X_bal, y_bal)
    return clf


def train_stage2(X_obstacle, y_class, feature_names):
    """Train 4-class obstacle classifier (on obstacle voxels only)."""
    print("\n" + "=" * 70)
    print("STAGE 2: Obstacle Classification (4 classes)")
    print("=" * 70)

    print(f"  Training on {len(y_class):,} obstacle voxels")
    class_counts = pd.Series(y_class).value_counts().sort_index()
    for cid, count in class_counts.items():
        print(f"    {CLASS_NAMES[cid]}: {count:,}")

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(clf, X_obstacle, y_class, cv=cv)

    f1_w = f1_score(y_class, y_pred, average='weighted')
    f1_m = f1_score(y_class, y_pred, average='macro')
    print(f"\n  5-Fold CV Results:")
    print(f"    F1 weighted: {f1_w:.4f}")
    print(f"    F1 macro:    {f1_m:.4f}")
    print(classification_report(y_class, y_pred,
                                target_names=CLASS_NAMES, digits=3))

    # Train on full set
    clf.fit(X_obstacle, y_class)
    return clf, y_pred


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────
def plot_report(y_class, y_pred, clf_stage2, feature_names, output_dir):
    """Generate training evaluation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Voxel Classifier — Training Report', fontsize=14, fontweight='bold')

    # 1. Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_class, y_pred, labels=list(range(4)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (% per row)')
    ax.tick_params(axis='x', rotation=45)

    # 2. Feature importance
    ax = axes[1]
    if hasattr(clf_stage2, 'feature_importances_'):
        imp = clf_stage2.feature_importances_
        sorted_idx = np.argsort(imp)[::-1][:15]
        ax.barh(range(len(sorted_idx)), imp[sorted_idx], color='steelblue')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Features (Stage 2)')
        ax.invert_yaxis()

    # 3. Per-class F1
    ax = axes[2]
    report = classification_report(y_class, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, digits=3)
    colors = ['#2617B4', '#B18430', '#815161', '#428409']
    f1s = [report[c]['f1-score'] for c in CLASS_NAMES]
    ax.bar(CLASS_NAMES, f1s, color=colors)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 (CV)')
    ax.set_ylim(0.5, 1.05)
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plot_path = Path(output_dir) / 'voxel_training_report.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Report saved: {plot_path}")


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
def save_pipeline(clf_stage1, clf_stage2, scaler, imputer, config, output_dir):
    """Save both classifiers and preprocessing objects."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in [('stage1_model.pkl', clf_stage1),
                       ('stage2_model.pkl', clf_stage2),
                       ('scaler.pkl', scaler),
                       ('imputer.pkl', imputer)]:
        with open(output_dir / name, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved: {name}")

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: config.json")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train voxel classifier')
    parser.add_argument('--voxel-dir', type=str, required=True,
                        help='Directory with consolidated *_voxels.parquet files')
    parser.add_argument('--output-dir', type=str, default='models_v2/',
                        help='Output directory for trained models')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("=" * 70)
    print("LOADING VOXEL DATA")
    print("=" * 70)
    df = load_voxel_data(args.voxel_dir)

    # ── Compute neighborhood features ──
    print("\n" + "=" * 70)
    print("COMPUTING NEIGHBORHOOD FEATURES")
    print("=" * 70)
    df = compute_neighborhood_features(df)

    # ── Prepare features ──
    feature_names = ALL_FEATURES
    X_raw = df[feature_names].values

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_imputed)

    print(f"\n  Feature matrix: {X.shape}")

    # ── Stage 1: Obstacle vs Background ──
    y_binary = df['is_obstacle'].astype(int).values
    clf_stage1 = train_stage1(X, y_binary, feature_names)

    # ── Stage 2: Obstacle classification ──
    obstacle_mask = df['is_obstacle'].values
    X_obstacle = X[obstacle_mask]
    y_class = df.loc[obstacle_mask, 'class_id'].astype(int).values

    clf_stage2, y_pred_class = train_stage2(X_obstacle, y_class, feature_names)

    # ── Save ──
    print("\n" + "=" * 70)
    print("SAVING PIPELINE")
    print("=" * 70)
    config = {
        'feature_names': feature_names,
        'class_names': CLASS_NAMES,
        'n_voxels_total': len(df),
        'n_voxels_obstacle': int(obstacle_mask.sum()),
    }
    save_pipeline(clf_stage1, clf_stage2, scaler, imputer, config, output_dir)

    # ── Plot ──
    plot_report(y_class, y_pred_class, clf_stage2, feature_names, output_dir)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
