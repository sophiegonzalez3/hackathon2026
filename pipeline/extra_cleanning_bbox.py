#!/usr/bin/env python3
"""
Bbox Cleanup & QA Report
=========================
Takes the GT bounding-box CSV from generate_gt_bboxes.py (e.g. gt_bboxes_run_03_merge.csv)
and produces a cleaned version ready for training, plus a detailed QA report.

Cleanup steps:
  1. Drop bboxes with too few points (noise fragments)
  2. Drop bboxes with suspiciously tiny volume (padding artifacts)
  3. Drop bboxes with extreme aspect ratios (likely miscluster)
  4. Drop bboxes with very large dimensions (runaway clusters)
  5. Report class balance & per-scene stats
  6. Save cleaned CSV

Usage:
    python cleanup_bboxes.py --input gt_bboxes_run_03_merge.csv
    python cleanup_bboxes.py --input gt_bboxes_run_03_merge.csv --min-points 10 --out gt_bboxes_clean.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Default thresholds (per-class where it matters)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MIN_POINTS = {
    "Antenna":       15,
    "Cable":         10,
    "Electric Pole": 15,
    "Wind Turbine":  15,
}

# Max single dimension (meters) — anything larger is a clustering failure
DEFAULT_MAX_DIM = {
    "Antenna":       180,
    "Cable":        150,   # cables can be long
    "Electric Pole": 80,
    "Wind Turbine": 170,
}

# Min volume (m³) — tiny padding-only boxes
DEFAULT_MIN_VOLUME = {
    "Antenna":       0.5,
    "Cable":         0.0,  # cables are thin, skip volume filter
    "Electric Pole": 0.5,
    "Wind Turbine":  1.0,
}

# Max aspect ratio (longest / shortest dim) — catches misshapen clusters
DEFAULT_MAX_ASPECT = {
    "Antenna":       30,
    "Cable":        500,   # cables are extremely elongated by nature
    "Electric Pole": 30,
    "Wind Turbine":  30,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup logic
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(df):
    """Add helper columns for filtering."""
    df = df.copy()
    dims = df[["bbox_width", "bbox_length", "bbox_height"]].values
    df["volume"] = dims[:, 0] * dims[:, 1] * dims[:, 2]
    df["max_dim"] = dims.max(axis=1)
    df["min_dim"] = dims.min(axis=1)
    # Avoid division by zero for min_dim
    safe_min = np.maximum(df["min_dim"].values, 0.01)
    df["aspect_ratio"] = df["max_dim"].values / safe_min
    return df


def cleanup(df, min_points=None, max_dim=None, min_volume=None, max_aspect=None,
            verbose=True):
    """Apply per-class filters and return (cleaned_df, drop_log)."""

    min_points = min_points or DEFAULT_MIN_POINTS
    max_dim = max_dim or DEFAULT_MAX_DIM
    min_volume = min_volume or DEFAULT_MIN_VOLUME
    max_aspect = max_aspect or DEFAULT_MAX_ASPECT

    df = compute_stats(df)
    n_orig = len(df)

    drop_reasons = []  # list of (index, reason) for reporting

    keep_mask = np.ones(len(df), dtype=bool)

    for label in df["class_label"].unique():
        cls_mask = df["class_label"] == label
        cls_idx = df.index[cls_mask]

        # 1. Min points
        mp = min_points.get(label, 10)
        low_pts = cls_mask & (df["num_points"] < mp)
        if low_pts.any():
            for idx in df.index[low_pts]:
                drop_reasons.append((idx, label, f"num_points={df.loc[idx, 'num_points']} < {mp}"))
            keep_mask &= ~low_pts

        # 2. Max dimension
        md = max_dim.get(label, 200)
        big_dim = cls_mask & (df["max_dim"] > md)
        if big_dim.any():
            for idx in df.index[big_dim]:
                drop_reasons.append((idx, label, f"max_dim={df.loc[idx, 'max_dim']:.1f}m > {md}m"))
            keep_mask &= ~big_dim

        # 3. Min volume
        mv = min_volume.get(label, 0.0)
        if mv > 0:
            tiny_vol = cls_mask & (df["volume"] < mv)
            if tiny_vol.any():
                for idx in df.index[tiny_vol]:
                    drop_reasons.append((idx, label, f"volume={df.loc[idx, 'volume']:.3f}m³ < {mv}m³"))
                keep_mask &= ~tiny_vol

        # 4. Max aspect ratio
        ma = max_aspect.get(label, 100)
        bad_aspect = cls_mask & (df["aspect_ratio"] > ma)
        if bad_aspect.any():
            for idx in df.index[bad_aspect]:
                drop_reasons.append((idx, label, f"aspect={df.loc[idx, 'aspect_ratio']:.1f} > {ma}"))
            keep_mask &= ~bad_aspect

    df_clean = df[keep_mask].copy()

    # Drop helper columns (not needed in training CSV)
    for col in ["volume", "max_dim", "min_dim", "aspect_ratio"]:
        if col in df_clean.columns:
            df_clean.drop(columns=col, inplace=True)

    n_dropped = n_orig - len(df_clean)

    # ── Frame-loss detection (always computed) ────────────────────────────
    # Find frames that HAD objects before cleanup but have NONE after
    frames_before = set(df.groupby(["scene", "pose_index"]).groups.keys())
    if len(df_clean) > 0:
        frames_after = set(df_clean.groupby(["scene", "pose_index"]).groups.keys())
    else:
        frames_after = set()
    lost_frames = sorted(frames_before - frames_after)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  CLEANUP SUMMARY")
        print(f"{'='*70}")
        print(f"  Input bboxes  : {n_orig:,}")
        print(f"  Dropped       : {n_dropped:,}  ({100*n_dropped/max(n_orig,1):.1f}%)")
        print(f"  Kept          : {len(df_clean):,}")

        # Breakdown by class
        print(f"\n  Drop breakdown by class:")
        drop_df = pd.DataFrame(drop_reasons, columns=["idx", "class", "reason"])
        if len(drop_df) > 0:
            for label in sorted(drop_df["class"].unique()):
                sub = drop_df[drop_df["class"] == label]
                orig_cls = (df["class_label"] == label).sum()
                print(f"    {label:17s}: dropped {len(sub):4d} / {orig_cls:4d}  "
                      f"({100*len(sub)/max(orig_cls,1):.1f}%)")
                # Sub-breakdown by reason type
                reason_types = sub["reason"].str.extract(r'^(\w+)=')[0].value_counts()
                for rt, cnt in reason_types.items():
                    print(f"      └─ {rt}: {cnt}")
        else:
            print("    (none dropped)")

        # ── Report frame losses ───────────────────────────────────────────
        if lost_frames:
            print(f"\n  ⚠ FRAME LOSS: {len(lost_frames)} frames lost ALL objects after cleanup!")
            print(f"    These frames had objects before but zero after filtering.")
            print(f"    They will become negative examples (no GT) during training.\n")
            for scene, pose in lost_frames[:15]:
                # Show what was dropped from this frame
                frame_drops = [
                    (cls, reason) for idx, cls, reason in drop_reasons
                    if df.loc[idx, "scene"] == scene and df.loc[idx, "pose_index"] == pose
                ]
                orig_count = len(df[(df["scene"] == scene) & (df["pose_index"] == pose)])
                reasons_summary = ", ".join(
                    sorted(set(r.split("=")[0] for _, r in frame_drops))
                )
                print(f"    {scene:15s} pose {pose:3d}  "
                      f"({orig_count} obj removed — {reasons_summary})")
            if len(lost_frames) > 15:
                print(f"    ... and {len(lost_frames) - 15} more")

            print(f"\n    Options:")
            print(f"      1. Accept: these become negative training examples (often fine)")
            print(f"      2. Lower thresholds: re-run with --min-points 8 to recover some")
            print(f"      3. Exclude: remove these frames from training entirely")
        else:
            print(f"\n  ✓ No frames lost all objects — every frame with GT still has GT.")

    return df_clean, drop_reasons, lost_frames


# ═══════════════════════════════════════════════════════════════════════════════
# QA Report
# ═══════════════════════════════════════════════════════════════════════════════

def qa_report(df, label=""):
    """Print a comprehensive QA report for a bbox dataframe."""

    tag = f" ({label})" if label else ""
    print(f"\n{'='*70}")
    print(f"  QA REPORT{tag}")
    print(f"{'='*70}")

    # ── Overview ──
    n_scenes = df["scene"].nunique()
    n_frames = df.groupby(["scene", "pose_index"]).ngroups
    print(f"\n  Total bboxes  : {len(df):,}")
    print(f"  Scenes        : {n_scenes}")
    print(f"  Frames w/ obj : {n_frames}")

    # ── Per-class stats ──
    print(f"\n  {'Class':17s} | {'Count':>6s} | {'Med pts':>7s} | "
          f"{'Med W×L×H (m)':>20s} | {'% of total':>10s}")
    print(f"  {'-'*17}-+-{'-'*6}-+-{'-'*7}-+-{'-'*20}-+-{'-'*10}")

    for label_cls in ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]:
        sub = df[df["class_label"] == label_cls]
        if len(sub) == 0:
            print(f"  {label_cls:17s} | {'0':>6s} |")
            continue
        med_w = sub["bbox_width"].median()
        med_l = sub["bbox_length"].median()
        med_h = sub["bbox_height"].median()
        med_pts = sub["num_points"].median()
        pct = 100 * len(sub) / len(df)
        print(f"  {label_cls:17s} | {len(sub):6d} | {med_pts:7.0f} | "
              f"{med_w:5.1f} × {med_l:5.1f} × {med_h:5.1f}     | {pct:9.1f}%")

    # ── Objects per frame distribution ──
    per_frame = df.groupby(["scene", "pose_index"]).size()
    print(f"\n  Objects per frame:")
    print(f"    mean: {per_frame.mean():.1f}  |  median: {per_frame.median():.0f}  |  "
          f"max: {per_frame.max()}  |  min: {per_frame.min()}")

    # ── Per-scene breakdown ──
    print(f"\n  Per-scene object counts:")
    scene_counts = df.groupby(["scene", "class_label"]).size().unstack(fill_value=0)
    scene_totals = scene_counts.sum(axis=1)
    for scene in sorted(df["scene"].unique()):
        if scene in scene_counts.index:
            row = scene_counts.loc[scene]
            parts = []
            for cls in ["Antenna", "Cable", "Electric Pole", "Wind Turbine"]:
                if cls in row.index and row[cls] > 0:
                    parts.append(f"{cls[:3]}={row[cls]}")
            total = scene_totals[scene]
            print(f"    {scene:15s}: {total:4d} total  ({', '.join(parts)})")

    # ── Potential issues ──
    print(f"\n  Potential issues:")
    issues = 0

    # Very low-point bboxes
    low = df[df["num_points"] < 20]
    if len(low) > 0:
        issues += 1
        print(f"    ⚠ {len(low)} bboxes with < 20 points")
        for cls in low["class_label"].unique():
            n = (low["class_label"] == cls).sum()
            print(f"      └─ {cls}: {n}")

    # Very large bboxes
    df_tmp = compute_stats(df)
    huge = df_tmp[df_tmp["max_dim"] > 150]
    if len(huge) > 0:
        issues += 1
        print(f"    ⚠ {len(huge)} bboxes with max dim > 150m")

    # Extreme class imbalance
    counts = df["class_label"].value_counts()
    if len(counts) > 1:
        ratio = counts.iloc[0] / max(counts.iloc[-1], 1)
        if ratio > 10:
            issues += 1
            top_cls = counts.index[0]
            bot_cls = counts.index[-1]
            print(f"    ⚠ Class imbalance: {top_cls} has {ratio:.0f}× more bboxes than {bot_cls}")
            print(f"       Consider: class-weighted loss or oversampling rare classes during training")

    # Frames with very many objects
    crowded = per_frame[per_frame > 30]
    if len(crowded) > 0:
        issues += 1
        print(f"    ⚠ {len(crowded)} frames with > 30 objects (may indicate over-fragmentation)")

    if issues == 0:
        print("    ✓ No issues detected")

    # ── Training readiness checklist ──
    print(f"\n  Training readiness:")
    checks = {
        "4 classes present": len(df["class_label"].unique()) == 4,
        "Multiple scenes": n_scenes >= 5,
        "Sufficient bboxes (>500)": len(df) > 500,
        "No NaN in coords": not df[["bbox_center_x", "bbox_center_y", "bbox_center_z"]].isna().any().any(),
        "No NaN in dims": not df[["bbox_width", "bbox_length", "bbox_height"]].isna().any().any(),
        "No negative dims": (df[["bbox_width", "bbox_length", "bbox_height"]] > 0).all().all(),
    }
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")

    print(f"\n{'='*70}")
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# Frame-level inspection (notebook-friendly)
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_frame(df, scene, pose_index, min_points=None, max_dim=None,
                  min_volume=None, max_aspect=None):
    """
    Show every object in a single frame with pass/fail for each filter.

    Usage in notebook:
        import cleanup_bboxes as cb
        df = pd.read_csv("gt_bboxes_run_03_merge.csv")
        cb.inspect_frame(df, "scene_3", 69)

        # Try different thresholds interactively:
        cb.inspect_frame(df, "scene_3", 69, min_points={"Cable": 5, ...})
    """
    min_points = min_points or DEFAULT_MIN_POINTS
    max_dim = max_dim or DEFAULT_MAX_DIM
    min_volume = min_volume or DEFAULT_MIN_VOLUME
    max_aspect = max_aspect or DEFAULT_MAX_ASPECT

    frame = df[(df["scene"] == scene) & (df["pose_index"] == pose_index)].copy()
    if len(frame) == 0:
        print(f"No objects found in {scene} pose {pose_index}")
        return None

    frame = compute_stats(frame)

    print(f"\n{'='*70}")
    print(f"  FRAME INSPECTION: {scene} pose {pose_index}")
    print(f"  {len(frame)} objects in raw data")
    print(f"{'='*70}\n")

    kept_count = 0
    dropped_count = 0
    rows = []

    for i, (idx, row) in enumerate(frame.iterrows()):
        label = row["class_label"]
        pts = row["num_points"]
        w, l, h = row["bbox_width"], row["bbox_length"], row["bbox_height"]
        vol = row["volume"]
        mx = row["max_dim"]
        ar = row["aspect_ratio"]

        # Evaluate each filter
        checks = {}
        checks["min_pts"] = (pts >= min_points.get(label, 10),
                             f"{pts} {'≥' if pts >= min_points.get(label, 10) else '<'} {min_points.get(label, 10)}")
        checks["max_dim"] = (mx <= max_dim.get(label, 200),
                             f"{mx:.1f}m {'≤' if mx <= max_dim.get(label, 200) else '>'} {max_dim.get(label, 200)}m")
        mv = min_volume.get(label, 0.0)
        if mv > 0:
            checks["min_vol"] = (vol >= mv,
                                 f"{vol:.2f}m³ {'≥' if vol >= mv else '<'} {mv}m³")
        else:
            checks["min_vol"] = (True, "skipped")
        ma = max_aspect.get(label, 100)
        checks["max_asp"] = (ar <= ma,
                             f"{ar:.1f} {'≤' if ar <= ma else '>'} {ma}")

        passed = all(ok for ok, _ in checks.values())
        status = "✓ KEEP" if passed else "✗ DROP"

        if passed:
            kept_count += 1
        else:
            dropped_count += 1

        # Print object details
        failed_filters = [name for name, (ok, _) in checks.items() if not ok]
        failed_str = f"  ← FAILED: {', '.join(failed_filters)}" if failed_filters else ""

        print(f"  Obj {i+1:2d} | {status} | {label:15s} | "
              f"{pts:4d} pts | {w:.1f}×{l:.1f}×{h:.1f}m{failed_str}")

        # Detailed filter breakdown
        for name, (ok, detail) in checks.items():
            mark = "✓" if ok else "✗"
            print(f"         {mark} {name:8s}: {detail}")
        print()

        rows.append({
            "obj_index": i + 1,
            "class_label": label,
            "num_points": pts,
            "bbox_width": w, "bbox_length": l, "bbox_height": h,
            "volume": vol, "max_dim": mx, "aspect_ratio": ar,
            "verdict": "KEEP" if passed else "DROP",
            "failed_filters": ", ".join(failed_filters) if failed_filters else "",
        })

    print(f"  {'─'*60}")
    print(f"  Result: {kept_count} kept, {dropped_count} dropped")
    if dropped_count == len(frame):
        print(f"  ⚠ ALL objects dropped — this frame becomes a negative example!")
    print()

    return pd.DataFrame(rows)


def inspect_worst(df, n=10, min_points=None, max_dim=None,
                  min_volume=None, max_aspect=None):
    """
    Find the N frames most affected by cleanup (highest drop rate).
    Useful for tuning thresholds — focus on the frames that hurt most.

    Usage:
        import cleanup_bboxes as cb
        df = pd.read_csv("gt_bboxes_run_03_merge.csv")
        cb.inspect_worst(df, n=10)
    """
    min_points = min_points or DEFAULT_MIN_POINTS
    max_dim = max_dim or DEFAULT_MAX_DIM
    min_volume = min_volume or DEFAULT_MIN_VOLUME
    max_aspect = max_aspect or DEFAULT_MAX_ASPECT

    df_stats = compute_stats(df)
    results = []

    for (scene, pose), group in df_stats.groupby(["scene", "pose_index"]):
        n_orig = len(group)
        n_kept = 0
        for _, row in group.iterrows():
            label = row["class_label"]
            ok = True
            ok &= row["num_points"] >= min_points.get(label, 10)
            ok &= row["max_dim"] <= max_dim.get(label, 200)
            mv = min_volume.get(label, 0.0)
            if mv > 0:
                ok &= row["volume"] >= mv
            ok &= row["aspect_ratio"] <= max_aspect.get(label, 100)
            if ok:
                n_kept += 1

        n_dropped = n_orig - n_kept
        results.append({
            "scene": scene,
            "pose_index": pose,
            "n_original": n_orig,
            "n_kept": n_kept,
            "n_dropped": n_dropped,
            "drop_pct": 100 * n_dropped / n_orig,
            "all_lost": n_kept == 0,
        })

    results_df = pd.DataFrame(results).sort_values(
        ["all_lost", "drop_pct"], ascending=[False, False]
    ).head(n)

    print(f"\n{'='*70}")
    print(f"  TOP {n} MOST AFFECTED FRAMES")
    print(f"{'='*70}\n")
    print(f"  {'Scene':15s} {'Pose':>5s} | {'Orig':>5s} {'Kept':>5s} {'Drop':>5s} {'Drop%':>6s} | {'Status'}")
    print(f"  {'─'*15} {'─'*5} + {'─'*5} {'─'*5} {'─'*5} {'─'*6} + {'─'*12}")

    for _, row in results_df.iterrows():
        status = "⚠ ALL LOST" if row["all_lost"] else ""
        print(f"  {row['scene']:15s} {row['pose_index']:5d} | "
              f"{row['n_original']:5d} {row['n_kept']:5d} {row['n_dropped']:5d} "
              f"{row['drop_pct']:5.1f}% | {status}")

    all_lost_count = results_df["all_lost"].sum()
    if all_lost_count > 0:
        print(f"\n  Tip: run inspect_frame() on the ALL LOST frames to see why.")
        print(f"  Example:")
        first_lost = results_df[results_df["all_lost"]].iloc[0]
        print(f'    cb.inspect_frame(df, "{first_lost["scene"]}", {first_lost["pose_index"]})')

    print()
    return results_df


def list_drops(df_raw, df_clean, class_label, reason, min_points=None,
               max_dim=None, min_volume=None, max_aspect=None):
    """
    List every frame where an object of `class_label` was dropped for `reason`.

    Parameters
    ----------
    df_raw   : DataFrame from the raw CSV  (gt_bboxes_run_03_merge.csv)
    df_clean : DataFrame from the cleaned CSV (gt_bboxes_run_03_merge_clean.csv)
    class_label : "Antenna", "Cable", "Electric Pole", or "Wind Turbine"
    reason      : "num_points", "max_dim", "min_vol", or "aspect"

    Usage:
        import cleanup_bboxes as cb
        import pandas as pd

        df_raw   = pd.read_csv("gt_bboxes_run_03_merge.csv")
        df_clean = pd.read_csv("gt_bboxes_run_03_merge_clean.csv")

        # All Antenna objects dropped because of max_dim
        cb.list_drops(df_raw, df_clean, "Antenna", "max_dim")

        # All Cable objects dropped because of num_points
        cb.list_drops(df_raw, df_clean, "Cable", "num_points")
    """
    min_points = min_points or DEFAULT_MIN_POINTS
    max_dim = max_dim or DEFAULT_MAX_DIM
    min_volume = min_volume or DEFAULT_MIN_VOLUME
    max_aspect = max_aspect or DEFAULT_MAX_ASPECT

    valid_reasons = {"num_points", "max_dim", "min_vol", "aspect"}
    if reason not in valid_reasons:
        print(f"Invalid reason '{reason}'. Choose from: {', '.join(sorted(valid_reasons))}")
        return None

    # Work on the raw class subset
    cls_raw = df_raw[df_raw["class_label"] == class_label].copy()
    if len(cls_raw) == 0:
        print(f"No '{class_label}' objects in raw data.")
        return None

    cls_raw = compute_stats(cls_raw)

    # Identify which objects fail THIS specific filter
    mp = min_points.get(class_label, 10)
    md = max_dim.get(class_label, 200)
    mv = min_volume.get(class_label, 0.0)
    ma = max_aspect.get(class_label, 100)

    if reason == "num_points":
        fail_mask = cls_raw["num_points"] < mp
        val_col, thresh_str = "num_points", f"< {mp}"
    elif reason == "max_dim":
        fail_mask = cls_raw["max_dim"] > md
        val_col, thresh_str = "max_dim", f"> {md}m"
    elif reason == "min_vol":
        if mv <= 0:
            print(f"min_vol filter is disabled for {class_label} (threshold = 0)")
            return None
        fail_mask = cls_raw["volume"] < mv
        val_col, thresh_str = "volume", f"< {mv}m³"
    elif reason == "aspect":
        fail_mask = cls_raw["aspect_ratio"] > ma
        val_col, thresh_str = "aspect_ratio", f"> {ma}"

    dropped = cls_raw[fail_mask].copy()

    if len(dropped) == 0:
        print(f"No '{class_label}' objects dropped for '{reason}'.")
        return None

    # For each dropped object, check if this frame still has objects of this class after cleanup
    cls_clean_frames = set()
    if len(df_clean) > 0:
        cls_clean_sub = df_clean[df_clean["class_label"] == class_label]
        cls_clean_frames = set(zip(cls_clean_sub["scene"], cls_clean_sub["pose_index"]))

    # Build results
    rows = []
    for _, row in dropped.iterrows():
        frame_key = (row["scene"], row["pose_index"])
        still_has_class = frame_key in cls_clean_frames
        # Count how many of this class were in the raw frame
        same_frame_cls = cls_raw[
            (cls_raw["scene"] == row["scene"]) &
            (cls_raw["pose_index"] == row["pose_index"])
        ]
        rows.append({
            "scene": row["scene"],
            "pose_index": int(row["pose_index"]),
            "num_points": int(row["num_points"]),
            "bbox_w": round(row["bbox_width"], 1),
            "bbox_l": round(row["bbox_length"], 1),
            "bbox_h": round(row["bbox_height"], 1),
            "filter_value": round(row[val_col], 2),
            "class_in_frame_raw": len(same_frame_cls),
            "class_still_present": still_has_class,
        })

    result_df = pd.DataFrame(rows).sort_values("filter_value",
        ascending=(reason in ("num_points", "min_vol")))

    # Print
    n_frames_affected = result_df.groupby(["scene", "pose_index"]).ngroups
    n_frames_class_lost = (~result_df.groupby(["scene", "pose_index"])
                           ["class_still_present"].any()).sum()

    print(f"\n{'='*70}")
    print(f"  DROPS: {class_label} × {reason} ({thresh_str})")
    print(f"{'='*70}")
    print(f"  {len(result_df)} objects dropped across {n_frames_affected} frames")
    if n_frames_class_lost > 0:
        print(f"  ⚠ {n_frames_class_lost} frames lost ALL {class_label} objects")
    print()

    # Compact table
    print(f"  {'Scene':15s} {'Pose':>5s} | {'Pts':>5s} | {'W×L×H (m)':>18s} | "
          f"{'Value':>8s} | {'Raw#':>4s} | {'Still?'}")
    print(f"  {'─'*15} {'─'*5}-+-{'─'*5}-+-{'─'*18}-+-{'─'*8}-+-{'─'*4}-+-{'─'*6}")

    for _, r in result_df.iterrows():
        still = "yes" if r["class_still_present"] else "⚠ NO"
        dims = f"{r['bbox_w']:5.1f}×{r['bbox_l']:5.1f}×{r['bbox_h']:5.1f}"
        print(f"  {r['scene']:15s} {r['pose_index']:5d} | {r['num_points']:5d} | "
              f"{dims:>18s} | {r['filter_value']:8.2f} | {r['class_in_frame_raw']:4d} | {still}")

    print(f"\n  Tip: inspect a specific frame with:")
    sample = result_df.iloc[0]
    print(f'    cb.inspect_frame(df, "{sample["scene"]}", {sample["pose_index"]})')
    print()

    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Clean GT bboxes for training")
    parser.add_argument("--input", required=True, help="Input CSV (e.g. gt_bboxes_run_03_merge.csv)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <input>_clean.csv)")
    parser.add_argument("--min-points", type=int, default=None,
                        help="Override min points for ALL classes (default: per-class)")
    parser.add_argument("--no-save", action="store_true", help="Just report, don't save")
    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} bboxes from {args.input}")

    # QA on raw data
    qa_report(df, label="BEFORE cleanup")

    # Override min_points if provided
    min_pts = None
    if args.min_points is not None:
        min_pts = {cls: args.min_points for cls in DEFAULT_MIN_POINTS}

    # Cleanup
    df_clean, drops, lost_frames = cleanup(df, min_points=min_pts)

    # QA on cleaned data
    checks = qa_report(df_clean, label="AFTER cleanup")

    # Save
    if not args.no_save:
        if args.out:
            out_path = args.out
        else:
            stem = Path(args.input).stem
            out_path = str(Path(args.input).parent / f"{stem}_clean.csv")

        # Keep only the standard columns for training
        train_cols = [
            "scene", "pose_index", "frame_file",
            "ego_x", "ego_y", "ego_z", "ego_yaw",
            "bbox_center_x", "bbox_center_y", "bbox_center_z",
            "bbox_width", "bbox_length", "bbox_height", "bbox_yaw",
            "class_ID", "class_label", "num_points",
        ]
        df_clean[train_cols].to_csv(out_path, index=False)
        print(f"\n  Saved cleaned CSV → {out_path}")
        print(f"  ({len(df_clean):,} bboxes, ready for training)")

        # Also save the lost-frames list for reference
        if lost_frames:
            lost_path = str(Path(out_path).with_suffix(".lost_frames.csv"))
            lost_df = pd.DataFrame(lost_frames, columns=["scene", "pose_index"])
            lost_df.to_csv(lost_path, index=False)
            print(f"  Lost-frames list → {lost_path}")


if __name__ == "__main__":
    main()