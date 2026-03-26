"""
PointPillars + CenterHead — Inference Script
==============================================
Loads a trained checkpoint, runs detection on new data, and outputs
the competition CSV with the exact required columns.

Usage:
    # On preprocessed evaluation data
    python -m pointpillars.inference \
        --checkpoint checkpoints/best.pth \
        --processed-dir processed_eval/ \
        --out predictions.csv

    # Also works on training data (for sanity checks)
    python -m pointpillars.inference \
        --checkpoint checkpoints/best.pth \
        --processed-dir processed/ \
        --out predictions_train.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .model import PointPillarsCenterHead
from .dataset import AirbusLidarDataset, collate_fn
from .utils import decode_predictions


@torch.no_grad()
def run_inference(model, loader, cfg, device):
    """Run detection on all frames and return a list of result dicts."""
    model.eval()
    all_results = []

    for batch_idx, batch in enumerate(loader):
        # Move to device
        pf = batch['pillar_features'].to(device)
        np_ = batch['num_points'].to(device)
        pc = batch['pillar_coords'].to(device)
        B = len(batch['scenes'])

        preds = model(pf, np_, pc, B)
        results = decode_predictions(preds, cfg)

        for i, res in enumerate(results):
            scene = batch['scenes'][i]
            pose_index = batch['pose_indices'][i]
            ego = batch['ego_poses'][i]

            boxes = res['boxes'].cpu().numpy()   # (K, 7)
            scores = res['scores'].cpu().numpy()
            labels = res['labels'].cpu().numpy()

            for j in range(len(boxes)):
                cx, cy, cz, w, l, h, yaw = boxes[j]
                cid = int(labels[j])
                cls_name = cfg.class_names[cid]

                all_results.append({
                    'ego_x': ego[0],
                    'ego_y': ego[1],
                    'ego_z': ego[2],
                    'ego_yaw': ego[3],
                    'bbox_center_x': float(cx),
                    'bbox_center_y': float(cy),
                    'bbox_center_z': float(cz),
                    'bbox_width': float(w),
                    'bbox_length': float(l),
                    'bbox_height': float(h),
                    'bbox_yaw': float(yaw),
                    'class_ID': cid,
                    'class_label': cls_name,
                    # Extra columns (not required but useful)
                    'confidence': float(scores[j]),
                    'scene': scene,
                    'pose_index': pose_index,
                })

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
            n_det = sum(len(r['boxes']) for r in results)
            print(f"  Batch {batch_idx+1}/{len(loader)} — {n_det} detections in batch")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run PointPillars inference")
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--processed-dir', required=True, help='Preprocessed data directory')
    parser.add_argument('--out', default='predictions.csv', help='Output CSV path')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override score threshold')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Reconstruct config
    if 'config' in ckpt:
        cfg = ckpt['config']
    else:
        cfg = Config()

    cfg.processed_dir = args.processed_dir
    cfg.batch_size = args.batch
    cfg.num_workers = args.workers
    if args.threshold is not None:
        cfg.score_threshold = args.threshold

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.device = str(device)

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"Val loss at checkpoint: {ckpt.get('val_loss', '?')}")
    cfg.print_summary()

    # ── Build model ───────────────────────────────────────────────────────
    model = PointPillarsCenterHead(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    total, trainable = model.count_parameters()
    print(f"Model: {trainable:,} params")

    # ── Build dataset (no GT) ─────────────────────────────────────────────
    dataset = AirbusLidarDataset(cfg, split='all', gt_csv=None,
                                 processed_dir=args.processed_dir)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Run inference ─────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(dataset)} frames...")
    results = run_inference(model, loader, cfg, device)

    # ── Save CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    # Competition format columns (drop extras for the submission CSV)
    competition_cols = [
        'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
        'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
        'bbox_width', 'bbox_length', 'bbox_height', 'bbox_yaw',
        'class_ID', 'class_label',
    ]

    # Save full version (with confidence + metadata)
    df.to_csv(args.out, index=False)
    print(f"\nFull CSV saved → {args.out}")
    print(f"  {len(df)} detections across {df['pose_index'].nunique()} frames")

    # Also save competition-format version
    comp_path = args.out.replace('.csv', '_submission.csv')
    df[competition_cols].to_csv(comp_path, index=False)
    print(f"  Competition CSV → {comp_path}")

    # ── Quick summary ─────────────────────────────────────────────────────
    if len(df) > 0:
        print(f"\nClass breakdown:")
        print(df['class_label'].value_counts().to_string())
        print(f"\nMean confidence: {df['confidence'].mean():.3f}")
        det_per_frame = df.groupby(['scene', 'pose_index']).size()
        print(f"Detections/frame: mean={det_per_frame.mean():.1f}  "
              f"max={det_per_frame.max()}")
    else:
        print("\n⚠ No detections! Check score_threshold or model quality.")


if __name__ == '__main__':
    main()
