#!/usr/bin/env python3
"""
PointPillars Inference
======================
Run trained PointPillars model on preprocessed LiDAR frames.

Works for both training data validation and final evaluation.

Usage as CLI:
    # From package
    python -m pointpillars.inference --checkpoint best.pth --processed-dir cleaned/

    # Standalone
    python inference.py --checkpoint best.pth --processed-dir cleaned/

Usage in notebook:
    from pointpillars.inference import run_inference, load_model
    
    model, cfg, device = load_model('checkpoints/best.pth')
    results = run_inference(model, loader, cfg, device)

Usage for evaluation day:
    python -m pointpillars.inference \\
        --checkpoint checkpoints/best.pth \\
        --processed-dir test_cleaned/ \\
        --output submission.csv \\
        --threshold 0.2
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS (handle both module and standalone usage)
# ══════════════════════════════════════════════════════════════════════════════

def _import_pointpillars():
    """Import PointPillars components, handling both module and standalone usage."""
    try:
        # Try relative imports first (when used as module)
        from .config import Config
        from .model import PointPillarsCenterHead
        from .dataset import AirbusLidarDataset, collate_fn
        from .utils import decode_predictions
        return Config, PointPillarsCenterHead, AirbusLidarDataset, collate_fn, decode_predictions
    except ImportError:
        try:
            # Try absolute imports (when used standalone)
            from pointpillars.config import Config
            from pointpillars.model import PointPillarsCenterHead
            from pointpillars.dataset import AirbusLidarDataset, collate_fn
            from pointpillars.utils import decode_predictions
            return Config, PointPillarsCenterHead, AirbusLidarDataset, collate_fn, decode_predictions
        except ImportError as e:
            print(f"✗ Failed to import PointPillars components: {e}")
            print("\n  Make sure pointpillars package is in your Python path:")
            print("    export PYTHONPATH=$PYTHONPATH:/path/to/pointpillars")
            print("  Or install it:")
            print("    pip install -e /path/to/pointpillars")
            sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    score_threshold: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, object, torch.device]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        device: Device to load to (auto-detected if None)
        score_threshold: Override detection threshold
        verbose: Print loading info
    
    Returns:
        model: Loaded model in eval mode
        cfg: Config object
        device: Device model is on
    
    Example:
        model, cfg, device = load_model('checkpoints/best.pth')
    """
    Config, PointPillarsCenterHead, _, _, _ = _import_pointpillars()
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct config
    cfg = ckpt.get('config', Config())
    
    if score_threshold is not None:
        cfg.score_threshold = score_threshold
    
    # Build and load model
    model = PointPillarsCenterHead(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    if verbose:
        print(f"✓ Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
        val_loss = ckpt.get('val_loss')
        if val_loss is not None:
            print(f"  Val loss: {val_loss:.4f}")
        print(f"  Score threshold: {cfg.score_threshold}")
        if hasattr(model, 'count_parameters'):
            total, trainable = model.count_parameters()
            print(f"  Parameters: {trainable:,}")
    
    return model, cfg, device


def build_dataloader(
    cfg,
    processed_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[DataLoader, int]:
    """
    Build DataLoader for inference.
    
    Args:
        cfg: Config object
        processed_dir: Directory with preprocessed .npz files
        batch_size: Batch size
        num_workers: DataLoader workers
        max_frames: Limit number of frames (for testing)
        verbose: Print info
    
    Returns:
        loader: DataLoader
        num_frames: Number of frames
    
    Example:
        loader, num_frames = build_dataloader(cfg, 'cleaned/', batch_size=8)
    """
    _, _, AirbusLidarDataset, collate_fn, _ = _import_pointpillars()
    
    # Update config
    cfg.batch_size = batch_size
    cfg.num_workers = num_workers
    cfg.processed_dir = str(processed_dir)
    
    # Build dataset (no GT needed for inference)
    dataset = AirbusLidarDataset(
        cfg,
        split='all',
        gt_csv=None,
        processed_dir=str(processed_dir)
    )
    
    # Limit frames if requested
    if max_frames and max_frames < len(dataset):
        dataset.samples = dataset.samples[:max_frames]
        if verbose:
            print(f"  ⚠ Limited to {max_frames} frames (--max-frames)")
    
    # Build loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    if verbose:
        print(f"✓ Loaded {len(dataset)} frames from {processed_dir}")
    
    return loader, len(dataset)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg,
    device: torch.device,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run inference on all frames.
    
    Args:
        model: Trained model (will be set to eval mode)
        loader: DataLoader with preprocessed frames
        cfg: Config object
        device: Device to run on
        verbose: Show progress bar
    
    Returns:
        List of detection dicts, each containing:
            - ego_x, ego_y, ego_z, ego_yaw (frame identifier)
            - bbox_center_x/y/z, bbox_width/length/height, bbox_yaw
            - class_ID, class_label
            - confidence, scene, pose_index (extras)
    
    Example:
        results = run_inference(model, loader, cfg, device)
        df = pd.DataFrame(results)
    """
    _, _, _, _, decode_predictions = _import_pointpillars()
    
    model.eval()
    all_results = []
    
    iterator = tqdm(loader, desc="Inference") if verbose else loader
    
    for batch_idx, batch in enumerate(iterator):
        # Move to device
        pf = batch['pillar_features'].to(device)
        np_ = batch['num_points'].to(device)
        pc = batch['pillar_coords'].to(device)
        B = len(batch['scenes'])
        
        # Forward pass
        preds = model(pf, np_, pc, B)
        results = decode_predictions(preds, cfg)
        
        # Collect results for each frame in batch
        for i, res in enumerate(results):
            scene = batch['scenes'][i]
            pose_index = batch['pose_indices'][i]
            ego = batch['ego_poses'][i]
            
            boxes = res['boxes'].cpu().numpy()    # (K, 7)
            scores = res['scores'].cpu().numpy()
            labels = res['labels'].cpu().numpy()
            
            for j in range(len(boxes)):
                cx, cy, cz, w, l, h, yaw = boxes[j]
                cid = int(labels[j])
                cls_name = cfg.class_names[cid]
                
                all_results.append({
                    # Ego pose (frame identifier)
                    'ego_x': float(ego[0]),
                    'ego_y': float(ego[1]),
                    'ego_z': float(ego[2]),
                    'ego_yaw': float(ego[3]),
                    # Bounding box
                    'bbox_center_x': float(cx),
                    'bbox_center_y': float(cy),
                    'bbox_center_z': float(cz),
                    'bbox_width': float(w),
                    'bbox_length': float(l),
                    'bbox_height': float(h),
                    'bbox_yaw': float(yaw),
                    # Class
                    'class_ID': cid,
                    'class_label': cls_name,
                    # Extras (not in competition format)
                    'confidence': float(scores[j]),
                    'scene': scene,
                    'pose_index': pose_index,
                })
    
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS HANDLING
# ══════════════════════════════════════════════════════════════════════════════

# Competition format columns
COMPETITION_COLUMNS = [
    'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
    'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
    'bbox_width', 'bbox_length', 'bbox_height', 'bbox_yaw',
    'class_ID', 'class_label',
]


def save_predictions(
    results: List[Dict],
    output_path: Path,
    save_submission: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Save predictions to CSV.
    
    Args:
        results: List of detection dicts from run_inference
        output_path: Output CSV path
        save_submission: Also save competition format CSV
        verbose: Print info
    
    Returns:
        DataFrame with all predictions
    """
    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full CSV
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"✓ Saved predictions → {output_path}")
    
    # Save competition format
    if save_submission:
        submission_path = output_path.with_name(
            output_path.stem + '_submission' + output_path.suffix
        )
        df[COMPETITION_COLUMNS].to_csv(submission_path, index=False)
        if verbose:
            print(f"✓ Saved submission → {submission_path}")
    
    return df


def print_summary(df: pd.DataFrame, num_frames: int):
    """Print detection summary statistics."""
    print(f"\n{'═'*60}")
    print("  INFERENCE SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total detections: {len(df):,}")
    print(f"  Frames processed: {num_frames}")
    
    if len(df) > 0:
        print(f"\n  Class breakdown:")
        for cls, count in df['class_label'].value_counts().items():
            print(f"    {cls}: {count:,}")
        
        print(f"\n  Confidence stats:")
        print(f"    Mean: {df['confidence'].mean():.3f}")
        print(f"    Min:  {df['confidence'].min():.3f}")
        print(f"    Max:  {df['confidence'].max():.3f}")
        
        det_per_frame = df.groupby(['scene', 'pose_index']).size()
        print(f"\n  Detections per frame:")
        print(f"    Mean: {det_per_frame.mean():.1f}")
        print(f"    Max:  {det_per_frame.max()}")
    else:
        print("\n  ⚠ No detections!")
        print("    Try: lowering --threshold")
        print("    Or:  check model quality")


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ══════════════════════════════════════════════════════════════════════════════

def predict(
    checkpoint_path: str,
    processed_dir: str,
    output_path: str = 'predictions.csv',
    batch_size: int = 4,
    num_workers: int = 4,
    score_threshold: Optional[float] = None,
    max_frames: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    High-level prediction function — load model, run inference, save results.
    
    Args:
        checkpoint_path: Path to trained model
        processed_dir: Directory with preprocessed frames
        output_path: Output CSV path
        batch_size: Batch size
        num_workers: DataLoader workers
        score_threshold: Override detection threshold
        max_frames: Limit frames (for testing)
        device: Device (auto-detected if None)
        verbose: Print progress
    
    Returns:
        DataFrame with predictions
    
    Example:
        # Quick prediction
        df = predict('best.pth', 'cleaned/')
        
        # With custom settings
        df = predict(
            'best.pth', 
            'test_cleaned/',
            output_path='submission.csv',
            score_threshold=0.15,
            batch_size=8
        )
    """
    # Load model
    model, cfg, device = load_model(
        checkpoint_path, 
        device=device,
        score_threshold=score_threshold,
        verbose=verbose
    )
    
    # Build dataloader
    loader, num_frames = build_dataloader(
        cfg,
        processed_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_frames=max_frames,
        verbose=verbose
    )
    
    # Run inference
    if verbose:
        print(f"\nRunning inference...")
    
    results = run_inference(model, loader, cfg, device, verbose=verbose)
    
    # Save results
    df = save_predictions(results, Path(output_path), verbose=verbose)
    
    # Print summary
    if verbose:
        print_summary(df, num_frames)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run PointPillars inference on preprocessed LiDAR frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference
    python -m pointpillars.inference \\
        --checkpoint checkpoints/best.pth \\
        --processed-dir cleaned/

    # Evaluation day
    python -m pointpillars.inference \\
        --checkpoint checkpoints/best.pth \\
        --processed-dir test_cleaned/ \\
        --output submission.csv \\
        --threshold 0.2

    # Quick test
    python -m pointpillars.inference \\
        --checkpoint checkpoints/best.pth \\
        --processed-dir cleaned/ \\
        --max-frames 50
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--processed-dir', required=True, type=Path,
                        help='Directory with preprocessed .npz frames')
    
    # Output
    parser.add_argument('--output', '-o', type=Path, default=Path('predictions.csv'),
                        help='Output CSV path (default: predictions.csv)')
    
    # Inference settings
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold (overrides checkpoint config)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    
    # Testing/debug
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Limit number of frames (for testing)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda, cpu, cuda:0, etc. (auto-detected)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.checkpoint.exists():
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not args.processed_dir.exists():
        print(f"✗ Processed directory not found: {args.processed_dir}")
        sys.exit(1)
    
    manifest_path = args.processed_dir / 'manifest.csv'
    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}")
        print("  Run preprocessing or ground_removal.py first")
        sys.exit(1)
    
    # Setup device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Print header
    if not args.quiet:
        print(f"\n{'═'*60}")
        print("  POINTPILLARS INFERENCE")
        print(f"{'═'*60}")
        print(f"  Checkpoint:    {args.checkpoint}")
        print(f"  Data dir:      {args.processed_dir}")
        print(f"  Output:        {args.output}")
        print(f"  Batch size:    {args.batch_size}")
        if args.threshold:
            print(f"  Threshold:     {args.threshold}")
        print(f"{'═'*60}")
    
    # Run prediction
    predict(
        checkpoint_path=str(args.checkpoint),
        processed_dir=str(args.processed_dir),
        output_path=str(args.output),
        batch_size=args.batch_size,
        num_workers=args.workers,
        score_threshold=args.threshold,
        max_frames=args.max_frames,
        device=device,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
