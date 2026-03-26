"""
PointPillars + CenterHead — Training Script
=============================================
Features:
  - Resumes automatically from checkpoints/latest.pth if it exists
  - Saves every epoch AND every N batches (mid-epoch safety net)
  - Tracks best val loss across restarts

Usage:
    python -m pointpillars.train
    python -m pointpillars.train --epochs 40 --lr 3e-4 --batch 4

    # Explicitly resume (auto-detected anyway)
    python -m pointpillars.train --resume

    # Fresh start even if checkpoint exists
    python -m pointpillars.train --fresh
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .config import Config
from .model import PointPillarsCenterHead
from .dataset import build_dataloaders
from .losses import CenterHeadLoss
from .utils import decode_predictions


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════════

CKPT_DIR = Path("checkpoints")
SAVE_EVERY_N_BATCHES = 100   # mid-epoch safety save


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, cfg):
    """Save a full resumable checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': cfg,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint → returns (start_epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt.get('epoch', 0)
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    return start_epoch, best_val_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Training & validation loops
# ═══════════════════════════════════════════════════════════════════════════════

def move_to_device(batch, device):
    """Move batch tensors to device, skip metadata (strings/lists)."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, criterion, optimizer, scheduler, cfg, epoch,
                    best_val_loss):
    """Train for one epoch with mid-epoch checkpointing. Returns avg loss dict."""
    model.train()
    device = cfg.device
    running = {}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        batch = move_to_device(batch, device)

        # Forward
        preds = model(
            pillar_features=batch['pillar_features'],
            num_points_per_pillar=batch['num_points'],
            pillar_coords=batch['pillar_coords'],
            batch_size=cfg.batch_size,
        )

        # Loss
        targets = {
            'heatmap': batch['heatmap'],
            'offset': batch['offset'],
            'z_target': batch['z_target'],
            'dim_target': batch['dim_target'],
            'rot_target': batch['rot_target'],
            'reg_mask': batch['reg_mask'],
        }

        total_loss, loss_dict = criterion(preds, targets)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Accumulate
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0) + v
        n_batches += 1

        # ── Mid-epoch checkpoint (safety net) ─────────────────────────
        if (batch_idx + 1) % SAVE_EVERY_N_BATCHES == 0:
            save_checkpoint(
                CKPT_DIR / 'latest.pth', model, optimizer, scheduler,
                epoch, best_val_loss, cfg,
            )
            avg_total = running['total'] / n_batches
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader)} | "
                  f"loss={avg_total:.4f}  lr={lr:.2e}  [saved checkpoint]")

        elif (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            lr = optimizer.param_groups[0]['lr']
            avg_total = running['total'] / n_batches
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader)} | "
                  f"loss={avg_total:.4f}  lr={lr:.2e}")

    return {k: v / n_batches for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, criterion, cfg):
    """Validate and return average loss dict + detection count."""
    model.eval()
    device = cfg.device
    running = {}
    n_batches = 0
    total_detections = 0

    for batch in loader:
        batch = move_to_device(batch, device)

        preds = model(
            pillar_features=batch['pillar_features'],
            num_points_per_pillar=batch['num_points'],
            pillar_coords=batch['pillar_coords'],
            batch_size=len(batch['scenes']),
        )

        targets = {
            'heatmap': batch['heatmap'],
            'offset': batch['offset'],
            'z_target': batch['z_target'],
            'dim_target': batch['dim_target'],
            'rot_target': batch['rot_target'],
            'reg_mask': batch['reg_mask'],
        }

        _, loss_dict = criterion(preds, targets)

        # Also run decoding to check detection quality
        results = decode_predictions(preds, cfg)
        for r in results:
            total_detections += len(r['boxes'])

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0) + v
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg_det_per_frame = total_detections / max(n_batches * cfg.batch_size, 1)
    return avg, avg_det_per_frame


# ═══════════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg, resume=True):
    """Full training loop with automatic checkpoint resume."""
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    cfg.device = str(device)
    print(f"Using device: {device}")

    cfg.print_summary()

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = PointPillarsCenterHead(cfg).to(device)
    total_params, trainable_params = model.count_parameters()
    print(f"Model: {trainable_params:,} trainable params ({total_params:,} total)")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_epochs / cfg.num_epochs,
        anneal_strategy='cos',
    )

    criterion = CenterHeadLoss(cfg).to(device)

    # ── Resume from checkpoint ────────────────────────────────────────────
    CKPT_DIR.mkdir(exist_ok=True)
    latest_path = CKPT_DIR / 'latest.pth'
    start_epoch = 0
    best_val_loss = float('inf')

    if resume and latest_path.exists():
        print(f"\n  ↻ Resuming from {latest_path}")
        start_epoch, best_val_loss = load_checkpoint(
            latest_path, model, optimizer, scheduler
        )
        # Move optimizer state to GPU (it was loaded on CPU)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"    Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        print()
    elif resume:
        print(f"\n  No checkpoint found at {latest_path} — starting fresh\n")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, cfg.num_epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, scheduler, cfg, epoch,
                                     best_val_loss)

        # Validate
        val_loss, avg_det = validate(model, val_loader, criterion, cfg)

        dt = time.time() - t0

        # Print summary
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch:3d}/{cfg.num_epochs}  [{dt:.0f}s]")
        print(f"  Train loss: {train_loss['total']:.4f}  "
              f"(hm={train_loss['heatmap']:.4f}  off={train_loss['offset']:.4f}  "
              f"z={train_loss['z']:.4f}  dim={train_loss['dim']:.4f}  "
              f"rot={train_loss['rot']:.4f})")
        print(f"  Val   loss: {val_loss['total']:.4f}  "
              f"(hm={val_loss['heatmap']:.4f}  off={val_loss['offset']:.4f}  "
              f"z={val_loss['z']:.4f}  dim={val_loss['dim']:.4f}  "
              f"rot={val_loss['rot']:.4f})")
        print(f"  Avg detections/frame (val): {avg_det:.1f}")
        print(f"{'─'*70}\n")

        # ── Always save latest (for resume) ───────────────────────────
        save_checkpoint(
            latest_path, model, optimizer, scheduler,
            epoch, best_val_loss, cfg,
        )

        # ── Save best ─────────────────────────────────────────────────
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            best_path = CKPT_DIR / 'best.pth'
            save_checkpoint(
                best_path, model, optimizer, scheduler,
                epoch, best_val_loss, cfg,
            )
            print(f"  ★ New best model saved → {best_path}")

        # ── Periodic milestone ────────────────────────────────────────
        if epoch % 20 == 0:
            mile_path = CKPT_DIR / f'epoch_{epoch:03d}.pth'
            save_checkpoint(
                mile_path, model, optimizer, scheduler,
                epoch, best_val_loss, cfg,
            )

    print(f"\n{'='*70}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {CKPT_DIR}/")
    print(f"  latest.pth  — last epoch (for resume)")
    print(f"  best.pth    — lowest val loss")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train PointPillars + CenterHead")
    parser.add_argument('--processed-dir', default='processed/', help='Preprocessed data dir')
    parser.add_argument('--gt-csv', default='gt_bboxes_run_05_merge_clean.csv', help='GT CSV')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from latest checkpoint (default: True)')
    parser.add_argument('--fresh', action='store_true', default=False,
                        help='Ignore existing checkpoints, start fresh')
    args = parser.parse_args()

    cfg = Config()
    cfg.processed_dir = args.processed_dir
    cfg.gt_csv = args.gt_csv
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.lr is not None:
        cfg.lr = args.lr
    if args.workers is not None:
        cfg.num_workers = args.workers

    train(cfg, resume=not args.fresh)


if __name__ == '__main__':
    main()