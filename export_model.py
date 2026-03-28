"""
Export PointPillars + CenterHead model to PyTorch (.pt) and ONNX formats.

Usage:
    python export_model.py --checkpoint checkpoints/best.pth --output-dir exported_models/

This will create:
    - exported_models/pointpillars_weights.pth    (state_dict only, smallest)
    - exported_models/pointpillars_full.pt        (full model with config)
    - exported_models/pointpillars.onnx           (ONNX for deployment)
    - exported_models/config.yaml                 (config for reference)
"""

import argparse
from pathlib import Path
import yaml
import torch
import torch.onnx

# Add parent to path if running as script
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pointpillars.config import Config
from pointpillars.model import PointPillarsCenterHead


def export_pytorch_weights(model, output_path):
    """Export just the state_dict (smallest file, needs Config to reload)."""
    torch.save(model.state_dict(), output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Weights only: {output_path} ({size_mb:.1f} MB)")


def export_pytorch_full(model, cfg, checkpoint_info, output_path):
    """Export full model with config and metadata (easy to reload)."""
    # Build save dict with safe attribute access
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
        # Metadata
        'original_checkpoint': str(checkpoint_info.get('path', 'unknown')),
        'original_epoch': checkpoint_info.get('epoch', 'unknown'),
        'original_val_loss': checkpoint_info.get('val_loss', 'unknown'),
        'export_format': 'pytorch_full',
    }
    
    # Add config attributes if they exist (handles different Config versions)
    config_attrs = [
        'class_names', 'num_classes', 'point_cloud_range',
        'pillar_x', 'pillar_y', 'pillar_z',
        'grid_x', 'grid_y', 'grid_z',
        'max_pillars', 'max_points_per_pillar',
        'pillar_feat_channels', 'head_stride',
        'heatmap_h', 'heatmap_w',
        'score_threshold', 'nms_kernel', 'max_detections',
    ]
    
    for attr in config_attrs:
        if hasattr(cfg, attr):
            save_dict[attr] = getattr(cfg, attr)
    
    torch.save(save_dict, output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Full model:   {output_path} ({size_mb:.1f} MB)")


def export_torchscript(model, cfg, output_path):
    """Export as TorchScript (good for C++ deployment)."""
    model.eval()
    
    # Create example inputs
    B = 1
    max_pillars = cfg.max_pillars
    max_points = cfg.max_points_per_pillar
    
    example_pillar_features = torch.randn(B, max_pillars, max_points, 9)
    example_num_points = torch.randint(1, max_points, (B, max_pillars))
    example_pillar_coords = torch.randint(0, min(cfg.grid_x, cfg.grid_y), (B, max_pillars, 2))
    
    try:
        # Try tracing
        traced = torch.jit.trace(
            model,
            (example_pillar_features, example_num_points, example_pillar_coords, B)
        )
        traced.save(str(output_path))
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  ✓ TorchScript:  {output_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ⚠ TorchScript export failed: {e}")
        return False


def export_onnx(model, cfg, output_path, opset_version=14):
    """Export to ONNX format."""
    model.eval()
    
    # Create example inputs matching your model's forward signature
    B = 1
    max_pillars = cfg.max_pillars
    max_points = cfg.max_points_per_pillar
    
    example_pillar_features = torch.randn(B, max_pillars, max_points, 9)
    example_num_points = torch.randint(1, max_points, (B, max_pillars))
    example_pillar_coords = torch.randint(0, min(cfg.grid_x, cfg.grid_y), (B, max_pillars, 2))
    
    # Input names for ONNX
    input_names = ['pillar_features', 'num_points_per_pillar', 'pillar_coords']
    output_names = ['heatmap', 'offset', 'z', 'dim', 'rot']
    
    # Dynamic axes for variable batch size
    dynamic_axes = {
        'pillar_features': {0: 'batch_size'},
        'num_points_per_pillar': {0: 'batch_size'},
        'pillar_coords': {0: 'batch_size'},
        'heatmap': {0: 'batch_size'},
        'offset': {0: 'batch_size'},
        'z': {0: 'batch_size'},
        'dim': {0: 'batch_size'},
        'rot': {0: 'batch_size'},
    }
    
    try:
        # We need a wrapper since the model takes batch_size as a separate arg
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, pillar_features, num_points_per_pillar, pillar_coords):
                batch_size = pillar_features.shape[0]
                out = self.model(pillar_features, num_points_per_pillar, pillar_coords, batch_size)
                return out['heatmap'], out['offset'], out['z'], out['dim'], out['rot']
        
        wrapper = ONNXWrapper(model)
        wrapper.eval()
        
        torch.onnx.export(
            wrapper,
            (example_pillar_features, example_num_points, example_pillar_coords),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  ✓ ONNX:         {output_path} ({size_mb:.1f} MB)")
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print(f"    ONNX model verified successfully")
        except ImportError:
            print(f"    (install 'onnx' package to verify the exported model)")
        except Exception as e:
            print(f"    ONNX verification warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_config_yaml(cfg, output_path):
    """Save config as YAML for reference."""
    config_attrs = [
        'class_names', 'num_classes', 'point_cloud_range',
        'pillar_x', 'pillar_y', 'pillar_z',
        'grid_x', 'grid_y', 'grid_z',
        'max_pillars', 'max_points_per_pillar',
        'pillar_feat_channels', 'head_stride',
        'heatmap_h', 'heatmap_w',
        'score_threshold', 'nms_kernel', 'max_detections',
    ]
    
    config_dict = {}
    for attr in config_attrs:
        if hasattr(cfg, attr):
            config_dict[attr] = getattr(cfg, attr)
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Config YAML:  {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export PointPillars model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                        help='Output directory for exported models')
    parser.add_argument('--no-onnx', action='store_true',
                        help='Skip ONNX export')
    parser.add_argument('--no-torchscript', action='store_true',
                        help='Skip TorchScript export')
    parser.add_argument('--onnx-opset', type=int, default=14,
                        help='ONNX opset version')
    args = parser.parse_args()
    
    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    print(f"\n{'═'*60}")
    print(f"  EXPORTING MODEL")
    print(f"{'═'*60}")
    print(f"  Checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Get config from checkpoint
    if 'config' in ckpt:
        cfg = ckpt['config']
        print(f"  Config: loaded from checkpoint")
    else:
        cfg = Config()
        print(f"  Config: using default Config()")
    
    epoch = ckpt.get('epoch', 'unknown')
    val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', 'unknown'))
    print(f"  Epoch: {epoch}")
    print(f"  Val loss: {val_loss}")
    
    # Build model and load weights
    model = PointPillarsCenterHead(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Output directory: {output_dir}")
    print(f"\nExporting...\n")
    
    checkpoint_info = {
        'path': ckpt_path,
        'epoch': epoch,
        'val_loss': val_loss,
    }
    
    # Export formats
    export_pytorch_weights(model, output_dir / 'pointpillars_weights.pth')
    export_pytorch_full(model, cfg, checkpoint_info, output_dir / 'pointpillars_full.pt')
    save_config_yaml(cfg, output_dir / 'config.yaml')
    
    if not args.no_torchscript:
        export_torchscript(model, cfg, output_dir / 'pointpillars_scripted.pt')
    
    if not args.no_onnx:
        export_onnx(model, cfg, output_dir / 'pointpillars.onnx', args.onnx_opset)
    
    print(f"\n{'═'*60}")
    print(f"  EXPORT COMPLETE")
    print(f"{'═'*60}")
    print(f"\nTo reload the model later:\n")
    print(f"  # Option 1: Full checkpoint (easiest)")
    print(f"  ckpt = torch.load('exported_models/pointpillars_full.pt')")
    print(f"  cfg = ckpt['config']")
    print(f"  model = PointPillarsCenterHead(cfg)")
    print(f"  model.load_state_dict(ckpt['model_state_dict'])")
    print(f"\n  # Option 2: Weights only (need matching config)")
    print(f"  model = PointPillarsCenterHead(cfg)")
    print(f"  model.load_state_dict(torch.load('exported_models/pointpillars_weights.pth'))")
    print(f"\n  # Option 3: ONNX (for deployment)")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('exported_models/pointpillars.onnx')")
    print()


if __name__ == '__main__':
    main()