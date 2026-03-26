"""
PointPillars + CenterHead for Airbus Helicopter LiDAR Obstacle Detection
=========================================================================
Custom pure-PyTorch implementation — no spconv dependency.
"""

from .config import Config
from .model import PointPillarsCenterHead
from .dataset import AirbusLidarDataset, build_dataloaders
from .losses import CenterHeadLoss
from .utils import decode_predictions, pillarize

__all__ = [
    'Config',
    'PointPillarsCenterHead',
    'AirbusLidarDataset',
    'build_dataloaders',
    'CenterHeadLoss',
    'decode_predictions',
    'pillarize',
]
