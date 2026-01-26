# Gait Skeleton Visualization Package
"""
A package for visualizing gait analysis marker data as skeleton animations.
"""

from .data_loader import DataLoader
from .skeleton_model import SkeletonModel
from .visualizer_3d import Visualizer3D
from .visualizer_2d import Visualizer2D
from .video_exporter import VideoExporter
from .phase_dynamics import compute_joint_angles
from .visualizer_combined import CombinedVisualizer

__version__ = "1.0.0"
__all__ = [
    "DataLoader",
    "SkeletonModel",
    "Visualizer3D",
    "Visualizer2D",
    "VideoExporter",
    "compute_joint_angles",
    "CombinedVisualizer",
]
