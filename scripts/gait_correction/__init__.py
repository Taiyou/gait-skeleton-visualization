"""
Gait Correction Module
Provides tools for noise removal and drift correction in motion capture data.
"""

from .loader import load_xsens_data, XsensDataLoader
from .smooth_pca import apply_smooth_pca_correction, SmoothPCAParams
from .drift_removal import (
    remove_y_drift,
    align_horizontally,
    apply_full_drift_correction,
)
from .export import export_to_csv

__all__ = [
    "load_xsens_data",
    "XsensDataLoader",
    "apply_smooth_pca_correction",
    "SmoothPCAParams",
    "remove_y_drift",
    "align_horizontally",
    "apply_full_drift_correction",
    "export_to_csv",
]
