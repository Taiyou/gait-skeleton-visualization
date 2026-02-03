"""
Shared utilities for gait analysis scripts.

This module provides common functionality used across experiment scripts,
including plotting utilities, configuration management, and matplotlib setup.
"""

from scripts.utils.plotting import (
    plot_trajectory,
    plot_trajectory_comparison,
    plot_multi_method_comparison,
    plot_time_series_comparison,
    plot_segment_distribution,
    create_figure_grid,
    PlotConfig,
)
from scripts.utils.config import (
    setup_matplotlib,
    GaitCorrectionConfig,
    SegmentExtractionConfig,
    PlottingConfig,
)

__all__ = [
    # Plotting
    "plot_trajectory",
    "plot_trajectory_comparison",
    "plot_multi_method_comparison",
    "plot_time_series_comparison",
    "plot_segment_distribution",
    "create_figure_grid",
    "PlotConfig",
    # Configuration
    "setup_matplotlib",
    "GaitCorrectionConfig",
    "SegmentExtractionConfig",
    "PlottingConfig",
]
