#!/usr/bin/env python3
"""
Centralized configuration for gait analysis scripts.

This module provides:
1. Matplotlib backend configuration
2. Common parameter dataclasses for gait correction and analysis
3. Eliminates magic numbers scattered across scripts

Usage:
    from scripts.utils.config import (
        setup_matplotlib,
        GaitCorrectionConfig,
        SegmentExtractionConfig,
    )

    # Setup matplotlib (call once at module level)
    setup_matplotlib()

    # Use configuration objects
    config = GaitCorrectionConfig()
    window_frames = int(config.highpass_window_seconds * config.frame_rate)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


def setup_matplotlib(backend: Optional[str] = None) -> None:
    """Configure matplotlib with specified backend.

    This should be called before importing matplotlib.pyplot in scripts.
    The default backend is 'Agg' for non-interactive rendering, which is
    suitable for generating plots to files.

    Args:
        backend: Matplotlib backend name. Defaults to 'Agg'.
                 Can be overridden via MATPLOTLIB_BACKEND env variable.
    """
    import matplotlib

    if backend is None:
        backend = os.environ.get('MATPLOTLIB_BACKEND', 'Agg')

    matplotlib.use(backend)


@dataclass
class GaitCorrectionConfig:
    """Configuration for gait drift correction algorithms.

    Centralizes the magic numbers used across correction scripts like
    drift_removal.py, smooth_pca.py, and advanced_correction.py.

    Attributes:
        frame_rate: Frame rate in Hz. Default: 60
        pelvis_index: Index of pelvis segment in data array. Default: 0

        # Drift removal parameters
        drift_window_seconds: Window size for drift estimation. Default: 30.0
        highpass_window_seconds: Window for high-pass filtering. Default: 10.0
        max_drift_threshold: Maximum expected drift in meters. Default: 2.0

        # Smooth PCA parameters
        pca_window_seconds: Window for local PCA computation. Default: 30.0
        pca_sample_interval_seconds: Interval between PCA samples. Default: 5.0
        pca_smoothing_factor: Smoothing factor for angle interpolation. Default: 0.1

        # Reference period parameters
        reference_seconds: Duration of reference period for adaptive correction. Default: 60.0
        skip_start_seconds: Seconds to skip at start (unstable period). Default: 5.0
        skip_end_seconds: Seconds to skip at end. Default: 5.0

        # Segment-based correction
        min_segment_seconds: Minimum segment length for segment-wise correction. Default: 5.0
        smooth_transition_seconds: Transition smoothing window. Default: 0.5
    """
    frame_rate: int = 60
    pelvis_index: int = 0

    # Drift removal
    drift_window_seconds: float = 30.0
    highpass_window_seconds: float = 10.0
    max_drift_threshold: float = 2.0

    # Smooth PCA
    pca_window_seconds: float = 30.0
    pca_sample_interval_seconds: float = 5.0
    pca_smoothing_factor: float = 0.1

    # Reference period
    reference_seconds: float = 60.0
    skip_start_seconds: float = 5.0
    skip_end_seconds: float = 5.0

    # Segment-based
    min_segment_seconds: float = 5.0
    smooth_transition_seconds: float = 0.5

    @property
    def drift_window_frames(self) -> int:
        """Get drift window size in frames."""
        return int(self.drift_window_seconds * self.frame_rate)

    @property
    def highpass_window_frames(self) -> int:
        """Get high-pass window size in frames (odd number)."""
        frames = int(self.highpass_window_seconds * self.frame_rate)
        return frames + 1 if frames % 2 == 0 else frames

    @property
    def pca_window_frames(self) -> int:
        """Get PCA window size in frames."""
        return int(self.pca_window_seconds * self.frame_rate)

    @property
    def reference_frames(self) -> int:
        """Get reference period length in frames."""
        return int(self.reference_seconds * self.frame_rate)


@dataclass
class SegmentExtractionConfig:
    """Configuration for gait segment extraction.

    Centralizes parameters used in segment extraction scripts like
    segment_extraction_pipeline.py and improved_segment_extraction.py.

    Attributes:
        frame_rate: Frame rate in Hz. Default: 60

        # Velocity thresholds
        velocity_threshold: Minimum velocity to consider as walking (m/s). Default: 0.4
        min_segment_meters: Minimum segment length in meters. Default: 5.0

        # Heading analysis
        heading_change_threshold: Max heading change rate (rad/frame). Default: 0.1
                                  (~5.7 degrees, suitable for straight walking)
        heading_change_threshold_raw: Higher threshold for raw data. Default: 0.3
                                      (tolerates drift-induced direction changes)

        # Trimming
        trim_start_seconds: Seconds to trim from segment start. Default: 0.5
        trim_end_seconds: Seconds to trim from segment end. Default: 0.3

        # Windowing
        use_overlapping_windows: Whether to use overlapping windows. Default: True
        window_seconds: Window size for analysis. Default: 5.0
        window_overlap: Overlap ratio between windows. Default: 0.5
    """
    frame_rate: int = 60

    # Velocity thresholds
    velocity_threshold: float = 0.4
    min_segment_meters: float = 5.0

    # Heading analysis
    heading_change_threshold: float = 0.1  # ~5.7 degrees, for preprocessed data
    heading_change_threshold_raw: float = 0.3  # Higher for raw data with drift

    # Trimming
    trim_start_seconds: float = 0.5
    trim_end_seconds: float = 0.3

    # Windowing
    use_overlapping_windows: bool = True
    window_seconds: float = 5.0
    window_overlap: float = 0.5

    @property
    def trim_start_frames(self) -> int:
        """Get trim start in frames."""
        return int(self.trim_start_seconds * self.frame_rate)

    @property
    def trim_end_frames(self) -> int:
        """Get trim end in frames."""
        return int(self.trim_end_seconds * self.frame_rate)

    @property
    def window_frames(self) -> int:
        """Get window size in frames."""
        return int(self.window_seconds * self.frame_rate)

    @property
    def min_segment_frames(self) -> int:
        """Estimate minimum segment frames based on velocity and distance."""
        # Assuming average walking speed ~1 m/s
        return int(self.min_segment_meters * self.frame_rate)


@dataclass
class PlottingConfig:
    """Configuration for visualization settings.

    Attributes:
        figsize_single: Default figure size for single plots. Default: (10, 8)
        figsize_comparison: Default figure size for comparisons. Default: (14, 6)
        figsize_multi: Default figure size for multi-method plots. Default: (15, 10)
        dpi: Resolution for saved figures. Default: 150
        colormap: Default colormap for time-based coloring. Default: 'viridis'
        grid_alpha: Grid line transparency. Default: 0.3
        scatter_alpha: Scatter point transparency. Default: 0.5
        scatter_size: Scatter point size. Default: 0.5
        line_width: Default line width. Default: 0.5
    """
    figsize_single: tuple = (10, 8)
    figsize_comparison: tuple = (14, 6)
    figsize_multi: tuple = (15, 10)
    dpi: int = 150
    colormap: str = 'viridis'
    grid_alpha: float = 0.3
    scatter_alpha: float = 0.5
    scatter_size: float = 0.5
    line_width: float = 0.5


@dataclass
class TurnaroundDetectionConfig:
    """Configuration for turnaround point detection.

    Attributes:
        frame_rate: Frame rate in Hz. Default: 60
        velocity_smooth_window_seconds: Window for velocity smoothing. Default: 0.5
        min_pause_seconds: Minimum pause duration at turnaround. Default: 0.3
        velocity_threshold_ratio: Ratio of mean velocity for threshold. Default: 0.3
        direction_change_threshold: Minimum direction change (degrees). Default: 120.0
    """
    frame_rate: int = 60
    velocity_smooth_window_seconds: float = 0.5
    min_pause_seconds: float = 0.3
    velocity_threshold_ratio: float = 0.3
    direction_change_threshold: float = 120.0

    @property
    def velocity_smooth_frames(self) -> int:
        """Get smoothing window in frames (odd number)."""
        frames = int(self.velocity_smooth_window_seconds * self.frame_rate)
        return frames + 1 if frames % 2 == 0 else frames

    @property
    def min_pause_frames(self) -> int:
        """Get minimum pause duration in frames."""
        return int(self.min_pause_seconds * self.frame_rate)


@dataclass
class LSTMPreprocessingConfig:
    """Configuration for LSTM preprocessing pipeline.

    Attributes:
        frame_rate: Frame rate in Hz. Default: 60
        target_duration_seconds: Target segment duration for LSTM. Default: 5.0
        min_duration_seconds: Minimum acceptable segment duration. Default: 3.0
        max_duration_seconds: Maximum acceptable segment duration. Default: 10.0
        normalize_segments: Whether to normalize segment lengths. Default: True
        preserve_gait_features: Whether to use feature-preserving correction. Default: True
        drift_correction_strength: Strength of drift correction. Default: 'moderate'
    """
    frame_rate: int = 60
    target_duration_seconds: float = 5.0
    min_duration_seconds: float = 3.0
    max_duration_seconds: float = 10.0
    normalize_segments: bool = True
    preserve_gait_features: bool = True
    drift_correction_strength: str = 'moderate'  # 'minimal', 'moderate', 'aggressive'

    @property
    def target_frames(self) -> int:
        """Get target segment length in frames."""
        return int(self.target_duration_seconds * self.frame_rate)

    @property
    def min_frames(self) -> int:
        """Get minimum segment length in frames."""
        return int(self.min_duration_seconds * self.frame_rate)

    @property
    def max_frames(self) -> int:
        """Get maximum segment length in frames."""
        return int(self.max_duration_seconds * self.frame_rate)


# Convenience function for creating configs with common presets
def get_default_configs() -> dict:
    """Get all default configuration objects.

    Returns:
        Dictionary with keys 'gait_correction', 'segment_extraction',
        'plotting', 'turnaround', 'lstm'
    """
    return {
        'gait_correction': GaitCorrectionConfig(),
        'segment_extraction': SegmentExtractionConfig(),
        'plotting': PlottingConfig(),
        'turnaround': TurnaroundDetectionConfig(),
        'lstm': LSTMPreprocessingConfig(),
    }
