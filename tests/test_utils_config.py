#!/usr/bin/env python3
"""
Tests for the configuration module.

Run with: pytest tests/test_utils_config.py -v
"""

import pytest
import numpy as np
import os


class TestSetupMatplotlib:
    """Test setup_matplotlib function."""

    def test_default_backend(self):
        """Test default Agg backend."""
        from scripts.utils.config import setup_matplotlib
        import matplotlib

        # Clear any existing backend
        matplotlib.rcdefaults()

        setup_matplotlib()

        # Should be Agg (case-insensitive)
        assert matplotlib.get_backend().lower() == 'agg'

    def test_custom_backend(self):
        """Test custom backend via argument."""
        from scripts.utils.config import setup_matplotlib
        import matplotlib

        setup_matplotlib(backend='Agg')

        assert matplotlib.get_backend().lower() == 'agg'

    def test_env_variable_backend(self, monkeypatch):
        """Test backend from environment variable."""
        from scripts.utils.config import setup_matplotlib
        import matplotlib

        monkeypatch.setenv('MATPLOTLIB_BACKEND', 'Agg')

        setup_matplotlib()

        assert matplotlib.get_backend().lower() == 'agg'


class TestGaitCorrectionConfig:
    """Test GaitCorrectionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from scripts.utils.config import GaitCorrectionConfig

        config = GaitCorrectionConfig()

        assert config.frame_rate == 60
        assert config.pelvis_index == 0
        assert config.drift_window_seconds == 30.0
        assert config.highpass_window_seconds == 10.0
        assert config.max_drift_threshold == 2.0
        assert config.pca_window_seconds == 30.0
        assert config.pca_sample_interval_seconds == 5.0
        assert config.pca_smoothing_factor == 0.1
        assert config.reference_seconds == 60.0
        assert config.skip_start_seconds == 5.0
        assert config.skip_end_seconds == 5.0
        assert config.min_segment_seconds == 5.0
        assert config.smooth_transition_seconds == 0.5

    def test_custom_values(self):
        """Test custom configuration values."""
        from scripts.utils.config import GaitCorrectionConfig

        config = GaitCorrectionConfig(
            frame_rate=100,
            pelvis_index=1,
            drift_window_seconds=20.0,
            pca_window_seconds=25.0,
        )

        assert config.frame_rate == 100
        assert config.pelvis_index == 1
        assert config.drift_window_seconds == 20.0
        assert config.pca_window_seconds == 25.0

    def test_drift_window_frames(self):
        """Test drift_window_frames property."""
        from scripts.utils.config import GaitCorrectionConfig

        config = GaitCorrectionConfig(frame_rate=60, drift_window_seconds=30.0)

        assert config.drift_window_frames == 1800  # 30 * 60

    def test_highpass_window_frames_odd(self):
        """Test highpass_window_frames returns odd number."""
        from scripts.utils.config import GaitCorrectionConfig

        # With 10 seconds at 60 Hz = 600 frames (even)
        config = GaitCorrectionConfig(frame_rate=60, highpass_window_seconds=10.0)

        # Should be 601 (odd)
        assert config.highpass_window_frames == 601
        assert config.highpass_window_frames % 2 == 1

    def test_pca_window_frames(self):
        """Test pca_window_frames property."""
        from scripts.utils.config import GaitCorrectionConfig

        config = GaitCorrectionConfig(frame_rate=100, pca_window_seconds=20.0)

        assert config.pca_window_frames == 2000  # 20 * 100

    def test_reference_frames(self):
        """Test reference_frames property."""
        from scripts.utils.config import GaitCorrectionConfig

        config = GaitCorrectionConfig(frame_rate=60, reference_seconds=60.0)

        assert config.reference_frames == 3600  # 60 * 60


class TestSegmentExtractionConfig:
    """Test SegmentExtractionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from scripts.utils.config import SegmentExtractionConfig

        config = SegmentExtractionConfig()

        assert config.frame_rate == 60
        assert config.velocity_threshold == 0.4
        assert config.min_segment_meters == 5.0
        assert config.heading_change_threshold == 0.1
        assert config.heading_change_threshold_raw == 0.3
        assert config.trim_start_seconds == 0.5
        assert config.trim_end_seconds == 0.3
        assert config.use_overlapping_windows == True
        assert config.window_seconds == 5.0
        assert config.window_overlap == 0.5

    def test_trim_start_frames(self):
        """Test trim_start_frames property."""
        from scripts.utils.config import SegmentExtractionConfig

        config = SegmentExtractionConfig(frame_rate=60, trim_start_seconds=0.5)

        assert config.trim_start_frames == 30  # 0.5 * 60

    def test_trim_end_frames(self):
        """Test trim_end_frames property."""
        from scripts.utils.config import SegmentExtractionConfig

        config = SegmentExtractionConfig(frame_rate=100, trim_end_seconds=0.3)

        assert config.trim_end_frames == 30  # 0.3 * 100

    def test_window_frames(self):
        """Test window_frames property."""
        from scripts.utils.config import SegmentExtractionConfig

        config = SegmentExtractionConfig(frame_rate=60, window_seconds=5.0)

        assert config.window_frames == 300  # 5 * 60

    def test_min_segment_frames(self):
        """Test min_segment_frames property."""
        from scripts.utils.config import SegmentExtractionConfig

        config = SegmentExtractionConfig(frame_rate=60, min_segment_meters=5.0)

        # Assuming ~1 m/s walking speed: 5m = 5s = 300 frames
        assert config.min_segment_frames == 300


class TestPlottingConfig:
    """Test PlottingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from scripts.utils.config import PlottingConfig

        config = PlottingConfig()

        assert config.figsize_single == (10, 8)
        assert config.figsize_comparison == (14, 6)
        assert config.figsize_multi == (15, 10)
        assert config.dpi == 150
        assert config.colormap == 'viridis'
        assert config.grid_alpha == 0.3
        assert config.scatter_alpha == 0.5
        assert config.scatter_size == 0.5
        assert config.line_width == 0.5

    def test_custom_values(self):
        """Test custom configuration values."""
        from scripts.utils.config import PlottingConfig

        config = PlottingConfig(
            dpi=300,
            colormap='plasma',
            grid_alpha=0.5,
        )

        assert config.dpi == 300
        assert config.colormap == 'plasma'
        assert config.grid_alpha == 0.5


class TestTurnaroundDetectionConfig:
    """Test TurnaroundDetectionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from scripts.utils.config import TurnaroundDetectionConfig

        config = TurnaroundDetectionConfig()

        assert config.frame_rate == 60
        assert config.velocity_smooth_window_seconds == 0.5
        assert config.min_pause_seconds == 0.3
        assert config.velocity_threshold_ratio == 0.3
        assert config.direction_change_threshold == 120.0

    def test_velocity_smooth_frames_odd(self):
        """Test velocity_smooth_frames returns odd number."""
        from scripts.utils.config import TurnaroundDetectionConfig

        # 0.5s at 60Hz = 30 frames (even)
        config = TurnaroundDetectionConfig(
            frame_rate=60,
            velocity_smooth_window_seconds=0.5,
        )

        # Should be 31 (odd)
        assert config.velocity_smooth_frames == 31
        assert config.velocity_smooth_frames % 2 == 1

    def test_min_pause_frames(self):
        """Test min_pause_frames property."""
        from scripts.utils.config import TurnaroundDetectionConfig

        config = TurnaroundDetectionConfig(frame_rate=60, min_pause_seconds=0.3)

        assert config.min_pause_frames == 18  # 0.3 * 60


class TestLSTMPreprocessingConfig:
    """Test LSTMPreprocessingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from scripts.utils.config import LSTMPreprocessingConfig

        config = LSTMPreprocessingConfig()

        assert config.frame_rate == 60
        assert config.target_duration_seconds == 5.0
        assert config.min_duration_seconds == 3.0
        assert config.max_duration_seconds == 10.0
        assert config.normalize_segments == True
        assert config.preserve_gait_features == True
        assert config.drift_correction_strength == 'moderate'

    def test_target_frames(self):
        """Test target_frames property."""
        from scripts.utils.config import LSTMPreprocessingConfig

        config = LSTMPreprocessingConfig(frame_rate=60, target_duration_seconds=5.0)

        assert config.target_frames == 300  # 5 * 60

    def test_min_max_frames(self):
        """Test min_frames and max_frames properties."""
        from scripts.utils.config import LSTMPreprocessingConfig

        config = LSTMPreprocessingConfig(
            frame_rate=100,
            min_duration_seconds=3.0,
            max_duration_seconds=10.0,
        )

        assert config.min_frames == 300  # 3 * 100
        assert config.max_frames == 1000  # 10 * 100


class TestGetDefaultConfigs:
    """Test get_default_configs function."""

    def test_returns_all_configs(self):
        """Test that all config types are returned."""
        from scripts.utils.config import get_default_configs

        configs = get_default_configs()

        assert 'gait_correction' in configs
        assert 'segment_extraction' in configs
        assert 'plotting' in configs
        assert 'turnaround' in configs
        assert 'lstm' in configs

    def test_correct_types(self):
        """Test that configs have correct types."""
        from scripts.utils.config import (
            get_default_configs,
            GaitCorrectionConfig,
            SegmentExtractionConfig,
            PlottingConfig,
            TurnaroundDetectionConfig,
            LSTMPreprocessingConfig,
        )

        configs = get_default_configs()

        assert isinstance(configs['gait_correction'], GaitCorrectionConfig)
        assert isinstance(configs['segment_extraction'], SegmentExtractionConfig)
        assert isinstance(configs['plotting'], PlottingConfig)
        assert isinstance(configs['turnaround'], TurnaroundDetectionConfig)
        assert isinstance(configs['lstm'], LSTMPreprocessingConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
