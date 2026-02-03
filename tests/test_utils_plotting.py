#!/usr/bin/env python3
"""
Tests for the shared plotting utilities.

Run with: pytest tests/test_utils_plotting.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os


class TestPlotConfig:
    """Test PlotConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from scripts.utils.plotting import PlotConfig

        config = PlotConfig()

        assert config.figsize == (14, 6)
        assert config.dpi == 150
        assert config.colormap == "viridis"
        assert config.scatter_size == 0.5
        assert config.scatter_alpha == 0.5
        assert config.line_width == 0.5
        assert config.grid_alpha == 0.3
        assert config.title_fontsize == 11
        assert config.label_fontsize == 10
        assert config.pelvis_index == 0
        assert config.frame_rate == 60

    def test_custom_config(self):
        """Test custom configuration."""
        from scripts.utils.plotting import PlotConfig

        config = PlotConfig(
            figsize=(20, 10),
            dpi=300,
            frame_rate=100,
            pelvis_index=1,
            unit_scale=1000,
            unit_label="mm",
        )

        assert config.figsize == (20, 10)
        assert config.dpi == 300
        assert config.frame_rate == 100
        assert config.pelvis_index == 1
        assert config.unit_scale == 1000
        assert config.unit_label == "mm"


class TestCalcRange:
    """Test calc_range function."""

    def test_basic_range(self):
        """Test basic range calculation."""
        from scripts.utils.plotting import calc_range

        data = np.zeros((100, 5, 3))
        data[:, 0, 0] = np.linspace(0, 10, 100)  # X: 0 to 10
        data[:, 0, 1] = np.linspace(-2, 3, 100)  # Y: -2 to 3

        x_range = calc_range(data, axis=0, pelvis_index=0)
        y_range = calc_range(data, axis=1, pelvis_index=0)

        assert abs(x_range - 10.0) < 0.01
        assert abs(y_range - 5.0) < 0.01

    def test_zero_range(self):
        """Test zero range for constant values."""
        from scripts.utils.plotting import calc_range

        data = np.ones((100, 5, 3)) * 5.0

        x_range = calc_range(data, axis=0, pelvis_index=0)

        assert x_range == 0.0

    def test_different_pelvis_index(self):
        """Test range with different pelvis index."""
        from scripts.utils.plotting import calc_range

        data = np.zeros((100, 5, 3))
        data[:, 0, 1] = np.linspace(0, 5, 100)
        data[:, 2, 1] = np.linspace(0, 10, 100)

        y_range_pelvis0 = calc_range(data, axis=1, pelvis_index=0)
        y_range_pelvis2 = calc_range(data, axis=1, pelvis_index=2)

        assert abs(y_range_pelvis0 - 5.0) < 0.01
        assert abs(y_range_pelvis2 - 10.0) < 0.01


class TestCreateFigureGrid:
    """Test create_figure_grid function."""

    def test_basic_grid(self):
        """Test basic grid creation."""
        from scripts.utils.plotting import create_figure_grid

        fig, axes = create_figure_grid(6, n_cols=3)

        assert len(axes) == 6

    def test_single_plot(self):
        """Test single plot creation."""
        from scripts.utils.plotting import create_figure_grid

        fig, axes = create_figure_grid(1, n_cols=1)

        assert len(axes) == 1

    def test_uneven_grid(self):
        """Test uneven grid (more spots than plots)."""
        from scripts.utils.plotting import create_figure_grid

        fig, axes = create_figure_grid(5, n_cols=3)

        # Should create 2 rows x 3 cols = 6 spots
        assert len(axes) == 6


class TestPlotTrajectoryComparison:
    """Test plot_trajectory_comparison function."""

    def create_test_data(self, n_frames=300, n_segments=5):
        """Create test trajectory data."""
        data = np.zeros((n_frames, n_segments, 3))
        t = np.arange(n_frames)

        # X: forward movement
        data[:, 0, 0] = t * 0.01

        # Y: oscillation (walking pattern)
        data[:, 0, 1] = 0.05 * np.sin(2 * np.pi * t / 30)

        return data

    def test_basic_comparison(self):
        """Test basic comparison plot."""
        from scripts.utils.plotting import plot_trajectory_comparison

        original = self.create_test_data()
        corrected = self.create_test_data()
        corrected[:, 0, 1] *= 0.5  # Reduce Y oscillation

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_comparison.png"
            plot_trajectory_comparison(original, corrected, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_custom_titles(self):
        """Test with custom titles."""
        from scripts.utils.plotting import plot_trajectory_comparison

        original = self.create_test_data()
        corrected = self.create_test_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_titles.png"
            plot_trajectory_comparison(
                original, corrected, output_path,
                titles=["Before", "After"],
                suptitle="Test Comparison",
            )

            assert output_path.exists()

    def test_custom_config(self):
        """Test with custom PlotConfig."""
        from scripts.utils.plotting import plot_trajectory_comparison, PlotConfig

        original = self.create_test_data()
        corrected = self.create_test_data()

        config = PlotConfig(
            dpi=100,
            unit_scale=1000,
            unit_label="mm",
            show_colorbar=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_config.png"
            plot_trajectory_comparison(
                original, corrected, output_path,
                config=config,
            )

            assert output_path.exists()


class TestPlotMultiMethodComparison:
    """Test plot_multi_method_comparison function."""

    def create_test_data(self, n_frames=300, n_segments=5):
        """Create test trajectory data."""
        data = np.zeros((n_frames, n_segments, 3))
        t = np.arange(n_frames)
        data[:, 0, 0] = t * 0.01
        data[:, 0, 1] = 0.05 * np.sin(2 * np.pi * t / 30)
        return data

    def test_multi_method_plot(self):
        """Test multi-method comparison plot."""
        from scripts.utils.plotting import plot_multi_method_comparison

        original = self.create_test_data()
        results = [
            {"name": "Method A", "data": self.create_test_data()},
            {"name": "Method B", "data": self.create_test_data()},
            {"name": "Method C", "data": self.create_test_data()},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_multi.png"
            plot_multi_method_comparison(results, original, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_without_original(self):
        """Test without including original."""
        from scripts.utils.plotting import plot_multi_method_comparison

        original = self.create_test_data()
        results = [
            {"name": "Method A", "data": self.create_test_data()},
            {"name": "Method B", "data": self.create_test_data()},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_no_orig.png"
            plot_multi_method_comparison(
                results, original, output_path,
                include_original=False,
            )

            assert output_path.exists()


class TestPlotTimeSeriesComparison:
    """Test plot_time_series_comparison function."""

    def create_test_data(self, n_frames=300, n_segments=5):
        """Create test trajectory data."""
        data = np.zeros((n_frames, n_segments, 3))
        t = np.arange(n_frames)
        data[:, 0, 0] = t * 0.01
        data[:, 0, 1] = 0.05 * np.sin(2 * np.pi * t / 30) + np.random.randn(n_frames) * 0.01
        return data

    def test_time_series_plot(self):
        """Test time series comparison plot."""
        from scripts.utils.plotting import plot_time_series_comparison

        datasets = [
            {"name": "Original", "data": self.create_test_data(), "color": "blue"},
            {"name": "Corrected", "data": self.create_test_data(), "color": "red"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_timeseries.png"
            plot_time_series_comparison(datasets, output_path)

            assert output_path.exists()

    def test_with_zoom(self):
        """Test time series with zoomed view."""
        from scripts.utils.plotting import plot_time_series_comparison

        datasets = [
            {"name": "Original", "data": self.create_test_data()},
            {"name": "Corrected", "data": self.create_test_data()},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_zoom.png"
            plot_time_series_comparison(
                datasets, output_path,
                zoom_seconds=(2.0, 4.0),
            )

            assert output_path.exists()


class TestPlotSegmentDistribution:
    """Test plot_segment_distribution function."""

    def test_basic_distribution(self):
        """Test basic segment distribution plot."""
        from scripts.utils.plotting import plot_segment_distribution

        # Create mock segments with duration_sec attribute
        class MockSegment:
            def __init__(self, duration):
                self.duration_sec = duration

        segments_list = [
            {
                "name": "Method A",
                "segments": [MockSegment(d) for d in [3.0, 4.5, 5.0, 3.5, 6.0]],
            },
            {
                "name": "Method B",
                "segments": [MockSegment(d) for d in [2.5, 3.0, 4.0, 5.5]],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dist.png"
            plot_segment_distribution(segments_list, output_path)

            assert output_path.exists()

    def test_empty_segments(self):
        """Test with empty segments."""
        from scripts.utils.plotting import plot_segment_distribution

        segments_list = [
            {"name": "Empty", "segments": []},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_empty.png"
            plot_segment_distribution(segments_list, output_path)

            assert output_path.exists()


class TestPrintSummaryTable:
    """Test print_summary_table function."""

    def test_basic_table(self, capsys):
        """Test basic summary table output."""
        from scripts.utils.plotting import print_summary_table

        results = [
            {"name": "Method A", "y_range_corrected": 2.5},
            {"name": "Method B", "y_range_corrected": 1.8},
            {"name": "Method C", "y_range_corrected": 3.0},
        ]

        print_summary_table(results, original_y_range=5.0)

        captured = capsys.readouterr()
        assert "SUMMARY TABLE" in captured.out
        assert "Method A" in captured.out
        assert "Method B" in captured.out
        assert "Best method: Method B" in captured.out

    def test_with_baseline(self, capsys):
        """Test summary table with baseline."""
        from scripts.utils.plotting import print_summary_table

        results = [
            {"name": "Advanced", "y_range_corrected": 1.5},
        ]

        print_summary_table(results, original_y_range=5.0, baseline_y_range=3.0)

        captured = capsys.readouterr()
        assert "Baseline" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
