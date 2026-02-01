#!/usr/bin/env python3
"""
Tests for feature preserving correction module.

Run with: pytest tests/test_feature_preserving_correction.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_analysis.feature_preserving_correction import (
    apply_feature_preserving_correction,
    CorrectionResult,
)


class TestFeaturePreservingCorrection:
    """Test feature preserving correction functions."""

    def create_synthetic_data_with_drift(
        self,
        n_frames: int = 600,
        n_joints: int = 23,
        frame_rate: int = 60,
        drift_amount: float = 5.0
    ) -> np.ndarray:
        """Create synthetic walking data with Y-axis drift."""
        data = np.zeros((n_frames, n_joints, 3))
        time = np.arange(n_frames) / frame_rate

        # Pelvis (joint 0) walks in X direction with Y drift
        data[:, 0, 0] = 1.2 * time  # X: forward movement
        data[:, 0, 1] = drift_amount * time / (n_frames / frame_rate)  # Y: drift
        data[:, 0, 1] += 0.02 * np.sin(2 * np.pi * 1.8 * time)  # Y: gait oscillation

        # Other joints follow pelvis
        for j in range(1, n_joints):
            data[:, j, :] = data[:, 0, :] + np.random.randn(3) * 0.05

        return data

    def test_correction_reduces_y_drift(self):
        """Test that correction reduces Y-axis drift."""
        data = self.create_synthetic_data_with_drift(drift_amount=5.0)
        original_y_range = data[:, 0, 1].max() - data[:, 0, 1].min()

        result = apply_feature_preserving_correction(
            data,
            frame_rate=60,
            drift_correction_strength='moderate'
        )

        corrected_y_range = result.data[:, 0, 1].max() - result.data[:, 0, 1].min()

        # Corrected Y range should be significantly smaller
        assert corrected_y_range < original_y_range * 0.5

    def test_correction_preserves_x_movement(self):
        """Test that correction preserves forward movement."""
        data = self.create_synthetic_data_with_drift()
        original_x_range = data[:, 0, 0].max() - data[:, 0, 0].min()

        result = apply_feature_preserving_correction(data, frame_rate=60)

        corrected_x_range = result.data[:, 0, 0].max() - result.data[:, 0, 0].min()

        # X range should be roughly preserved (within 20%)
        assert abs(corrected_x_range - original_x_range) / original_x_range < 0.2

    def test_correction_result_structure(self):
        """Test that CorrectionResult contains expected fields."""
        data = self.create_synthetic_data_with_drift()

        result = apply_feature_preserving_correction(data, frame_rate=60)

        assert isinstance(result, CorrectionResult)
        assert hasattr(result, 'data')
        assert hasattr(result, 'info')
        assert result.data.shape == data.shape

    def test_correction_info_contents(self):
        """Test that correction info contains expected keys."""
        data = self.create_synthetic_data_with_drift()

        result = apply_feature_preserving_correction(data, frame_rate=60)

        # Check info dict
        assert 'rotation_angle_deg' in result.info
        assert 'original_y_range' in result.info
        assert 'corrected_y_range' in result.info
        assert 'y_range_reduction_pct' in result.info

    def test_strength_minimal(self):
        """Test minimal correction strength."""
        data = self.create_synthetic_data_with_drift(drift_amount=5.0)

        result = apply_feature_preserving_correction(
            data,
            frame_rate=60,
            drift_correction_strength='minimal'
        )

        # Minimal should still reduce drift but less aggressively
        original_y_range = data[:, 0, 1].max() - data[:, 0, 1].min()
        corrected_y_range = result.data[:, 0, 1].max() - result.data[:, 0, 1].min()

        assert corrected_y_range < original_y_range

    def test_strength_aggressive(self):
        """Test aggressive correction strength."""
        data = self.create_synthetic_data_with_drift(drift_amount=5.0)

        result = apply_feature_preserving_correction(
            data,
            frame_rate=60,
            drift_correction_strength='aggressive'
        )

        corrected_y_range = result.data[:, 0, 1].max() - result.data[:, 0, 1].min()

        # Aggressive should reduce Y range significantly
        assert corrected_y_range < 2.0  # Should be quite small

    def test_data_shape_preserved(self):
        """Test that data shape is preserved after correction."""
        shapes = [
            (300, 23, 3),
            (600, 23, 3),
            (1000, 23, 3),
        ]

        for shape in shapes:
            data = np.random.randn(*shape)
            # Add some structure to make it valid walking data
            data[:, 0, 0] = np.linspace(0, 10, shape[0])

            result = apply_feature_preserving_correction(data, frame_rate=60)

            assert result.data.shape == shape

    def test_no_nan_values(self):
        """Test that correction doesn't produce NaN values."""
        data = self.create_synthetic_data_with_drift()

        result = apply_feature_preserving_correction(data, frame_rate=60)

        assert not np.any(np.isnan(result.data))

    def test_no_inf_values(self):
        """Test that correction doesn't produce infinite values."""
        data = self.create_synthetic_data_with_drift()

        result = apply_feature_preserving_correction(data, frame_rate=60)

        assert not np.any(np.isinf(result.data))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_data(self):
        """Test with very short data sequence."""
        data = np.random.randn(30, 23, 3)  # 0.5 seconds at 60fps
        data[:, 0, 0] = np.linspace(0, 0.6, 30)  # Some forward movement

        # Should not crash
        result = apply_feature_preserving_correction(data, frame_rate=60)
        assert result.data.shape == data.shape

    def test_stationary_data(self):
        """Test with stationary (no movement) data."""
        data = np.random.randn(300, 23, 3) * 0.01  # Very small random noise

        # Should not crash
        result = apply_feature_preserving_correction(data, frame_rate=60)
        assert result.data.shape == data.shape

    def test_different_frame_rates(self):
        """Test with different frame rates."""
        frame_rates = [30, 60, 100, 120]

        for fr in frame_rates:
            n_frames = fr * 10  # 10 seconds
            data = np.zeros((n_frames, 23, 3))
            data[:, 0, 0] = np.linspace(0, 12, n_frames)  # Walk forward

            result = apply_feature_preserving_correction(data, frame_rate=fr)
            assert result.data.shape == data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
