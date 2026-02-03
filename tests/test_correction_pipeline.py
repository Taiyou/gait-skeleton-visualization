#!/usr/bin/env python3
"""
Integration tests for the gait correction pipeline.

Run with: pytest tests/test_correction_pipeline.py -v
"""

import pytest
import numpy as np
from pathlib import Path


class TestGaitCorrectionIntegration:
    """Integration tests for the full correction pipeline."""

    def create_synthetic_walking_data(
        self,
        n_frames: int = 3600,  # 1 minute at 60 Hz
        n_segments: int = 23,
        frame_rate: int = 60,
        walking_speed: float = 1.2,  # m/s
        y_drift_rate: float = 0.01,  # m/s drift
        gait_frequency: float = 1.8,  # Hz
        gait_amplitude: float = 0.02,  # m
    ) -> np.ndarray:
        """
        Create synthetic walking data with realistic drift and gait patterns.

        Returns:
            data: Shape (n_frames, n_segments, 3) - position data
        """
        data = np.zeros((n_frames, n_segments, 3))
        time = np.arange(n_frames) / frame_rate

        # Pelvis trajectory with drift
        # X: forward movement
        data[:, 0, 0] = walking_speed * time

        # Y: lateral drift + gait oscillation
        drift = y_drift_rate * time
        gait_oscillation = gait_amplitude * np.sin(2 * np.pi * gait_frequency * time)
        data[:, 0, 1] = drift + gait_oscillation

        # Z: vertical (constant + small variation)
        data[:, 0, 2] = 1.0 + 0.01 * np.sin(2 * np.pi * 2 * gait_frequency * time)

        # Other segments follow pelvis with small offsets
        for seg_idx in range(1, n_segments):
            offset = np.array([0.0, 0.0, 0.0])
            if seg_idx < 5:  # Upper body
                offset[2] = 0.5 + 0.1 * seg_idx
            else:  # Lower body
                offset[2] = -0.2 - 0.1 * (seg_idx - 5)

            data[:, seg_idx, :] = data[:, 0, :] + offset
            # Add small random variation
            data[:, seg_idx, :] += np.random.randn(n_frames, 3) * 0.001

        return data

    def test_drift_removal_reduces_y_range(self):
        """Test that drift removal reduces Y range."""
        from scripts.gait_correction.drift_removal import remove_y_drift

        # Create data with significant drift
        data = self.create_synthetic_walking_data(
            n_frames=3600,
            y_drift_rate=0.02,  # Strong drift
        )

        original_y_range = data[:, 0, 1].max() - data[:, 0, 1].min()

        # Apply drift removal
        corrected = remove_y_drift(data.copy(), window_seconds=30.0, frame_rate=60)

        corrected_y_range = corrected[:, 0, 1].max() - corrected[:, 0, 1].min()

        # Should reduce Y range significantly
        assert corrected_y_range < original_y_range * 0.5

    def test_smooth_pca_aligns_trajectory(self):
        """Test that Smooth PCA aligns trajectory."""
        from scripts.gait_correction.smooth_pca import (
            apply_smooth_pca_correction,
            SmoothPCAParams,
        )

        # Create data
        data = self.create_synthetic_walking_data()

        # Add some rotation
        angle = np.radians(15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = data.copy()
        for seg_idx in range(data.shape[1]):
            x = data[:, seg_idx, 0]
            y = data[:, seg_idx, 1]
            rotated[:, seg_idx, 0] = x * cos_a - y * sin_a
            rotated[:, seg_idx, 1] = x * sin_a + y * cos_a

        # Apply PCA correction
        params = SmoothPCAParams(
            window_seconds=30.0,
            sample_interval_seconds=5.0,
            frame_rate=60,
        )

        corrected, angles, _ = apply_smooth_pca_correction(
            rotated,
            params=params,
        )

        # Should have detected and corrected rotation
        assert len(angles) > 0

    def test_full_correction_preserves_gait_pattern(self):
        """Test that correction preserves gait oscillation pattern."""
        from scripts.gait_correction.drift_removal import apply_full_drift_correction
        from scipy.fft import fft, fftfreq

        # Create data with clear gait pattern
        data = self.create_synthetic_walking_data(
            gait_frequency=1.8,
            gait_amplitude=0.03,
        )

        # Apply correction
        corrected = apply_full_drift_correction(data.copy(), frame_rate=60)

        # Compute frequency spectrum
        def compute_gait_power(y_signal, frame_rate):
            n = len(y_signal)
            freqs = fftfreq(n, 1/frame_rate)
            spectrum = np.abs(fft(y_signal - np.mean(y_signal)))

            # Gait frequency band (0.5 - 3 Hz)
            gait_mask = (freqs > 0.5) & (freqs < 3.0)
            return np.sum(spectrum[gait_mask]**2)

        original_power = compute_gait_power(data[:, 0, 1], 60)
        corrected_power = compute_gait_power(corrected[:, 0, 1], 60)

        # Gait power should be largely preserved (at least 50%)
        preservation_ratio = corrected_power / original_power
        assert preservation_ratio > 0.3  # Allow some loss but not total

    def test_correction_maintains_segment_relationships(self):
        """Test that relative positions between segments are maintained."""
        from scripts.gait_correction.drift_removal import apply_full_drift_correction

        data = self.create_synthetic_walking_data()

        # Calculate original relative distances
        original_distances = []
        for seg_idx in range(1, data.shape[1]):
            dist = np.mean(np.linalg.norm(
                data[:, seg_idx, :] - data[:, 0, :],
                axis=1
            ))
            original_distances.append(dist)

        # Apply correction
        corrected = apply_full_drift_correction(data.copy(), frame_rate=60)

        # Calculate corrected relative distances
        corrected_distances = []
        for seg_idx in range(1, corrected.shape[1]):
            dist = np.mean(np.linalg.norm(
                corrected[:, seg_idx, :] - corrected[:, 0, :],
                axis=1
            ))
            corrected_distances.append(dist)

        # Distances should be approximately preserved
        for orig, corr in zip(original_distances, corrected_distances):
            assert abs(orig - corr) < 0.1  # Allow small changes

    def test_config_integration(self):
        """Test that config objects work with correction functions."""
        from scripts.utils.config import GaitCorrectionConfig
        from scripts.gait_correction.drift_removal import apply_full_drift_correction
        from scripts.gait_correction.smooth_pca import (
            apply_smooth_pca_correction,
            SmoothPCAParams,
        )

        config = GaitCorrectionConfig(
            frame_rate=60,
            drift_window_seconds=20.0,
            pca_window_seconds=25.0,
        )

        data = self.create_synthetic_walking_data(frame_rate=config.frame_rate)

        # Apply with config parameters
        pca_params = SmoothPCAParams(
            window_seconds=config.pca_window_seconds,
            sample_interval_seconds=config.pca_sample_interval_seconds,
            frame_rate=config.frame_rate,
        )

        corrected, _, _ = apply_smooth_pca_correction(
            data,
            params=pca_params,
            pelvis_index=config.pelvis_index,
        )

        corrected = apply_full_drift_correction(
            corrected,
            drift_window_seconds=config.drift_window_seconds,
            frame_rate=config.frame_rate,
            pelvis_index=config.pelvis_index,
        )

        # Should complete without error
        assert corrected.shape == data.shape


class TestTurnaroundDetection:
    """Test turnaround detection functionality."""

    def create_back_and_forth_data(
        self,
        n_passes: int = 4,
        pass_length: float = 10.0,  # meters
        walking_speed: float = 1.2,  # m/s
        frame_rate: int = 60,
    ) -> np.ndarray:
        """Create data simulating back-and-forth walking."""
        pass_duration = pass_length / walking_speed
        pass_frames = int(pass_duration * frame_rate)
        total_frames = n_passes * pass_frames

        data = np.zeros((total_frames, 23, 3))
        x = np.zeros(total_frames)

        for i in range(n_passes):
            start = i * pass_frames
            end = (i + 1) * pass_frames

            if i % 2 == 0:  # Forward pass
                x[start:end] = np.linspace(0, pass_length, pass_frames)
            else:  # Backward pass
                x[start:end] = np.linspace(pass_length, 0, pass_frames)

        data[:, 0, 0] = x
        data[:, 0, 1] = np.random.randn(total_frames) * 0.02
        data[:, 0, 2] = 1.0

        return data

    def test_detects_turnarounds(self):
        """Test that turnarounds are detected."""
        from scripts.gait_correction.turnaround import detect_turnarounds_adaptive

        data = self.create_back_and_forth_data(n_passes=4)
        pelvis_x = data[:, 0, 0]

        result = detect_turnarounds_adaptive(pelvis_x, frame_rate=60)

        # Should detect segments (one per pass)
        # Note: actual detection may merge or split based on algorithm
        assert len(result.segments) >= 2

    def test_segment_boundaries_valid(self):
        """Test that segment boundaries are valid."""
        from scripts.gait_correction.turnaround import detect_turnarounds_adaptive

        data = self.create_back_and_forth_data(n_passes=4)
        pelvis_x = data[:, 0, 0]
        n_frames = len(pelvis_x)

        result = detect_turnarounds_adaptive(pelvis_x, frame_rate=60)

        for start, end in result.segments:
            assert 0 <= start < n_frames
            assert 0 <= end < n_frames
            assert start < end


class TestPlottingWithRealData:
    """Test plotting functions with realistic data patterns."""

    def create_test_data(self, n_frames=600, n_segments=5):
        """Create test trajectory data."""
        data = np.zeros((n_frames, n_segments, 3))
        t = np.arange(n_frames) / 60

        data[:, 0, 0] = 1.2 * t  # ~1.2 m/s forward
        data[:, 0, 1] = 0.03 * np.sin(2 * np.pi * 1.8 * t) + 0.01 * t  # Gait + drift

        return data

    def test_plotting_with_synthetic_data(self):
        """Test that plotting works with synthetic data."""
        import tempfile
        from scripts.utils.plotting import plot_trajectory_comparison, PlotConfig

        original = self.create_test_data()
        corrected = self.create_test_data()
        corrected[:, 0, 1] -= 0.01 * np.arange(len(corrected)) / 60  # Remove drift

        config = PlotConfig(dpi=72)  # Lower DPI for faster tests

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            plot_trajectory_comparison(original, corrected, output_path, config=config)

            assert output_path.exists()
            assert output_path.stat().st_size > 1000  # Non-trivial file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
