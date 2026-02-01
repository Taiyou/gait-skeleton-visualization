#!/usr/bin/env python3
"""
Tests for segment extraction pipeline.

Run with: pytest tests/test_segment_extraction.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_analysis.improved_segment_extraction import (
    SegmentExtractionParams,
    ExtractedSegment,
    extract_segments_improved,
    compute_heading,
    compute_heading_change_rate,
    detect_straight_regions,
    find_continuous_regions,
)


class TestHeadingComputation:
    """Test heading computation functions."""

    def test_compute_heading_positive_x(self):
        """Test heading for movement in positive X direction."""
        velocity = np.array([[1, 0], [1, 0], [1, 0]])
        heading = compute_heading(velocity)
        np.testing.assert_array_almost_equal(heading, [0, 0, 0])

    def test_compute_heading_positive_y(self):
        """Test heading for movement in positive Y direction."""
        velocity = np.array([[0, 1], [0, 1], [0, 1]])
        heading = compute_heading(velocity)
        expected = np.array([np.pi/2, np.pi/2, np.pi/2])
        np.testing.assert_array_almost_equal(heading, expected)

    def test_compute_heading_diagonal(self):
        """Test heading for diagonal movement."""
        velocity = np.array([[1, 1], [1, 1], [1, 1]])
        heading = compute_heading(velocity)
        expected = np.array([np.pi/4, np.pi/4, np.pi/4])
        np.testing.assert_array_almost_equal(heading, expected)

    def test_compute_heading_change_rate_constant(self):
        """Test heading change rate for constant direction."""
        heading = np.array([0, 0, 0, 0, 0])
        rate = compute_heading_change_rate(heading, frame_rate=60, smooth_window_sec=0.1)
        np.testing.assert_array_almost_equal(rate, np.zeros(5), decimal=5)

    def test_compute_heading_change_rate_turning(self):
        """Test heading change rate detects turns."""
        # Simulate a turn: heading changes from 0 to pi/2
        heading = np.linspace(0, np.pi/2, 60)  # 1 second turn at 60 fps
        rate = compute_heading_change_rate(heading, frame_rate=60, smooth_window_sec=0.1)
        # Rate should be positive during turn
        assert np.mean(rate) > 0


class TestStraightRegionDetection:
    """Test straight region detection."""

    def test_detect_straight_walking(self):
        """Test detection of straight walking regions."""
        n_frames = 300  # 5 seconds at 60fps

        # Create velocity: walking straight in X direction
        velocity = np.zeros((n_frames, 2))
        velocity[:, 0] = 1.0  # 1 m/s in X direction

        params = SegmentExtractionParams(
            velocity_threshold=0.4,
            heading_change_threshold=0.1,
            frame_rate=60
        )

        straight_mask, velocity_mag, heading_rate = detect_straight_regions(velocity, params)

        # Should be mostly straight
        assert np.mean(straight_mask) > 0.9

    def test_detect_turn_regions(self):
        """Test that turns are not marked as straight."""
        n_frames = 300
        velocity = np.zeros((n_frames, 2))

        # First half: walking in X direction
        velocity[:150, 0] = 1.0

        # Second half: walking in Y direction (90 degree turn)
        velocity[150:, 1] = 1.0

        params = SegmentExtractionParams(
            velocity_threshold=0.4,
            heading_change_threshold=0.1,
            heading_smooth_window=0.1,  # Smaller window to detect sharp turn
            frame_rate=60
        )

        straight_mask, _, heading_rate = detect_straight_regions(velocity, params)

        # Check that heading rate spikes at turn point
        # The turn should cause high heading change rate around frame 150
        turn_heading_rate = heading_rate[145:155]
        assert np.max(turn_heading_rate) > params.heading_change_threshold

    def test_detect_standing_still(self):
        """Test that standing still is not marked as walking."""
        n_frames = 300
        velocity = np.zeros((n_frames, 2))
        velocity[:, 0] = 0.1  # Below threshold

        params = SegmentExtractionParams(
            velocity_threshold=0.4,
            heading_change_threshold=0.1,
            frame_rate=60
        )

        straight_mask, _, _ = detect_straight_regions(velocity, params)

        # Should not be marked as walking
        assert np.sum(straight_mask) == 0


class TestContinuousRegions:
    """Test continuous region finding."""

    def test_find_single_region(self):
        """Test finding a single continuous region."""
        mask = np.array([False, True, True, True, False])
        regions = find_continuous_regions(mask)
        assert len(regions) == 1
        assert regions[0] == (1, 4)

    def test_find_multiple_regions(self):
        """Test finding multiple continuous regions."""
        mask = np.array([True, True, False, False, True, True, True, False])
        regions = find_continuous_regions(mask)
        assert len(regions) == 2
        assert regions[0] == (0, 2)
        assert regions[1] == (4, 7)

    def test_find_no_regions(self):
        """Test when no regions exist."""
        mask = np.array([False, False, False, False])
        regions = find_continuous_regions(mask)
        assert len(regions) == 0

    def test_find_all_true(self):
        """Test when all values are True."""
        mask = np.array([True, True, True, True])
        regions = find_continuous_regions(mask)
        assert len(regions) == 1
        assert regions[0] == (0, 4)


class TestSegmentExtraction:
    """Test full segment extraction pipeline."""

    def create_synthetic_walking_data(
        self,
        n_frames: int = 600,
        n_joints: int = 23,
        frame_rate: int = 60,
        walking_speed: float = 1.2
    ) -> tuple:
        """Create synthetic walking data for testing."""
        # Create position data: walking in X direction
        data = np.zeros((n_frames, n_joints, 3))

        # Pelvis (joint 0) moves forward
        time = np.arange(n_frames) / frame_rate
        data[:, 0, 0] = walking_speed * time  # X position

        # Add small Y oscillation (gait pattern)
        data[:, 0, 1] = 0.02 * np.sin(2 * np.pi * 1.8 * time)  # ~1.8 Hz gait rhythm

        # Other joints follow pelvis with small offsets
        for j in range(1, n_joints):
            data[:, j, 0] = data[:, 0, 0]
            data[:, j, 1] = data[:, 0, 1] + np.random.randn() * 0.01
            data[:, j, 2] = np.random.randn() * 0.1  # Random Z offset

        # Compute velocity
        pelvis_pos = data[:, 0, :2]
        velocity = np.gradient(pelvis_pos, 1/frame_rate, axis=0)

        return data, velocity

    def test_extract_basic_segments(self):
        """Test basic segment extraction."""
        data, velocity = self.create_synthetic_walking_data(n_frames=600)

        params = SegmentExtractionParams(
            velocity_threshold=0.4,
            heading_change_threshold=0.1,
            trim_start_seconds=0.5,
            trim_end_seconds=0.3,
            min_segment_seconds=2.0,
            min_segment_meters=5.0,
            use_overlapping_windows=False,
            frame_rate=60
        )

        segments, info = extract_segments_improved(data, velocity, params)

        # Should extract at least one segment
        assert len(segments) >= 1

        # Check segment properties
        for seg in segments:
            assert seg.duration_sec >= params.min_segment_seconds
            assert seg.distance_m >= params.min_segment_meters
            assert seg.data.shape[1] == 23  # n_joints
            assert seg.data.shape[2] == 3   # xyz

    def test_extract_with_overlapping_windows(self):
        """Test segment extraction with overlapping windows."""
        data, velocity = self.create_synthetic_walking_data(n_frames=900)  # 15 seconds

        params = SegmentExtractionParams(
            velocity_threshold=0.4,
            heading_change_threshold=0.1,
            use_overlapping_windows=True,
            window_seconds=5.0,
            window_overlap=0.5,
            min_segment_meters=5.0,
            frame_rate=60
        )

        segments, info = extract_segments_improved(data, velocity, params)

        # With overlapping windows, should get more segments
        assert len(segments) >= 2

        # Check window segments
        window_segments = [s for s in segments if s.is_window]
        if window_segments:
            # All windows should have same duration
            expected_frames = int(params.window_seconds * params.frame_rate)
            for seg in window_segments:
                assert seg.data.shape[0] == expected_frames

    def test_info_dict_contents(self):
        """Test that info dict contains expected keys."""
        data, velocity = self.create_synthetic_walking_data()

        params = SegmentExtractionParams(frame_rate=60)
        segments, info = extract_segments_improved(data, velocity, params)

        # Check required keys
        assert 'total_frames' in info
        assert 'straight_frames' in info
        assert 'straight_ratio' in info
        assert 'raw_regions' in info
        assert 'total_segments' in info

        # Check values are reasonable
        assert info['total_frames'] == len(data)
        assert 0 <= info['straight_ratio'] <= 1

    def test_minimum_segment_distance_filter(self):
        """Test that short segments are filtered out."""
        # Create short walking data (2 seconds = ~2.4m at 1.2 m/s)
        data, velocity = self.create_synthetic_walking_data(
            n_frames=120,
            walking_speed=1.2
        )

        params = SegmentExtractionParams(
            min_segment_meters=5.0,  # Require 5m minimum
            use_overlapping_windows=False,
            frame_rate=60
        )

        segments, info = extract_segments_improved(data, velocity, params)

        # Should be filtered out (only ~2.4m)
        assert len(segments) == 0


class TestExtractedSegment:
    """Test ExtractedSegment dataclass."""

    def test_segment_properties(self):
        """Test segment property access."""
        data = np.random.randn(300, 23, 3)

        segment = ExtractedSegment(
            data=data,
            start_frame=100,
            end_frame=400,
            duration_sec=5.0,
            distance_m=6.0,
            mean_velocity=1.2,
            is_window=False,
            parent_segment_id=0
        )

        assert segment.data.shape == (300, 23, 3)
        assert segment.start_frame == 100
        assert segment.end_frame == 400
        assert segment.duration_sec == 5.0
        assert segment.distance_m == 6.0
        assert segment.mean_velocity == 1.2
        assert segment.is_window == False


class TestSegmentExtractionParams:
    """Test parameter configuration."""

    def test_default_params(self):
        """Test default parameter values."""
        params = SegmentExtractionParams()

        assert params.velocity_threshold == 0.4
        assert params.heading_change_threshold == 0.1
        assert params.frame_rate == 60
        assert params.use_overlapping_windows == True

    def test_custom_params(self):
        """Test custom parameter values."""
        params = SegmentExtractionParams(
            velocity_threshold=0.5,
            heading_change_threshold=0.2,
            min_segment_meters=7.0,
            frame_rate=100
        )

        assert params.velocity_threshold == 0.5
        assert params.heading_change_threshold == 0.2
        assert params.min_segment_meters == 7.0
        assert params.frame_rate == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
