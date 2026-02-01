#!/usr/bin/env python3
"""
Tests for the high-level segment extraction pipeline API.

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_analysis.segment_extraction_pipeline import (
    SegmentExtractionConfig,
    PipelineResult,
    extract_segments_from_data,
    get_segments_as_array,
    compute_velocity,
)


class TestSegmentExtractionConfig:
    """Test configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SegmentExtractionConfig()

        assert config.frame_rate == 60
        assert config.use_preprocessing == True
        assert config.drift_correction_strength == 'moderate'
        assert config.velocity_threshold == 0.4
        assert config.heading_change_threshold == 0.1
        assert config.min_segment_meters == 5.0
        assert config.use_overlapping_windows == True
        assert config.window_seconds == 5.0
        assert config.window_overlap == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = SegmentExtractionConfig(
            frame_rate=100,
            use_preprocessing=False,
            min_segment_meters=7.0,
            window_seconds=3.0
        )

        assert config.frame_rate == 100
        assert config.use_preprocessing == False
        assert config.min_segment_meters == 7.0
        assert config.window_seconds == 3.0

    def test_to_extraction_params(self):
        """Test conversion to extraction params."""
        config = SegmentExtractionConfig(
            velocity_threshold=0.5,
            heading_change_threshold=0.2,
            frame_rate=100
        )

        params = config.to_extraction_params()

        assert params.velocity_threshold == 0.5
        assert params.heading_change_threshold == 0.2
        assert params.frame_rate == 100


class TestComputeVelocity:
    """Test velocity computation."""

    def test_stationary(self):
        """Test velocity for stationary data."""
        data = np.zeros((100, 23, 3))
        velocity = compute_velocity(data, frame_rate=60)

        assert velocity.shape == (100, 2)
        np.testing.assert_array_almost_equal(velocity, np.zeros((100, 2)))

    def test_constant_velocity(self):
        """Test velocity for constant speed movement."""
        data = np.zeros((100, 23, 3))
        # Move at 1 m/s in X direction
        data[:, 0, 0] = np.linspace(0, 100/60, 100)

        velocity = compute_velocity(data, frame_rate=60)

        # Velocity should be approximately 1 m/s in X
        assert np.abs(np.mean(velocity[:, 0]) - 1.0) < 0.1

    def test_output_shape(self):
        """Test output shape for various input sizes."""
        for n_frames in [60, 300, 600]:
            data = np.random.randn(n_frames, 23, 3)
            velocity = compute_velocity(data, frame_rate=60)

            assert velocity.shape == (n_frames, 2)


class TestExtractSegmentsFromData:
    """Test extraction from numpy array."""

    def create_walking_data(self, n_frames=600, speed=1.2):
        """Create synthetic walking data."""
        data = np.zeros((n_frames, 23, 3))
        time = np.arange(n_frames) / 60

        # Forward movement
        data[:, 0, 0] = speed * time

        # Small Y oscillation
        data[:, 0, 1] = 0.02 * np.sin(2 * np.pi * 1.8 * time)

        # Other joints follow
        for j in range(1, 23):
            data[:, j, :] = data[:, 0, :] + np.random.randn(3) * 0.02

        return data

    def test_basic_extraction(self):
        """Test basic segment extraction."""
        data = self.create_walking_data(n_frames=600)

        segments, info = extract_segments_from_data(data)

        assert len(segments) >= 1
        assert 'n_segments' in info
        assert info['n_segments'] == len(segments)

    def test_extraction_with_preprocessing(self):
        """Test extraction with preprocessing enabled."""
        data = self.create_walking_data()

        config = SegmentExtractionConfig(use_preprocessing=True)
        segments, info = extract_segments_from_data(data, config=config)

        assert 'preprocessing' in info

    def test_extraction_without_preprocessing(self):
        """Test extraction without preprocessing."""
        data = self.create_walking_data()

        config = SegmentExtractionConfig(use_preprocessing=False)
        segments, info = extract_segments_from_data(data, config=config)

        assert info['preprocessing'] == {'preprocessing': 'none'}

    def test_return_preprocessed_data(self):
        """Test returning preprocessed data."""
        data = self.create_walking_data()

        result = extract_segments_from_data(
            data,
            return_preprocessed=True
        )

        assert isinstance(result, PipelineResult)
        assert result.raw_data is not None
        assert result.preprocessed_data is not None
        assert result.raw_data.shape == data.shape

    def test_custom_min_distance(self):
        """Test custom minimum distance filter."""
        data = self.create_walking_data(n_frames=300)  # ~6m at 1.2 m/s

        # With 5m minimum, should get segments
        config1 = SegmentExtractionConfig(min_segment_meters=5.0)
        segments1, _ = extract_segments_from_data(data, config=config1)

        # With 10m minimum, should get fewer or no segments
        config2 = SegmentExtractionConfig(min_segment_meters=10.0)
        segments2, _ = extract_segments_from_data(data, config=config2)

        assert len(segments2) <= len(segments1)


class TestGetSegmentsAsArray:
    """Test conversion of segments to array."""

    def create_mock_segments(self, n_segments=5, n_frames=300):
        """Create mock segments for testing."""
        from scripts.gait_analysis.improved_segment_extraction import ExtractedSegment

        segments = []
        for i in range(n_segments):
            # Vary the length slightly
            length = n_frames + np.random.randint(-50, 50)
            data = np.random.randn(length, 23, 3)

            segments.append(ExtractedSegment(
                data=data,
                start_frame=i * 100,
                end_frame=i * 100 + length,
                duration_sec=length / 60,
                distance_m=length / 60 * 1.2,
                mean_velocity=1.2,
                is_window=False,
                parent_segment_id=i
            ))

        return segments

    def test_basic_conversion(self):
        """Test basic array conversion."""
        segments = self.create_mock_segments(n_segments=5, n_frames=300)

        array = get_segments_as_array(segments)

        assert array.ndim == 4
        assert array.shape[0] == 5  # n_segments
        assert array.shape[2] == 23  # n_joints
        assert array.shape[3] == 3  # xyz

    def test_pad_to_length(self):
        """Test padding to specific length."""
        segments = self.create_mock_segments(n_segments=3, n_frames=200)

        array = get_segments_as_array(segments, pad_to_length=500)

        assert array.shape[1] == 500

    def test_truncate_to_length(self):
        """Test truncation to specific length."""
        segments = self.create_mock_segments(n_segments=3, n_frames=500)

        array = get_segments_as_array(segments, pad_to_length=200)

        assert array.shape[1] == 200


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_result_structure(self):
        """Test result structure."""
        from scripts.gait_analysis.improved_segment_extraction import ExtractedSegment

        segments = [
            ExtractedSegment(
                data=np.random.randn(300, 23, 3),
                start_frame=0,
                end_frame=300,
                duration_sec=5.0,
                distance_m=6.0,
                mean_velocity=1.2
            )
        ]

        result = PipelineResult(
            segments=segments,
            info={'n_segments': 1},
            config=SegmentExtractionConfig(),
            raw_data=np.random.randn(600, 23, 3),
            preprocessed_data=np.random.randn(600, 23, 3)
        )

        assert len(result.segments) == 1
        assert result.info['n_segments'] == 1
        assert isinstance(result.config, SegmentExtractionConfig)
        assert result.raw_data is not None
        assert result.preprocessed_data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
