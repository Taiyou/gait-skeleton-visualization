"""
Gait Analysis Module

This module provides tools for preprocessing IMU gait data and extracting
straight-line walking segments for LSTM analysis.

Main API:
---------
    from scripts.gait_analysis import (
        # High-level pipeline (recommended)
        extract_segments_from_file,
        extract_segments_from_data,
        batch_extract_segments,
        SegmentExtractionConfig,

        # Low-level components
        apply_feature_preserving_correction,
        extract_segments_improved,
        SegmentExtractionParams,
        ExtractedSegment,
    )

Example:
--------
    # Simple usage
    from scripts.gait_analysis import extract_segments_from_file

    segments, info = extract_segments_from_file("path/to/data.xlsx")
    print(f"Extracted {len(segments)} segments")

    # With custom configuration
    from scripts.gait_analysis import extract_segments_from_file, SegmentExtractionConfig

    config = SegmentExtractionConfig(
        min_segment_meters=7.0,
        use_preprocessing=True
    )
    segments, info = extract_segments_from_file("path/to/data.xlsx", config=config)
"""

# High-level pipeline (recommended)
from scripts.gait_analysis.segment_extraction_pipeline import (
    extract_segments_from_file,
    extract_segments_from_data,
    batch_extract_segments,
    get_segments_as_array,
    quick_extract,
    SegmentExtractionConfig,
    PipelineResult,
)

# Low-level components
from scripts.gait_analysis.feature_preserving_correction import (
    apply_feature_preserving_correction,
    CorrectionResult,
)

from scripts.gait_analysis.improved_segment_extraction import (
    extract_segments_improved,
    SegmentExtractionParams,
    ExtractedSegment,
)

__all__ = [
    # High-level API
    'extract_segments_from_file',
    'extract_segments_from_data',
    'batch_extract_segments',
    'get_segments_as_array',
    'quick_extract',
    'SegmentExtractionConfig',
    'PipelineResult',

    # Low-level components
    'apply_feature_preserving_correction',
    'CorrectionResult',
    'extract_segments_improved',
    'SegmentExtractionParams',
    'ExtractedSegment',
]
