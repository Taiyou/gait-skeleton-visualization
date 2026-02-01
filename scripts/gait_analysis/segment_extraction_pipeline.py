#!/usr/bin/env python3
"""
Segment Extraction Pipeline for LSTM Gait Analysis

This is the recommended pipeline for extracting straight-line walking segments
from IMU gait data. Based on extensive experiments, this uses:
- Feature-preserving preprocessing (Smooth PCA + drift removal)
- Threshold-based segment extraction with optimized parameters

Usage:
    from scripts.gait_analysis.segment_extraction_pipeline import (
        extract_segments_from_file,
        extract_segments_from_data,
        SegmentExtractionConfig,
        ExtractedSegment
    )

    # From file
    segments, info = extract_segments_from_file("path/to/data.xlsx")

    # From numpy array
    segments, info = extract_segments_from_data(position_data)

    # With custom config
    config = SegmentExtractionConfig(min_segment_meters=7.0)
    segments, info = extract_segments_from_file("path/to/data.xlsx", config=config)
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

# Import internal modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_analysis.feature_preserving_correction import apply_feature_preserving_correction
from scripts.gait_analysis.improved_segment_extraction import (
    SegmentExtractionParams,
    ExtractedSegment,
    extract_segments_improved
)


@dataclass
class SegmentExtractionConfig:
    """
    Configuration for segment extraction pipeline.

    These are the recommended parameters based on experiments with 30 subjects.
    Modify only if you have specific requirements.
    """
    # Frame rate
    frame_rate: int = 60

    # Preprocessing
    use_preprocessing: bool = True
    drift_correction_strength: str = 'moderate'  # 'minimal', 'moderate', 'aggressive'

    # Segment extraction
    velocity_threshold: float = 0.4           # m/s - walking detection
    heading_change_threshold: float = 0.1     # rad/frame (~5.7°) - straight line detection
    trim_start_seconds: float = 0.5           # trim acceleration phase
    trim_end_seconds: float = 0.3             # trim deceleration phase
    min_segment_seconds: float = 2.0          # minimum segment duration
    min_segment_meters: float = 5.0           # minimum segment distance

    # Overlapping windows for LSTM
    use_overlapping_windows: bool = True
    window_seconds: float = 5.0               # LSTM input window size
    window_overlap: float = 0.5               # 50% overlap

    def to_extraction_params(self) -> SegmentExtractionParams:
        """Convert to SegmentExtractionParams."""
        return SegmentExtractionParams(
            velocity_threshold=self.velocity_threshold,
            heading_change_threshold=self.heading_change_threshold,
            trim_start_seconds=self.trim_start_seconds,
            trim_end_seconds=self.trim_end_seconds,
            min_segment_seconds=self.min_segment_seconds,
            min_segment_meters=self.min_segment_meters,
            use_overlapping_windows=self.use_overlapping_windows,
            window_seconds=self.window_seconds,
            window_overlap=self.window_overlap,
            frame_rate=self.frame_rate
        )


@dataclass
class PipelineResult:
    """Result from segment extraction pipeline."""
    segments: List[ExtractedSegment]
    info: dict
    config: SegmentExtractionConfig
    raw_data: Optional[np.ndarray] = None
    preprocessed_data: Optional[np.ndarray] = None


def compute_velocity(data: np.ndarray, frame_rate: int = 60) -> np.ndarray:
    """Compute pelvis velocity from position data."""
    pelvis_pos = data[:, 0, :2]  # Pelvis X, Y
    velocity = np.gradient(pelvis_pos, 1/frame_rate, axis=0)
    return velocity


def extract_segments_from_data(
    data: np.ndarray,
    config: Optional[SegmentExtractionConfig] = None,
    return_preprocessed: bool = False
) -> Union[Tuple[List[ExtractedSegment], dict], PipelineResult]:
    """
    Extract segments from position data array.

    Args:
        data: Position data array (n_frames, n_joints, 3)
        config: Extraction configuration (uses defaults if None)
        return_preprocessed: If True, return full PipelineResult with preprocessed data

    Returns:
        If return_preprocessed=False: (segments, info)
        If return_preprocessed=True: PipelineResult object
    """
    if config is None:
        config = SegmentExtractionConfig()

    raw_data = data.copy()

    # Step 1: Preprocessing
    if config.use_preprocessing:
        prep_result = apply_feature_preserving_correction(
            data.copy(),
            frame_rate=config.frame_rate,
            drift_correction_strength=config.drift_correction_strength
        )
        processed_data = prep_result.data
        preprocessing_info = prep_result.info
    else:
        processed_data = data.copy()
        preprocessing_info = {'preprocessing': 'none'}

    # Step 2: Compute velocity
    velocity = compute_velocity(processed_data, config.frame_rate)

    # Step 3: Extract segments
    extraction_params = config.to_extraction_params()
    segments, extraction_info = extract_segments_improved(
        processed_data, velocity, extraction_params
    )

    # Combine info
    info = {
        'n_segments': len(segments),
        'total_frames': len(data),
        'total_segment_frames': sum(s.data.shape[0] for s in segments),
        'straight_ratio': extraction_info['straight_ratio'],
        'preprocessing': preprocessing_info,
        'extraction': extraction_info,
    }

    if return_preprocessed:
        return PipelineResult(
            segments=segments,
            info=info,
            config=config,
            raw_data=raw_data,
            preprocessed_data=processed_data
        )

    return segments, info


def extract_segments_from_file(
    file_path: Union[str, Path],
    config: Optional[SegmentExtractionConfig] = None,
    return_preprocessed: bool = False
) -> Union[Tuple[List[ExtractedSegment], dict], PipelineResult]:
    """
    Extract segments from an Excel file.

    Args:
        file_path: Path to Excel file with Xsens data
        config: Extraction configuration (uses defaults if None)
        return_preprocessed: If True, return full PipelineResult with preprocessed data

    Returns:
        If return_preprocessed=False: (segments, info)
        If return_preprocessed=True: PipelineResult object
    """
    if config is None:
        config = SegmentExtractionConfig()

    file_path = Path(file_path)

    # Load data
    loader = load_xsens_data(file_path, frame_rate=config.frame_rate)
    data = loader.positions

    result = extract_segments_from_data(
        data, config, return_preprocessed=True
    )

    # Add file info
    result.info['file'] = str(file_path)
    result.info['file_name'] = file_path.stem

    if return_preprocessed:
        return result

    return result.segments, result.info


def batch_extract_segments(
    file_paths: List[Union[str, Path]],
    config: Optional[SegmentExtractionConfig] = None,
    verbose: bool = True
) -> dict:
    """
    Extract segments from multiple files.

    Args:
        file_paths: List of file paths
        config: Extraction configuration
        verbose: Print progress

    Returns:
        Dictionary with results for each file
    """
    if config is None:
        config = SegmentExtractionConfig()

    results = {}
    total_segments = 0

    for i, file_path in enumerate(file_paths, 1):
        file_path = Path(file_path)

        if verbose:
            print(f"[{i}/{len(file_paths)}] Processing: {file_path.stem}...")

        try:
            segments, info = extract_segments_from_file(file_path, config)
            results[file_path.stem] = {
                'segments': segments,
                'info': info,
                'success': True,
                'error': None
            }
            total_segments += len(segments)

            if verbose:
                print(f"    Extracted {len(segments)} segments")

        except Exception as e:
            results[file_path.stem] = {
                'segments': [],
                'info': {},
                'success': False,
                'error': str(e)
            }
            if verbose:
                print(f"    ERROR: {e}")

    if verbose:
        print(f"\nTotal: {total_segments} segments from {len(file_paths)} files")

    return results


def get_segments_as_array(
    segments: List[ExtractedSegment],
    pad_to_length: Optional[int] = None
) -> np.ndarray:
    """
    Convert list of segments to a single numpy array.

    Useful for batch processing in LSTM.

    Args:
        segments: List of ExtractedSegment
        pad_to_length: If specified, pad/truncate all segments to this length

    Returns:
        Array of shape (n_segments, n_frames, n_joints, 3)
    """
    if pad_to_length is None:
        # Find max length
        max_length = max(s.data.shape[0] for s in segments)
    else:
        max_length = pad_to_length

    n_segments = len(segments)
    n_joints = segments[0].data.shape[1]

    # Create output array
    result = np.zeros((n_segments, max_length, n_joints, 3))

    for i, seg in enumerate(segments):
        length = min(seg.data.shape[0], max_length)
        result[i, :length] = seg.data[:length]

    return result


# Convenience function for quick access
def quick_extract(file_path: Union[str, Path]) -> Tuple[List[ExtractedSegment], dict]:
    """
    Quick extraction with default parameters.

    Args:
        file_path: Path to Excel file

    Returns:
        (segments, info)
    """
    return extract_segments_from_file(file_path)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Extract walking segments from IMU data")
    parser.add_argument("input", help="Input Excel file or directory")
    parser.add_argument("--min-distance", type=float, default=5.0, help="Minimum segment distance (m)")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--output", help="Output directory for results")

    args = parser.parse_args()

    config = SegmentExtractionConfig(
        min_segment_meters=args.min_distance,
        use_preprocessing=not args.no_preprocess
    )

    input_path = Path(args.input)

    if input_path.is_file():
        segments, info = extract_segments_from_file(input_path, config)
        print(f"Extracted {len(segments)} segments from {input_path.name}")
        print(f"Straight ratio: {info['straight_ratio']*100:.1f}%")

    elif input_path.is_dir():
        files = list(input_path.glob("*.xlsx"))
        results = batch_extract_segments(files, config)

        total = sum(len(r['segments']) for r in results.values() if r['success'])
        print(f"\nTotal: {total} segments from {len(files)} files")
