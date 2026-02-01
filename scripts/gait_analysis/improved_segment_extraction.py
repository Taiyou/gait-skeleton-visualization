#!/usr/bin/env python3
"""
Improved Segment Extraction

Improvements over basic velocity-threshold method:
1. Direction change detection - identify turns by heading change rate
2. Acceleration trimming - remove start/end transition periods
3. Overlapping windows - increase data samples from long segments
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.ndimage import uniform_filter1d


@dataclass
class SegmentExtractionParams:
    """Parameters for improved segment extraction."""
    # Velocity thresholds
    velocity_threshold: float = 0.4  # m/s (lowered from 0.5)

    # Direction change detection
    heading_change_threshold: float = 0.1  # rad/frame (~6 deg/frame at 60Hz)
    heading_smooth_window: float = 0.5  # seconds

    # Acceleration trimming
    trim_start_seconds: float = 0.5  # Remove first N seconds
    trim_end_seconds: float = 0.3    # Remove last N seconds

    # Minimum segment requirements
    min_segment_seconds: float = 2.0  # Minimum duration after trimming
    min_segment_meters: float = 5.0   # Minimum distance (lowered from 7m)

    # Overlapping windows
    use_overlapping_windows: bool = True
    window_seconds: float = 5.0       # Window size for LSTM
    window_overlap: float = 0.5       # 50% overlap

    # Frame rate
    frame_rate: int = 60


@dataclass
class ExtractedSegment:
    """A single extracted segment with metadata."""
    data: np.ndarray           # (n_frames, n_segments, 3)
    start_frame: int
    end_frame: int
    duration_sec: float
    distance_m: float
    mean_velocity: float
    is_window: bool = False    # True if from overlapping window
    parent_segment_id: int = -1  # Original segment ID if windowed


def compute_heading(velocity: np.ndarray) -> np.ndarray:
    """Compute heading angle from velocity."""
    return np.arctan2(velocity[:, 1], velocity[:, 0])


def compute_heading_change_rate(
    heading: np.ndarray,
    frame_rate: int,
    smooth_window_sec: float = 0.5
) -> np.ndarray:
    """
    Compute rate of heading change (angular velocity).

    Returns absolute change rate in rad/frame.
    """
    # Unwrap heading to handle -pi/pi discontinuity
    heading_unwrapped = np.unwrap(heading)

    # Compute derivative
    heading_rate = np.abs(np.gradient(heading_unwrapped))

    # Smooth to reduce noise
    smooth_frames = int(smooth_window_sec * frame_rate)
    if smooth_frames > 1:
        heading_rate = uniform_filter1d(heading_rate, size=smooth_frames, mode='nearest')

    return heading_rate


def detect_straight_regions(
    velocity: np.ndarray,
    params: SegmentExtractionParams
) -> np.ndarray:
    """
    Detect frames that are straight-line walking.

    Criteria:
    1. Velocity above threshold
    2. Heading change rate below threshold (not turning)

    Returns boolean mask.
    """
    # Velocity magnitude
    velocity_mag = np.linalg.norm(velocity, axis=1)

    # Heading and change rate
    heading = compute_heading(velocity)
    heading_rate = compute_heading_change_rate(
        heading,
        params.frame_rate,
        params.heading_smooth_window
    )

    # Combined criteria
    is_walking = velocity_mag > params.velocity_threshold
    is_straight = heading_rate < params.heading_change_threshold

    return is_walking & is_straight, velocity_mag, heading_rate


def find_continuous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find start and end indices of continuous True regions."""
    # Add padding to detect edges
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return list(zip(starts, ends))


def trim_segment(
    start: int,
    end: int,
    trim_start_frames: int,
    trim_end_frames: int
) -> Tuple[int, int]:
    """Trim start and end of segment to remove transitions."""
    new_start = start + trim_start_frames
    new_end = end - trim_end_frames

    if new_start >= new_end:
        return None

    return new_start, new_end


def create_overlapping_windows(
    data: np.ndarray,
    start_frame: int,
    end_frame: int,
    window_frames: int,
    overlap_ratio: float,
    frame_rate: int,
    segment_id: int,
    velocity_mag: np.ndarray
) -> List[ExtractedSegment]:
    """Create overlapping windows from a segment."""
    windows = []

    segment_length = end_frame - start_frame
    step_frames = int(window_frames * (1 - overlap_ratio))

    if step_frames < 1:
        step_frames = 1

    window_start = 0
    while window_start + window_frames <= segment_length:
        abs_start = start_frame + window_start
        abs_end = abs_start + window_frames

        window_data = data[abs_start:abs_end].copy()

        # Calculate distance
        pelvis_x = window_data[:, 0, 0]
        distance = abs(pelvis_x[-1] - pelvis_x[0])

        # Mean velocity
        mean_vel = np.mean(velocity_mag[abs_start:abs_end])

        windows.append(ExtractedSegment(
            data=window_data,
            start_frame=abs_start,
            end_frame=abs_end,
            duration_sec=window_frames / frame_rate,
            distance_m=distance,
            mean_velocity=mean_vel,
            is_window=True,
            parent_segment_id=segment_id
        ))

        window_start += step_frames

    return windows


def extract_segments_improved(
    data: np.ndarray,
    velocity: np.ndarray,
    params: Optional[SegmentExtractionParams] = None
) -> Tuple[List[ExtractedSegment], dict]:
    """
    Extract segments with improved method.

    Args:
        data: Position data (n_frames, n_segments, 3)
        velocity: Velocity data (n_frames, 2) for pelvis X, Y
        params: Extraction parameters

    Returns:
        List of ExtractedSegment, and diagnostic info dict
    """
    if params is None:
        params = SegmentExtractionParams()

    n_frames = len(data)

    # Step 1: Detect straight walking regions
    straight_mask, velocity_mag, heading_rate = detect_straight_regions(velocity, params)

    # Step 2: Find continuous regions
    regions = find_continuous_regions(straight_mask)

    # Step 3: Process each region
    trim_start_frames = int(params.trim_start_seconds * params.frame_rate)
    trim_end_frames = int(params.trim_end_seconds * params.frame_rate)
    min_frames = int(params.min_segment_seconds * params.frame_rate)
    window_frames = int(params.window_seconds * params.frame_rate)

    segments = []
    full_segments = []  # Before windowing

    for seg_id, (start, end) in enumerate(regions):
        # Trim transitions
        trimmed = trim_segment(start, end, trim_start_frames, trim_end_frames)
        if trimmed is None:
            continue

        trimmed_start, trimmed_end = trimmed

        # Check minimum duration
        if trimmed_end - trimmed_start < min_frames:
            continue

        # Extract segment data
        segment_data = data[trimmed_start:trimmed_end].copy()

        # Calculate distance
        pelvis_x = segment_data[:, 0, 0]
        distance = abs(pelvis_x[-1] - pelvis_x[0])

        # Check minimum distance
        if distance < params.min_segment_meters:
            continue

        # Calculate metrics
        duration = (trimmed_end - trimmed_start) / params.frame_rate
        mean_vel = np.mean(velocity_mag[trimmed_start:trimmed_end])

        full_segment = ExtractedSegment(
            data=segment_data,
            start_frame=trimmed_start,
            end_frame=trimmed_end,
            duration_sec=duration,
            distance_m=distance,
            mean_velocity=mean_vel,
            is_window=False,
            parent_segment_id=seg_id
        )
        full_segments.append(full_segment)

        # Create overlapping windows if enabled
        if params.use_overlapping_windows and (trimmed_end - trimmed_start) >= window_frames:
            windows = create_overlapping_windows(
                data,
                trimmed_start,
                trimmed_end,
                window_frames,
                params.window_overlap,
                params.frame_rate,
                seg_id,
                velocity_mag
            )
            segments.extend(windows)
        else:
            # Use full segment if too short for windowing
            segments.append(full_segment)

    # Diagnostic info
    info = {
        'total_frames': n_frames,
        'straight_frames': np.sum(straight_mask),
        'straight_ratio': np.sum(straight_mask) / n_frames,
        'raw_regions': len(regions),
        'full_segments': len(full_segments),
        'total_segments': len(segments),
        'windowed_segments': sum(1 for s in segments if s.is_window),
        'velocity_mag': velocity_mag,
        'heading_rate': heading_rate,
        'straight_mask': straight_mask,
    }

    return segments, info


def compare_extraction_methods(
    data: np.ndarray,
    velocity: np.ndarray,
    frame_rate: int = 60
) -> dict:
    """
    Compare old vs new extraction methods.

    Returns dict with both results for comparison.
    """
    # Old method (velocity only)
    old_params = SegmentExtractionParams(
        velocity_threshold=0.5,
        heading_change_threshold=999.0,  # Effectively disable
        trim_start_seconds=0.0,
        trim_end_seconds=0.0,
        use_overlapping_windows=False,
        min_segment_meters=7.0,
        frame_rate=frame_rate
    )
    old_segments, old_info = extract_segments_improved(data, velocity, old_params)

    # New method (full improvements)
    new_params = SegmentExtractionParams(
        velocity_threshold=0.4,
        heading_change_threshold=0.1,
        trim_start_seconds=0.5,
        trim_end_seconds=0.3,
        use_overlapping_windows=True,
        window_seconds=5.0,
        window_overlap=0.5,
        min_segment_meters=5.0,
        frame_rate=frame_rate
    )
    new_segments, new_info = extract_segments_improved(data, velocity, new_params)

    return {
        'old': {'segments': old_segments, 'info': old_info, 'params': old_params},
        'new': {'segments': new_segments, 'info': new_info, 'params': new_params},
    }
