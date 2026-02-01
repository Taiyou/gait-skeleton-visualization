#!/usr/bin/env python3
"""
Spike-Based Segment Extraction

Instead of using a fixed threshold for heading change rate,
detect spikes (peaks) in heading change to identify turns.

This approach is more robust to:
- Baseline drift in heading rate
- Different walking styles
- Varying noise levels across subjects
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d


@dataclass
class SpikeExtractionParams:
    """Parameters for spike-based segment extraction."""
    # Velocity threshold
    velocity_threshold: float = 0.4  # m/s

    # Spike detection parameters
    spike_prominence: float = 0.05   # Minimum prominence of spike (rad/frame)
    spike_width_seconds: float = 0.5  # Minimum width of spike region to exclude
    spike_distance_seconds: float = 1.0  # Minimum distance between spikes

    # Smoothing for spike detection
    smooth_window_seconds: float = 0.3

    # Exclusion zone around spikes
    exclude_before_spike_seconds: float = 0.5  # Exclude before spike
    exclude_after_spike_seconds: float = 0.5   # Exclude after spike

    # Trimming
    trim_start_seconds: float = 0.5
    trim_end_seconds: float = 0.3

    # Minimum segment requirements
    min_segment_seconds: float = 2.0
    min_segment_meters: float = 5.0

    # Overlapping windows
    use_overlapping_windows: bool = True
    window_seconds: float = 5.0
    window_overlap: float = 0.5

    # Frame rate
    frame_rate: int = 60


@dataclass
class ExtractedSegment:
    """A single extracted segment with metadata."""
    data: np.ndarray
    start_frame: int
    end_frame: int
    duration_sec: float
    distance_m: float
    mean_velocity: float
    is_window: bool = False
    parent_segment_id: int = -1


def compute_heading_change_rate(
    velocity: np.ndarray,
    frame_rate: int,
    smooth_window_sec: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute heading and its change rate from velocity.

    Returns:
        heading: Unwrapped heading angle
        heading_rate: Smoothed absolute change rate
    """
    # Compute heading
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    heading_unwrapped = np.unwrap(heading)

    # Compute change rate
    heading_rate = np.abs(np.gradient(heading_unwrapped))

    # Smooth to reduce noise while preserving spikes
    smooth_frames = int(smooth_window_sec * frame_rate)
    if smooth_frames > 1 and smooth_frames % 2 == 0:
        smooth_frames += 1  # Must be odd for savgol

    if smooth_frames >= 5:
        # Savitzky-Golay preserves peak shape better than uniform filter
        heading_rate_smooth = savgol_filter(heading_rate, smooth_frames, 2)
        heading_rate_smooth = np.maximum(heading_rate_smooth, 0)
    else:
        heading_rate_smooth = uniform_filter1d(heading_rate, size=max(3, smooth_frames))

    return heading_unwrapped, heading_rate_smooth


def detect_turn_spikes(
    heading_rate: np.ndarray,
    params: SpikeExtractionParams
) -> Tuple[np.ndarray, dict]:
    """
    Detect spikes in heading change rate that indicate turns.

    Uses scipy.signal.find_peaks with:
    - prominence: how much the peak stands out from baseline
    - distance: minimum frames between peaks
    - width: peak must be wide enough to be a real turn

    Returns:
        spike_indices: Array of frame indices where spikes occur
        spike_info: Dict with additional spike properties
    """
    min_distance = int(params.spike_distance_seconds * params.frame_rate)
    min_width = int(params.spike_width_seconds * params.frame_rate * 0.3)  # 30% of width

    # Find peaks with prominence-based detection
    peaks, properties = find_peaks(
        heading_rate,
        prominence=params.spike_prominence,
        distance=min_distance,
        width=min_width,
    )

    spike_info = {
        'peaks': peaks,
        'prominences': properties.get('prominences', []),
        'widths': properties.get('widths', []),
        'left_bases': properties.get('left_bases', []),
        'right_bases': properties.get('right_bases', []),
    }

    return peaks, spike_info


def create_exclusion_mask(
    n_frames: int,
    spike_indices: np.ndarray,
    spike_info: dict,
    params: SpikeExtractionParams
) -> np.ndarray:
    """
    Create mask of frames to EXCLUDE (turn regions).

    Excludes region around each spike based on:
    - Fixed time before/after spike
    - Or based on detected peak width
    """
    exclude_mask = np.zeros(n_frames, dtype=bool)

    exclude_before = int(params.exclude_before_spike_seconds * params.frame_rate)
    exclude_after = int(params.exclude_after_spike_seconds * params.frame_rate)

    for i, peak in enumerate(spike_indices):
        # Use peak width if available, otherwise fixed window
        if len(spike_info.get('widths', [])) > i:
            width = int(spike_info['widths'][i])
            left = max(0, peak - max(exclude_before, width))
            right = min(n_frames, peak + max(exclude_after, width))
        else:
            left = max(0, peak - exclude_before)
            right = min(n_frames, peak + exclude_after)

        exclude_mask[left:right] = True

    return exclude_mask


def detect_straight_regions_spike_based(
    velocity: np.ndarray,
    params: SpikeExtractionParams
) -> Tuple[np.ndarray, dict]:
    """
    Detect straight-line walking regions using spike detection.

    Returns:
        straight_mask: Boolean mask of straight walking frames
        info: Diagnostic information
    """
    n_frames = len(velocity)

    # Velocity magnitude
    velocity_mag = np.linalg.norm(velocity, axis=1)
    is_walking = velocity_mag > params.velocity_threshold

    # Compute heading change rate
    heading, heading_rate = compute_heading_change_rate(
        velocity,
        params.frame_rate,
        params.smooth_window_seconds
    )

    # Detect spikes (turns)
    spikes, spike_info = detect_turn_spikes(heading_rate, params)

    # Create exclusion mask around spikes
    exclude_mask = create_exclusion_mask(n_frames, spikes, spike_info, params)

    # Straight = walking AND not in exclusion zone
    straight_mask = is_walking & ~exclude_mask

    info = {
        'velocity_mag': velocity_mag,
        'heading': heading,
        'heading_rate': heading_rate,
        'spikes': spikes,
        'spike_info': spike_info,
        'exclude_mask': exclude_mask,
        'n_spikes': len(spikes),
    }

    return straight_mask, info


def find_continuous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find start and end indices of continuous True regions."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


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
        pelvis_x = window_data[:, 0, 0]
        distance = abs(pelvis_x[-1] - pelvis_x[0])
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


def extract_segments_spike_based(
    data: np.ndarray,
    velocity: np.ndarray,
    params: Optional[SpikeExtractionParams] = None
) -> Tuple[List[ExtractedSegment], dict]:
    """
    Extract segments using spike-based turn detection.

    Args:
        data: Position data (n_frames, n_joints, 3)
        velocity: Velocity data (n_frames, 2) for pelvis X, Y
        params: Extraction parameters

    Returns:
        List of ExtractedSegment, diagnostic info
    """
    if params is None:
        params = SpikeExtractionParams()

    n_frames = len(data)

    # Step 1: Detect straight regions using spike detection
    straight_mask, detection_info = detect_straight_regions_spike_based(velocity, params)

    # Step 2: Find continuous regions
    regions = find_continuous_regions(straight_mask)

    # Step 3: Process each region
    trim_start_frames = int(params.trim_start_seconds * params.frame_rate)
    trim_end_frames = int(params.trim_end_seconds * params.frame_rate)
    min_frames = int(params.min_segment_seconds * params.frame_rate)
    window_frames = int(params.window_seconds * params.frame_rate)

    segments = []
    full_segments = []

    for seg_id, (start, end) in enumerate(regions):
        # Trim transitions
        trimmed_start = start + trim_start_frames
        trimmed_end = end - trim_end_frames

        if trimmed_start >= trimmed_end:
            continue

        if trimmed_end - trimmed_start < min_frames:
            continue

        segment_data = data[trimmed_start:trimmed_end].copy()

        # Calculate distance
        pelvis_x = segment_data[:, 0, 0]
        distance = abs(pelvis_x[-1] - pelvis_x[0])

        if distance < params.min_segment_meters:
            continue

        duration = (trimmed_end - trimmed_start) / params.frame_rate
        mean_vel = np.mean(detection_info['velocity_mag'][trimmed_start:trimmed_end])

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

        # Create overlapping windows
        if params.use_overlapping_windows and (trimmed_end - trimmed_start) >= window_frames:
            windows = create_overlapping_windows(
                data, trimmed_start, trimmed_end,
                window_frames, params.window_overlap,
                params.frame_rate, seg_id,
                detection_info['velocity_mag']
            )
            segments.extend(windows)
        else:
            segments.append(full_segment)

    # Full info
    info = {
        'total_frames': n_frames,
        'straight_frames': np.sum(straight_mask),
        'straight_ratio': np.sum(straight_mask) / n_frames,
        'raw_regions': len(regions),
        'full_segments': len(full_segments),
        'total_segments': len(segments),
        'n_spikes_detected': detection_info['n_spikes'],
        'velocity_mag': detection_info['velocity_mag'],
        'heading_rate': detection_info['heading_rate'],
        'spikes': detection_info['spikes'],
        'spike_info': detection_info['spike_info'],
        'exclude_mask': detection_info['exclude_mask'],
        'straight_mask': straight_mask,
    }

    return segments, info
