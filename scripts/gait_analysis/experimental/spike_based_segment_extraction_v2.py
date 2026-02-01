#!/usr/bin/env python3
"""
Spike-Based Segment Extraction v2

Improvements over v1:
1. Merge nearby spikes into single turn regions
2. Adaptive prominence based on signal statistics
3. Minimum peak height threshold
4. Better handling of consecutive turns
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d


@dataclass
class SpikeExtractionParamsV2:
    """Parameters for spike-based segment extraction v2."""
    # Velocity threshold
    velocity_threshold: float = 0.4  # m/s

    # Spike detection parameters
    spike_prominence_percentile: float = 75  # Use percentile of signal for prominence
    spike_min_prominence: float = 0.02       # Absolute minimum prominence (rad/frame)
    spike_min_height_percentile: float = 80  # Minimum height as percentile
    spike_width_seconds: float = 0.2         # Minimum width of spike

    # Spike merging parameters (NEW)
    merge_distance_seconds: float = 1.5      # Merge spikes within this distance

    # Smoothing for spike detection
    smooth_window_seconds: float = 0.3

    # Exclusion zone around merged spike regions
    exclude_padding_seconds: float = 0.3     # Padding before/after merged region

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
    """
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    heading_unwrapped = np.unwrap(heading)
    heading_rate = np.abs(np.gradient(heading_unwrapped))

    smooth_frames = int(smooth_window_sec * frame_rate)
    if smooth_frames > 1 and smooth_frames % 2 == 0:
        smooth_frames += 1

    if smooth_frames >= 5:
        heading_rate_smooth = savgol_filter(heading_rate, smooth_frames, 2)
        heading_rate_smooth = np.maximum(heading_rate_smooth, 0)
    else:
        heading_rate_smooth = uniform_filter1d(heading_rate, size=max(3, smooth_frames))

    return heading_unwrapped, heading_rate_smooth


def detect_turn_spikes_v2(
    heading_rate: np.ndarray,
    params: SpikeExtractionParamsV2
) -> Tuple[np.ndarray, dict]:
    """
    Detect spikes with adaptive thresholds.
    """
    # Adaptive prominence based on signal statistics
    prominence_adaptive = np.percentile(heading_rate, params.spike_prominence_percentile)
    prominence = max(params.spike_min_prominence, prominence_adaptive * 0.5)

    # Minimum height threshold
    min_height = np.percentile(heading_rate, params.spike_min_height_percentile)

    min_width = int(params.spike_width_seconds * params.frame_rate * 0.3)

    # Find peaks
    peaks, properties = find_peaks(
        heading_rate,
        prominence=prominence,
        height=min_height,
        width=min_width,
    )

    spike_info = {
        'peaks': peaks,
        'prominences': properties.get('prominences', np.array([])),
        'heights': properties.get('peak_heights', np.array([])),
        'widths': properties.get('widths', np.array([])),
        'left_bases': properties.get('left_bases', np.array([])),
        'right_bases': properties.get('right_bases', np.array([])),
        'prominence_used': prominence,
        'min_height_used': min_height,
    }

    return peaks, spike_info


def merge_nearby_spikes(
    spike_indices: np.ndarray,
    spike_info: dict,
    merge_distance_frames: int,
    n_frames: int
) -> List[Tuple[int, int]]:
    """
    Merge nearby spikes into continuous turn regions.

    If spikes are within merge_distance_frames of each other,
    they are considered part of the same turn and merged.

    Returns list of (start_frame, end_frame) for each merged region.
    """
    if len(spike_indices) == 0:
        return []

    # Get spike extents (using widths if available)
    widths = spike_info.get('widths', np.ones(len(spike_indices)) * 10)
    left_bases = spike_info.get('left_bases', spike_indices - widths/2)
    right_bases = spike_info.get('right_bases', spike_indices + widths/2)

    # Create initial regions around each spike
    regions = []
    for i, peak in enumerate(spike_indices):
        width = int(widths[i]) if i < len(widths) else 10
        left = max(0, int(left_bases[i]) if i < len(left_bases) else peak - width)
        right = min(n_frames, int(right_bases[i]) if i < len(right_bases) else peak + width)
        regions.append([left, right, [peak]])  # [start, end, list of peaks]

    # Merge overlapping or nearby regions
    merged = []
    current = regions[0]

    for i in range(1, len(regions)):
        next_region = regions[i]

        # Check if regions should be merged
        # (next region starts within merge_distance of current region's end)
        if next_region[0] <= current[1] + merge_distance_frames:
            # Merge: extend current region and add peaks
            current[1] = max(current[1], next_region[1])
            current[2].extend(next_region[2])
        else:
            # No merge: save current and start new
            merged.append(current)
            current = next_region

    # Don't forget the last region
    merged.append(current)

    # Convert to (start, end) tuples
    return [(r[0], r[1], r[2]) for r in merged]


def create_exclusion_mask_v2(
    n_frames: int,
    merged_regions: List[Tuple[int, int, List[int]]],
    padding_frames: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Create exclusion mask from merged spike regions.

    Returns:
        exclude_mask: Boolean mask of frames to exclude
        final_regions: List of (start, end) for excluded regions (with padding)
    """
    exclude_mask = np.zeros(n_frames, dtype=bool)
    final_regions = []

    for start, end, peaks in merged_regions:
        # Add padding around the merged region
        padded_start = max(0, start - padding_frames)
        padded_end = min(n_frames, end + padding_frames)

        exclude_mask[padded_start:padded_end] = True
        final_regions.append((padded_start, padded_end))

    return exclude_mask, final_regions


def detect_straight_regions_spike_v2(
    velocity: np.ndarray,
    params: SpikeExtractionParamsV2
) -> Tuple[np.ndarray, dict]:
    """
    Detect straight-line walking regions using improved spike detection.
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

    # Detect spikes (turns) with adaptive thresholds
    spikes, spike_info = detect_turn_spikes_v2(heading_rate, params)

    # Merge nearby spikes into turn regions
    merge_distance_frames = int(params.merge_distance_seconds * params.frame_rate)

    if len(spikes) > 0:
        merged_regions = merge_nearby_spikes(
            spikes, spike_info, merge_distance_frames, n_frames
        )
    else:
        merged_regions = []

    # Create exclusion mask with padding
    padding_frames = int(params.exclude_padding_seconds * params.frame_rate)
    exclude_mask, final_regions = create_exclusion_mask_v2(
        n_frames, merged_regions, padding_frames
    )

    # Straight = walking AND not in exclusion zone
    straight_mask = is_walking & ~exclude_mask

    info = {
        'velocity_mag': velocity_mag,
        'heading': heading,
        'heading_rate': heading_rate,
        'spikes': spikes,
        'spike_info': spike_info,
        'merged_regions': merged_regions,
        'exclude_mask': exclude_mask,
        'final_exclusion_regions': final_regions,
        'n_spikes': len(spikes),
        'n_merged_regions': len(merged_regions),
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


def extract_segments_spike_v2(
    data: np.ndarray,
    velocity: np.ndarray,
    params: Optional[SpikeExtractionParamsV2] = None
) -> Tuple[List[ExtractedSegment], dict]:
    """
    Extract segments using improved spike-based turn detection.
    """
    if params is None:
        params = SpikeExtractionParamsV2()

    n_frames = len(data)

    # Step 1: Detect straight regions using improved spike detection
    straight_mask, detection_info = detect_straight_regions_spike_v2(velocity, params)

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
        trimmed_start = start + trim_start_frames
        trimmed_end = end - trim_end_frames

        if trimmed_start >= trimmed_end:
            continue

        if trimmed_end - trimmed_start < min_frames:
            continue

        segment_data = data[trimmed_start:trimmed_end].copy()

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

    info = {
        'total_frames': n_frames,
        'straight_frames': np.sum(straight_mask),
        'straight_ratio': np.sum(straight_mask) / n_frames,
        'raw_regions': len(regions),
        'full_segments': len(full_segments),
        'total_segments': len(segments),
        'n_spikes_detected': detection_info['n_spikes'],
        'n_merged_regions': detection_info['n_merged_regions'],
        'velocity_mag': detection_info['velocity_mag'],
        'heading_rate': detection_info['heading_rate'],
        'spikes': detection_info['spikes'],
        'spike_info': detection_info['spike_info'],
        'merged_regions': detection_info['merged_regions'],
        'exclude_mask': detection_info['exclude_mask'],
        'straight_mask': straight_mask,
    }

    return segments, info
