"""
Turnaround Detection Module
Detects walking direction reversals in gait data.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TurnaroundInfo:
    """Information about detected turnarounds."""
    frames: np.ndarray  # Frame indices of turnarounds
    segments: List[Tuple[int, int]]  # List of (start, end) frame pairs
    directions: np.ndarray  # Direction of each segment (+1 or -1)


def detect_turnarounds(
    x: np.ndarray,
    frame_rate: int = 60,
    min_segment_seconds: float = 3.0,
    smooth_window_seconds: float = 1.0,
    min_displacement: float = 2.0,
) -> TurnaroundInfo:
    """
    Detect turnaround points in walking trajectory using peak detection.

    Args:
        x: X-coordinate trajectory (n_frames,)
        frame_rate: Frame rate in Hz
        min_segment_seconds: Minimum time between turnarounds
        smooth_window_seconds: Window for position smoothing
        min_displacement: Minimum displacement between peaks (meters)

    Returns:
        TurnaroundInfo with detected turnarounds
    """
    n_frames = len(x)
    min_segment_frames = int(min_segment_seconds * frame_rate)

    # Smooth position data
    smooth_window = int(smooth_window_seconds * frame_rate)
    if smooth_window % 2 == 0:
        smooth_window += 1
    x_smooth = uniform_filter1d(x, size=smooth_window, mode='nearest')

    # Find peaks (local maxima) and valleys (local minima)
    # Use prominence to filter out small fluctuations
    peaks_max, props_max = find_peaks(
        x_smooth,
        distance=min_segment_frames,
        prominence=min_displacement / 2,
    )

    peaks_min, props_min = find_peaks(
        -x_smooth,
        distance=min_segment_frames,
        prominence=min_displacement / 2,
    )

    # Combine and sort all extrema
    all_peaks = np.sort(np.concatenate([peaks_max, peaks_min]))

    # Filter peaks to ensure alternating max/min pattern
    if len(all_peaks) < 2:
        # No turnarounds detected, return single segment
        return TurnaroundInfo(
            frames=np.array([0, n_frames - 1]),
            segments=[(0, n_frames - 1)],
            directions=np.array([1 if x[-1] > x[0] else -1]),
        )

    # Build turnaround list ensuring minimum displacement between consecutive points
    turnaround_frames = [0]
    last_x = x_smooth[0]

    for peak in all_peaks:
        if abs(x_smooth[peak] - last_x) >= min_displacement:
            turnaround_frames.append(peak)
            last_x = x_smooth[peak]

    turnaround_frames.append(n_frames - 1)
    turnaround_frames = np.array(turnaround_frames)

    # Remove duplicates and sort
    turnaround_frames = np.unique(turnaround_frames)

    # Build segments
    segments = []
    directions = []

    for i in range(len(turnaround_frames) - 1):
        start = turnaround_frames[i]
        end = turnaround_frames[i + 1]

        # Skip very short segments
        if end - start < min_segment_frames // 2:
            continue

        segments.append((start, end))
        dx = x[end] - x[start]
        directions.append(1 if dx > 0 else -1)

    # Rebuild turnaround_frames from segments
    if len(segments) > 0:
        turnaround_frames = [segments[0][0]]
        for start, end in segments:
            turnaround_frames.append(end)
        turnaround_frames = np.array(turnaround_frames)
    else:
        turnaround_frames = np.array([0, n_frames - 1])
        segments = [(0, n_frames - 1)]
        directions = [1 if x[-1] > x[0] else -1]

    print(f"Detected {len(turnaround_frames) - 2} turnarounds")
    print(f"Created {len(segments)} segments")
    if len(segments) > 0:
        seg_lengths = [end - start for start, end in segments]
        print(f"Segment lengths: min={min(seg_lengths)}, max={max(seg_lengths)}, "
              f"mean={np.mean(seg_lengths):.0f} frames")

    return TurnaroundInfo(
        frames=turnaround_frames,
        segments=segments,
        directions=np.array(directions),
    )


def detect_turnarounds_adaptive(
    x: np.ndarray,
    frame_rate: int = 60,
    expected_walk_length_meters: float = 10.0,
) -> TurnaroundInfo:
    """
    Adaptively detect turnarounds based on data characteristics.

    Args:
        x: X-coordinate trajectory (n_frames,)
        frame_rate: Frame rate in Hz
        expected_walk_length_meters: Expected length of one walking pass

    Returns:
        TurnaroundInfo with detected turnarounds
    """
    # Calculate total displacement
    total_range = np.max(x) - np.min(x)

    # Calculate total distance traveled
    dx = np.diff(x)
    total_distance = np.sum(np.abs(dx))

    # Estimate number of passes
    estimated_passes = max(1, int(total_distance / expected_walk_length_meters))

    # Calculate minimum segment time based on typical walking speed (1.2 m/s)
    typical_speed = 1.2  # m/s
    min_segment_seconds = expected_walk_length_meters / typical_speed * 0.5

    # Adaptive min_displacement based on walk length
    min_displacement = min(total_range * 0.3, expected_walk_length_meters * 0.5)

    print(f"Adaptive detection:")
    print(f"  Total X range: {total_range:.2f}m")
    print(f"  Total distance traveled: {total_distance:.2f}m")
    print(f"  Estimated passes: {estimated_passes}")
    print(f"  Min segment duration: {min_segment_seconds:.1f}s")
    print(f"  Min displacement threshold: {min_displacement:.2f}m")

    return detect_turnarounds(
        x,
        frame_rate=frame_rate,
        min_segment_seconds=min_segment_seconds,
        min_displacement=min_displacement,
    )


def get_segment_data(
    data: np.ndarray,
    turnaround_info: TurnaroundInfo,
    segment_index: int,
) -> np.ndarray:
    """
    Extract data for a specific segment.

    Args:
        data: Full position data (n_frames, n_segments, 3)
        turnaround_info: TurnaroundInfo object
        segment_index: Index of segment to extract

    Returns:
        Position data for the segment
    """
    start, end = turnaround_info.segments[segment_index]
    return data[start:end + 1].copy()
