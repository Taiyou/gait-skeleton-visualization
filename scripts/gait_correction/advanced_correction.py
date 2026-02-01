"""
Advanced Gait Correction Methods
Multiple approaches for correcting walking trajectory drift.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .turnaround import detect_turnarounds, detect_turnarounds_adaptive, TurnaroundInfo


@dataclass
class CorrectionResult:
    """Result of a correction method."""
    name: str
    data: np.ndarray
    description: str
    y_range_original: float
    y_range_corrected: float


# =============================================================================
# Method 1: Segment-wise Correction
# =============================================================================

def method1_segmentwise_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    min_segment_seconds: float = 3.0,
    turnaround_reference: Optional[np.ndarray] = None,
) -> CorrectionResult:
    """
    Method 1: Turnaround detection + Segment-wise Y alignment.

    Detects turnaround points and aligns Y coordinate for each segment.
    Does NOT apply rotation (which can cause discontinuities).

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        min_segment_seconds: Minimum segment duration
        turnaround_reference: Reference data for turnaround detection

    Returns:
        CorrectionResult with corrected data
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    # Get pelvis trajectory
    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Use reference data for turnaround detection if provided
    ref_data = turnaround_reference if turnaround_reference is not None else data
    ref_pelvis_x = ref_data[:, pelvis_index, 0]

    # Detect turnarounds using adaptive method
    turnarounds = detect_turnarounds_adaptive(
        ref_pelvis_x,
        frame_rate=frame_rate,
    )

    if len(turnarounds.segments) <= 1:
        # No turnarounds, return original
        return CorrectionResult(
            name="Method 1: Segment-wise",
            data=corrected,
            description="No turnarounds detected",
            y_range_original=np.max(pelvis_y) - np.min(pelvis_y),
            y_range_corrected=np.max(pelvis_y) - np.min(pelvis_y),
        )

    # Calculate target Y for each segment (use start Y of first segment as baseline)
    baseline_y = pelvis_y[turnarounds.segments[0][0]]

    # Build Y correction for each frame
    y_correction = np.zeros(n_frames)

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        # Use mean Y of segment center as representative Y
        center_start = start + (end - start) // 4
        center_end = end - (end - start) // 4
        seg_mean_y = np.mean(pelvis_y[center_start:center_end + 1])

        # Calculate offset from baseline
        y_offset = seg_mean_y - baseline_y

        # Apply correction to this segment
        y_correction[start:end + 1] = y_offset

    # Smooth corrections at segment boundaries
    smooth_window = int(0.5 * frame_rate)  # 0.5 second smoothing
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_correction_smooth = uniform_filter1d(y_correction, size=smooth_window, mode='nearest')

    # Apply Y correction to all segments
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_correction_smooth

    # Calculate metrics
    orig_y_range = np.max(pelvis_y) - np.min(pelvis_y)
    corr_y_range = np.max(corrected[:, pelvis_index, 1]) - np.min(corrected[:, pelvis_index, 1])

    return CorrectionResult(
        name="Method 1: Segment-wise",
        data=corrected,
        description="Turnaround detection + per-segment PCA alignment + Y reset",
        y_range_original=orig_y_range,
        y_range_corrected=corr_y_range,
    )


# =============================================================================
# Method 2: Cumulative Y Correction
# =============================================================================

def method2_cumulative_y_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    min_segment_seconds: float = 3.0,
    turnaround_reference: Optional[np.ndarray] = None,
) -> CorrectionResult:
    """
    Method 2: Cumulative Y correction (stride-based).

    Calculates median Y for each segment and aligns to a common baseline.

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        min_segment_seconds: Minimum segment duration
        turnaround_reference: Reference data for turnaround detection

    Returns:
        CorrectionResult with corrected data
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Use reference data for turnaround detection if provided
    ref_data = turnaround_reference if turnaround_reference is not None else data
    ref_pelvis_x = ref_data[:, pelvis_index, 0]

    # Detect turnarounds using adaptive method
    turnarounds = detect_turnarounds_adaptive(
        ref_pelvis_x,
        frame_rate=frame_rate,
    )

    # Calculate median Y for each segment
    segment_medians = []
    for start, end in turnarounds.segments:
        seg_y = pelvis_y[start:end + 1]
        segment_medians.append(np.median(seg_y))

    # Target: overall median Y
    target_y = np.median(segment_medians)

    # Create smooth Y correction curve
    y_correction = np.zeros(n_frames)

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        offset = segment_medians[seg_idx] - target_y
        y_correction[start:end + 1] = offset

    # Smooth transitions between segments
    smooth_window = int(0.5 * frame_rate)
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_correction_smooth = uniform_filter1d(y_correction, size=smooth_window, mode='nearest')

    # Apply correction
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_correction_smooth

    # Calculate metrics
    orig_y_range = np.max(pelvis_y) - np.min(pelvis_y)
    corr_y_range = np.max(corrected[:, pelvis_index, 1]) - np.min(corrected[:, pelvis_index, 1])

    return CorrectionResult(
        name="Method 2: Cumulative Y",
        data=corrected,
        description="Segment median Y alignment with smooth transitions",
        y_range_original=orig_y_range,
        y_range_corrected=corr_y_range,
    )


# =============================================================================
# Method 3: Local PCA Correction
# =============================================================================

def method3_local_pca_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    window_seconds: float = 5.0,
    sample_interval_seconds: float = 1.0,
) -> CorrectionResult:
    """
    Method 3: Local PCA correction (window-based).

    Applies PCA in short windows to correct local heading deviations.

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        window_seconds: Window size for local PCA
        sample_interval_seconds: Interval between samples

    Returns:
        CorrectionResult with corrected data
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    window_frames = int(window_seconds * frame_rate)
    interval_frames = int(sample_interval_seconds * frame_rate)

    # Calculate local PCA angles
    sample_frames = []
    sample_angles = []

    for center in range(window_frames // 2, n_frames - window_frames // 2, interval_frames):
        start = center - window_frames // 2
        end = center + window_frames // 2

        seg_x = pelvis_x[start:end]
        seg_y = pelvis_y[start:end]

        # Calculate displacement to determine walking direction
        dx = seg_x[-1] - seg_x[0]

        # Fit PCA
        xy_data = np.column_stack([seg_x - np.mean(seg_x), seg_y - np.mean(seg_y)])
        if len(xy_data) < 10:
            continue

        pca = PCA(n_components=2)
        pca.fit(xy_data)
        angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

        # Flip angle if walking in negative X direction
        if dx < 0:
            angle += np.pi

        sample_frames.append(center)
        sample_angles.append(angle)

    if len(sample_angles) < 3:
        # Not enough samples, return original
        return CorrectionResult(
            name="Method 3: Local PCA",
            data=data.copy(),
            description="Insufficient data for local PCA",
            y_range_original=np.max(pelvis_y) - np.min(pelvis_y),
            y_range_corrected=np.max(pelvis_y) - np.min(pelvis_y),
        )

    sample_frames = np.array(sample_frames)
    sample_angles = np.array(sample_angles)

    # Unwrap angles
    for i in range(1, len(sample_angles)):
        diff = sample_angles[i] - sample_angles[i - 1]
        if diff > np.pi / 2:
            sample_angles[i:] -= np.pi
        elif diff < -np.pi / 2:
            sample_angles[i:] += np.pi

    # Smooth angles
    smooth_window = max(3, len(sample_angles) // 10)
    if smooth_window % 2 == 0:
        smooth_window += 1
    sample_angles_smooth = uniform_filter1d(sample_angles, size=smooth_window, mode='nearest')

    # Interpolate to all frames
    interpolator = interp1d(
        sample_frames,
        sample_angles_smooth,
        kind='cubic',
        bounds_error=False,
        fill_value=(sample_angles_smooth[0], sample_angles_smooth[-1]),
    )
    all_angles = interpolator(np.arange(n_frames))

    # Calculate correction (align to mean angle)
    mean_angle = np.mean(all_angles)
    correction_angles = mean_angle - all_angles

    # Apply rotation
    for frame in range(n_frames):
        cos_a = np.cos(correction_angles[frame])
        sin_a = np.sin(correction_angles[frame])

        origin_x = data[frame, pelvis_index, 0]
        origin_y = data[frame, pelvis_index, 1]

        for body_idx in range(n_segments):
            rel_x = corrected[frame, body_idx, 0] - origin_x
            rel_y = corrected[frame, body_idx, 1] - origin_y

            new_x = rel_x * cos_a - rel_y * sin_a
            new_y = rel_x * sin_a + rel_y * cos_a

            corrected[frame, body_idx, 0] = origin_x + new_x
            corrected[frame, body_idx, 1] = origin_y + new_y

    # Recalculate pelvis and apply global alignment
    new_pelvis_x = corrected[:, pelvis_index, 0]
    new_pelvis_y = corrected[:, pelvis_index, 1]

    # Final horizontal alignment
    xy_data = np.column_stack([new_pelvis_x, new_pelvis_y])
    pca = PCA(n_components=2)
    pca.fit(xy_data)
    final_angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

    cos_a = np.cos(-final_angle)
    sin_a = np.sin(-final_angle)
    origin_x = new_pelvis_x[0]
    origin_y = new_pelvis_y[0]

    for body_idx in range(n_segments):
        rel_x = corrected[:, body_idx, 0] - origin_x
        rel_y = corrected[:, body_idx, 1] - origin_y

        corrected[:, body_idx, 0] = origin_x + rel_x * cos_a - rel_y * sin_a
        corrected[:, body_idx, 1] = origin_y + rel_x * sin_a + rel_y * cos_a

    # Calculate metrics
    orig_y_range = np.max(pelvis_y) - np.min(pelvis_y)
    corr_y_range = np.max(corrected[:, pelvis_index, 1]) - np.min(corrected[:, pelvis_index, 1])

    return CorrectionResult(
        name="Method 3: Local PCA",
        data=corrected,
        description="Short-window PCA with smooth interpolation",
        y_range_original=orig_y_range,
        y_range_corrected=corr_y_range,
    )


# =============================================================================
# Method 4: Reference Trajectory Matching
# =============================================================================

def method4_reference_matching(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    min_segment_seconds: float = 3.0,
    turnaround_reference: Optional[np.ndarray] = None,
) -> CorrectionResult:
    """
    Method 4: Reference trajectory matching.

    Warps each segment to match an ideal straight-line trajectory.

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        min_segment_seconds: Minimum segment duration
        turnaround_reference: Reference data for turnaround detection

    Returns:
        CorrectionResult with corrected data
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Use reference data for turnaround detection if provided
    ref_data = turnaround_reference if turnaround_reference is not None else data
    ref_pelvis_x = ref_data[:, pelvis_index, 0]

    # Detect turnarounds using adaptive method
    turnarounds = detect_turnarounds_adaptive(
        ref_pelvis_x,
        frame_rate=frame_rate,
    )

    # Process each segment
    cumulative_x = 0
    baseline_y = pelvis_y[0]

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        if end - start < 10:
            continue

        # Original segment endpoints
        orig_start_x = pelvis_x[start]
        orig_start_y = pelvis_y[start]
        orig_end_x = pelvis_x[end]
        orig_end_y = pelvis_y[end]

        # Calculate segment length and direction
        seg_length = np.sqrt((orig_end_x - orig_start_x)**2 + (orig_end_y - orig_start_y)**2)
        direction = turnarounds.directions[seg_idx]

        # Target: straight line along X-axis
        target_start_x = cumulative_x
        target_start_y = baseline_y
        target_end_x = cumulative_x + direction * seg_length
        target_end_y = baseline_y

        # Update cumulative X for next segment
        cumulative_x = target_end_x

        # Calculate affine transformation
        # Source: (orig_start, orig_end)
        # Target: (target_start, target_end)

        # For each frame, interpolate position along the segment
        for frame in range(start, end + 1):
            # Calculate progress along segment (0 to 1)
            if seg_length > 0:
                # Project current position onto segment line
                dx = pelvis_x[frame] - orig_start_x
                dy = pelvis_y[frame] - orig_start_y

                seg_dx = orig_end_x - orig_start_x
                seg_dy = orig_end_y - orig_start_y

                # Progress along main axis
                t = (dx * seg_dx + dy * seg_dy) / (seg_length ** 2)
                t = np.clip(t, 0, 1)

                # Perpendicular deviation (for Y variation)
                perp_dist = (-dx * seg_dy + dy * seg_dx) / seg_length
            else:
                t = 0
                perp_dist = 0

            # Target position for pelvis
            target_pelvis_x = target_start_x + t * (target_end_x - target_start_x)
            target_pelvis_y = target_start_y + perp_dist  # Keep natural Y variation

            # Calculate offset from original pelvis
            offset_x = target_pelvis_x - pelvis_x[frame]
            offset_y = target_pelvis_y - pelvis_y[frame]

            # Apply offset to all body segments
            for body_idx in range(n_segments):
                corrected[frame, body_idx, 0] += offset_x
                corrected[frame, body_idx, 1] += offset_y

    # Calculate metrics
    orig_y_range = np.max(pelvis_y) - np.min(pelvis_y)
    corr_y_range = np.max(corrected[:, pelvis_index, 1]) - np.min(corrected[:, pelvis_index, 1])

    return CorrectionResult(
        name="Method 4: Reference Matching",
        data=corrected,
        description="Affine warp to ideal straight-line trajectory",
        y_range_original=orig_y_range,
        y_range_corrected=corr_y_range,
    )


# =============================================================================
# Combined Method: Best of All
# =============================================================================

def method_combined(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    min_segment_seconds: float = 3.0,
    turnaround_reference: Optional[np.ndarray] = None,
) -> CorrectionResult:
    """
    Combined method: Applies multiple corrections in sequence.

    Pipeline:
    1. Local PCA correction (fix local heading)
    2. Segment-wise Y alignment (fix drift between segments)

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        min_segment_seconds: Minimum segment duration
        turnaround_reference: Reference data for turnaround detection

    Returns:
        CorrectionResult with corrected data
    """
    # Step 1: Local PCA
    result1 = method3_local_pca_correction(
        data,
        frame_rate=frame_rate,
        pelvis_index=pelvis_index,
        window_seconds=5.0,
        sample_interval_seconds=1.0,
    )

    # Step 2: Cumulative Y correction
    result2 = method2_cumulative_y_correction(
        result1.data,
        frame_rate=frame_rate,
        pelvis_index=pelvis_index,
        min_segment_seconds=min_segment_seconds,
        turnaround_reference=turnaround_reference,
    )

    pelvis_y = data[:, pelvis_index, 1]
    orig_y_range = np.max(pelvis_y) - np.min(pelvis_y)
    corr_y_range = np.max(result2.data[:, pelvis_index, 1]) - np.min(result2.data[:, pelvis_index, 1])

    return CorrectionResult(
        name="Method 5: Combined",
        data=result2.data,
        description="Local PCA + Cumulative Y correction",
        y_range_original=orig_y_range,
        y_range_corrected=corr_y_range,
    )


def apply_all_methods(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    original_data: Optional[np.ndarray] = None,
) -> List[CorrectionResult]:
    """
    Apply all correction methods and return results.

    Args:
        data: Position data (n_frames, n_segments, 3) - may be pre-corrected
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        original_data: Original uncorrected data for turnaround detection

    Returns:
        List of CorrectionResult for each method
    """
    results = []

    # Use original data for turnaround detection if provided
    turnaround_data = original_data if original_data is not None else data

    print("\n" + "=" * 60)
    print("Applying Method 1: Segment-wise Correction")
    print("=" * 60)
    results.append(method1_segmentwise_correction(
        data, frame_rate, pelvis_index,
        turnaround_reference=turnaround_data,
    ))

    print("\n" + "=" * 60)
    print("Applying Method 2: Cumulative Y Correction")
    print("=" * 60)
    results.append(method2_cumulative_y_correction(
        data, frame_rate, pelvis_index,
        turnaround_reference=turnaround_data,
    ))

    print("\n" + "=" * 60)
    print("Applying Method 3: Local PCA Correction")
    print("=" * 60)
    results.append(method3_local_pca_correction(data, frame_rate, pelvis_index))

    print("\n" + "=" * 60)
    print("Applying Method 4: Reference Trajectory Matching")
    print("=" * 60)
    results.append(method4_reference_matching(
        data, frame_rate, pelvis_index,
        turnaround_reference=turnaround_data,
    ))

    print("\n" + "=" * 60)
    print("Applying Method 5: Combined (Local PCA + Cumulative Y)")
    print("=" * 60)
    results.append(method_combined(
        data, frame_rate, pelvis_index,
        turnaround_reference=turnaround_data,
    ))

    return results
