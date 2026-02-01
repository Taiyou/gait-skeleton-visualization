"""
Smooth PCA Correction
Corrects heading drift using windowed PCA with smooth interpolation.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class SmoothPCAParams:
    """Parameters for Smooth PCA correction."""
    window_seconds: float = 30.0  # Window size for PCA calculation
    sample_interval_seconds: float = 5.0  # Interval between PCA samples
    smoothing_factor: float = 0.1  # Fraction of samples for smoothing
    min_window_samples: int = 100  # Minimum samples per window
    frame_rate: int = 60  # Frame rate in Hz


def calculate_pca_angle(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the principal axis angle from X, Y data.

    Args:
        x: X coordinates
        y: Y coordinates

    Returns:
        Angle in radians
    """
    data = np.column_stack([x, y])
    pca = PCA(n_components=2)
    pca.fit(data)
    return np.arctan2(pca.components_[0, 1], pca.components_[0, 0])


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap 180-degree flips in angle sequence.

    Args:
        angles: Array of angles in radians

    Returns:
        Unwrapped angles
    """
    unwrapped = angles.copy()
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i - 1]
        if diff > np.pi / 2:
            unwrapped[i:] -= np.pi
        elif diff < -np.pi / 2:
            unwrapped[i:] += np.pi
    return unwrapped


def apply_smooth_pca_correction(
    data: np.ndarray,
    skip_start_seconds: float = 5.0,
    skip_end_seconds: float = 5.0,
    params: Optional[SmoothPCAParams] = None,
    pelvis_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Smooth PCA correction to position data.

    Args:
        data: Position data, shape (n_frames, n_segments, 3)
        skip_start_seconds: Seconds to skip at start
        skip_end_seconds: Seconds to skip at end
        params: SmoothPCAParams object
        pelvis_index: Index of pelvis segment

    Returns:
        Tuple of (corrected_data, original_x, original_y)
    """
    if params is None:
        params = SmoothPCAParams()

    n_frames = data.shape[0]
    frame_rate = params.frame_rate

    # Calculate frame ranges
    skip_start = int(skip_start_seconds * frame_rate)
    skip_end = int(skip_end_seconds * frame_rate)
    valid_start = skip_start
    valid_end = n_frames - skip_end

    if valid_end <= valid_start:
        raise ValueError("Skip ranges exceed data length")

    # Extract pelvis trajectory
    pelvis_x = data[:, pelvis_index, 0].copy()
    pelvis_y = data[:, pelvis_index, 1].copy()

    # Calculate PCA angles at regular intervals
    window_frames = int(params.window_seconds * frame_rate)
    interval_frames = int(params.sample_interval_seconds * frame_rate)

    sample_frames = []
    sample_angles = []

    for center in range(valid_start, valid_end, interval_frames):
        start = max(0, center - window_frames // 2)
        end = min(n_frames, center + window_frames // 2)

        if end - start < params.min_window_samples:
            continue

        x_window = pelvis_x[start:end]
        y_window = pelvis_y[start:end]

        angle = calculate_pca_angle(x_window, y_window)
        sample_frames.append(center)
        sample_angles.append(angle)

    if len(sample_angles) < 2:
        raise ValueError("Not enough samples for PCA correction")

    sample_frames = np.array(sample_frames)
    sample_angles = np.array(sample_angles)

    # Unwrap 180-degree flips
    sample_angles = unwrap_angles(sample_angles)

    # Smooth sampled angles
    smooth_window = max(3, int(len(sample_angles) * params.smoothing_factor))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed_angles = uniform_filter1d(sample_angles, size=smooth_window, mode='nearest')

    # Interpolate to all frames
    interpolator = interp1d(
        sample_frames,
        smoothed_angles,
        kind='cubic',
        bounds_error=False,
        fill_value=(smoothed_angles[0], smoothed_angles[-1]),
    )
    all_angles = interpolator(np.arange(n_frames))

    # Additional smoothing with 2-second window
    smooth_window_2 = int(2.0 * frame_rate)
    if smooth_window_2 % 2 == 0:
        smooth_window_2 += 1
    all_angles = uniform_filter1d(all_angles, size=smooth_window_2, mode='nearest')

    # Calculate correction angles (rotate to align with X-axis)
    # Use the angle at the beginning as reference
    reference_angle = all_angles[valid_start]
    correction_angles = reference_angle - all_angles

    # Apply rotation to all segments
    corrected_data = data.copy()
    pelvis_center_x = data[:, pelvis_index, 0]
    pelvis_center_y = data[:, pelvis_index, 1]

    n_segments = data.shape[1]

    for seg_idx in range(n_segments):
        # Shift relative to pelvis
        rel_x = data[:, seg_idx, 0] - pelvis_center_x
        rel_y = data[:, seg_idx, 1] - pelvis_center_y

        # Apply rotation
        cos_angle = np.cos(correction_angles)
        sin_angle = np.sin(correction_angles)

        new_rel_x = rel_x * cos_angle - rel_y * sin_angle
        new_rel_y = rel_x * sin_angle + rel_y * cos_angle

        corrected_data[:, seg_idx, 0] = pelvis_center_x + new_rel_x
        corrected_data[:, seg_idx, 1] = pelvis_center_y + new_rel_y

    # Also rotate pelvis center itself
    # Use frame 0 as origin
    origin_x = pelvis_center_x[0]
    origin_y = pelvis_center_y[0]

    for seg_idx in range(n_segments):
        rel_x = corrected_data[:, seg_idx, 0] - origin_x
        rel_y = corrected_data[:, seg_idx, 1] - origin_y

        cos_angle = np.cos(correction_angles)
        sin_angle = np.sin(correction_angles)

        new_x = rel_x * cos_angle - rel_y * sin_angle
        new_y = rel_x * sin_angle + rel_y * cos_angle

        corrected_data[:, seg_idx, 0] = origin_x + new_x
        corrected_data[:, seg_idx, 1] = origin_y + new_y

    print(f"Applied Smooth PCA correction")
    print(f"  Sampled {len(sample_frames)} PCA angles")
    print(f"  Reference angle: {np.degrees(reference_angle):.2f}°")
    print(f"  Max correction: {np.degrees(np.max(np.abs(correction_angles))):.2f}°")

    return corrected_data, pelvis_x, pelvis_y
