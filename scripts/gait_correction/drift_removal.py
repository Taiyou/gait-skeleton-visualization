"""
Drift Removal and Horizontal Alignment
Removes Y-axis drift and aligns trajectory with X-axis.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from typing import Optional


def remove_y_drift(
    data: np.ndarray,
    window_seconds: float = 30.0,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """
    Remove Y-axis drift using high-pass filtering.

    Args:
        data: Position data, shape (n_frames, n_segments, 3)
        window_seconds: Window size for drift estimation
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment

    Returns:
        Corrected position data
    """
    corrected = data.copy()

    # Extract pelvis Y trajectory
    pelvis_y = data[:, pelvis_index, 1].copy()

    # Estimate drift using low-pass filter
    window_frames = int(window_seconds * frame_rate)
    if window_frames % 2 == 0:
        window_frames += 1

    drift = uniform_filter1d(pelvis_y, size=window_frames, mode='nearest')

    # Remove drift from all segments
    y_correction = drift - drift[0]  # Keep starting position

    n_segments = data.shape[1]
    for seg_idx in range(n_segments):
        corrected[:, seg_idx, 1] -= y_correction

    print(f"Removed Y drift")
    print(f"  Max drift: {np.max(np.abs(y_correction)):.4f}")

    return corrected


def align_horizontally(
    data: np.ndarray,
    pelvis_index: int = 0,
) -> np.ndarray:
    """
    Align trajectory with X-axis using PCA.

    Args:
        data: Position data, shape (n_frames, n_segments, 3)
        pelvis_index: Index of pelvis segment

    Returns:
        Aligned position data
    """
    # Extract pelvis trajectory
    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Fit PCA to find principal axis
    xy_data = np.column_stack([pelvis_x, pelvis_y])
    pca = PCA(n_components=2)
    pca.fit(xy_data)

    # Calculate rotation angle to align with X-axis
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

    # Rotate all segments
    corrected = data.copy()
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)

    # Use starting position as origin
    origin_x = pelvis_x[0]
    origin_y = pelvis_y[0]

    n_segments = data.shape[1]
    for seg_idx in range(n_segments):
        rel_x = data[:, seg_idx, 0] - origin_x
        rel_y = data[:, seg_idx, 1] - origin_y

        new_x = rel_x * cos_angle - rel_y * sin_angle
        new_y = rel_x * sin_angle + rel_y * cos_angle

        corrected[:, seg_idx, 0] = origin_x + new_x
        corrected[:, seg_idx, 1] = origin_y + new_y

    print(f"Aligned horizontally")
    print(f"  Rotation angle: {np.degrees(angle):.2f}°")

    return corrected


def apply_full_drift_correction(
    data: np.ndarray,
    drift_window_seconds: float = 30.0,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """
    Apply full drift correction pipeline.

    Steps:
    1. Remove Y-axis drift
    2. Align horizontally with X-axis

    Args:
        data: Position data, shape (n_frames, n_segments, 3)
        drift_window_seconds: Window size for drift estimation
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment

    Returns:
        Corrected position data
    """
    print("Applying full drift correction...")

    # Step 1: Remove Y drift
    corrected = remove_y_drift(
        data,
        window_seconds=drift_window_seconds,
        frame_rate=frame_rate,
        pelvis_index=pelvis_index,
    )

    # Step 2: Align horizontally
    corrected = align_horizontally(corrected, pelvis_index=pelvis_index)

    return corrected
