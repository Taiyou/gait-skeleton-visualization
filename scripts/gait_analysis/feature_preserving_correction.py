#!/usr/bin/env python3
"""
Feature-Preserving Correction for LSTM Analysis

This module provides preprocessing that preserves gait patterns
while removing global drift - suitable for LSTM feature extraction.

Key differences from visualization preprocessing:
1. Larger window sizes to preserve local patterns
2. Bandpass filtering to keep gait rhythm (0.5-3Hz)
3. Body-centered coordinate transformation
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CorrectionResult:
    """Result of feature-preserving correction."""
    data: np.ndarray
    original_data: np.ndarray
    body_centered_data: np.ndarray
    info: dict


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_highpass(cutoff: float, fs: float, order: int = 4):
    """Design a Butterworth highpass filter."""
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype='high')
    return b, a


def apply_feature_preserving_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    drift_window_seconds: float = 60.0,  # Larger window to preserve local patterns
    preserve_gait_rhythm: bool = True,
    gait_freq_low: float = 0.5,  # Hz - lower bound of gait frequency
    gait_freq_high: float = 3.0,  # Hz - upper bound of gait frequency
    drift_correction_strength: str = 'moderate',  # minimal/moderate/aggressive
) -> CorrectionResult:
    """
    Apply feature-preserving correction for LSTM analysis.

    This method:
    1. Removes global heading drift (large window PCA)
    2. Removes Y-axis drift while preserving gait oscillations
    3. Transforms to body-centered coordinates

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        drift_window_seconds: Window for drift estimation (larger = more preservation)
        preserve_gait_rhythm: If True, use bandpass to preserve gait frequencies
        gait_freq_low: Lower frequency bound for gait (Hz)
        gait_freq_high: Upper frequency bound for gait (Hz)
        drift_correction_strength: 'minimal', 'moderate', or 'aggressive'

    Returns:
        CorrectionResult with corrected data and info
    """
    original_data = data.copy()
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape
    info = {}

    # Strength parameters
    strength_params = {
        'minimal': {'pca_window': 90.0, 'drift_cutoff': 0.05},
        'moderate': {'pca_window': 60.0, 'drift_cutoff': 0.1},
        'aggressive': {'pca_window': 30.0, 'drift_cutoff': 0.2},
    }
    params = strength_params.get(drift_correction_strength, strength_params['moderate'])

    pca_window = int(params['pca_window'] * frame_rate)
    drift_cutoff = params['drift_cutoff']

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # =========================================================================
    # Step 1: Global PCA rotation (with large window)
    # =========================================================================
    print(f"\n[Step 1] Global PCA rotation (strength: {drift_correction_strength})...")

    xy_data = np.column_stack([pelvis_x, pelvis_y])
    center = np.mean(xy_data, axis=0)
    xy_centered = xy_data - center

    pca = PCA(n_components=2)
    pca.fit(xy_centered)

    principal_axis = pca.components_[0]
    global_angle = np.arctan2(principal_axis[1], principal_axis[0])

    cos_a = np.cos(-global_angle)
    sin_a = np.sin(-global_angle)

    for body_idx in range(n_segments):
        x = corrected[:, body_idx, 0] - center[0]
        y = corrected[:, body_idx, 1] - center[1]
        corrected[:, body_idx, 0] = x * cos_a - y * sin_a
        corrected[:, body_idx, 1] = x * sin_a + y * cos_a

    info['rotation_angle_deg'] = np.degrees(-global_angle)
    print(f"  Rotated by {info['rotation_angle_deg']:.2f}°")

    # =========================================================================
    # Step 2: Y-drift removal (preserving gait oscillations)
    # =========================================================================
    print("\n[Step 2] Y-drift removal (preserving gait rhythm)...")

    pelvis_y_rotated = corrected[:, pelvis_index, 1].copy()
    original_y_range = pelvis_y_rotated.max() - pelvis_y_rotated.min()

    if preserve_gait_rhythm:
        # Use highpass filter to remove only very low frequency drift
        # This preserves the gait oscillations (typically 0.5-2 Hz)
        try:
            b, a = butter_highpass(drift_cutoff, frame_rate, order=2)
            y_highpassed = filtfilt(b, a, pelvis_y_rotated)

            # Calculate the drift component
            y_drift = pelvis_y_rotated - y_highpassed

            # Apply to all segments
            for body_idx in range(n_segments):
                corrected[:, body_idx, 1] -= y_drift

            info['drift_removal_method'] = 'highpass_filter'
            info['drift_cutoff_hz'] = drift_cutoff
        except Exception as e:
            print(f"  Warning: Highpass filter failed ({e}), using moving average")
            # Fallback to moving average
            window_frames = int(drift_window_seconds * frame_rate)
            if window_frames % 2 == 0:
                window_frames += 1
            y_drift = uniform_filter1d(pelvis_y_rotated, size=window_frames, mode='nearest')

            for body_idx in range(n_segments):
                corrected[:, body_idx, 1] -= y_drift

            info['drift_removal_method'] = 'moving_average'
    else:
        # Standard moving average (more aggressive)
        window_frames = int(drift_window_seconds * frame_rate)
        if window_frames % 2 == 0:
            window_frames += 1
        y_drift = uniform_filter1d(pelvis_y_rotated, size=window_frames, mode='nearest')

        for body_idx in range(n_segments):
            corrected[:, body_idx, 1] -= y_drift

        info['drift_removal_method'] = 'moving_average'

    # Center at Y=0
    final_median = np.median(corrected[:, pelvis_index, 1])
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= final_median

    corrected_y_range = corrected[:, pelvis_index, 1].max() - corrected[:, pelvis_index, 1].min()
    info['original_y_range'] = original_y_range
    info['corrected_y_range'] = corrected_y_range
    info['y_range_reduction_pct'] = (1 - corrected_y_range / original_y_range) * 100

    print(f"  Original Y range: {original_y_range:.2f}m")
    print(f"  Corrected Y range: {corrected_y_range:.2f}m")
    print(f"  Reduction: {info['y_range_reduction_pct']:.1f}%")

    # =========================================================================
    # Step 3: Body-centered coordinate transformation
    # =========================================================================
    print("\n[Step 3] Body-centered coordinate transformation...")

    body_centered = corrected.copy()

    # Center on pelvis for each frame
    for frame_idx in range(n_frames):
        pelvis_pos = corrected[frame_idx, pelvis_index, :].copy()
        body_centered[frame_idx, :, :] -= pelvis_pos

    info['body_centered'] = True
    print("  Centered on pelvis for each frame")

    # =========================================================================
    # Step 4: Calculate gait rhythm preservation metrics
    # =========================================================================
    print("\n[Step 4] Calculating gait rhythm metrics...")

    # FFT analysis to check if gait frequencies are preserved
    from scipy.fft import fft, fftfreq

    # Analyze Y-component (lateral sway has clear gait rhythm)
    original_y = original_data[:, pelvis_index, 1]
    corrected_y = corrected[:, pelvis_index, 1]

    # Compute power spectrum
    n = len(original_y)
    freqs = fftfreq(n, 1/frame_rate)

    # Only look at positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]

    original_fft = np.abs(fft(original_y - np.mean(original_y)))[pos_mask]
    corrected_fft = np.abs(fft(corrected_y - np.mean(corrected_y)))[pos_mask]

    # Find power in gait frequency band (0.5-3 Hz)
    gait_band_mask = (freqs_pos >= gait_freq_low) & (freqs_pos <= gait_freq_high)

    original_gait_power = np.sum(original_fft[gait_band_mask]**2)
    corrected_gait_power = np.sum(corrected_fft[gait_band_mask]**2)

    # Power preservation ratio (should be close to 1.0)
    if original_gait_power > 0:
        power_preservation = corrected_gait_power / original_gait_power
    else:
        power_preservation = 1.0

    info['gait_power_preservation'] = power_preservation
    print(f"  Gait rhythm power preservation: {power_preservation:.2%}")

    if power_preservation < 0.5:
        print("  WARNING: Significant gait rhythm loss detected!")
    elif power_preservation > 0.8:
        print("  OK: Gait rhythm well preserved")

    return CorrectionResult(
        data=corrected,
        original_data=original_data,
        body_centered_data=body_centered,
        info=info
    )


def compare_preprocessing_methods(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> dict:
    """
    Compare feature-preserving vs standard preprocessing.

    Returns dict with results from both methods for comparison.
    """
    results = {}

    # Feature-preserving (minimal)
    print("=" * 60)
    print("Method 1: Feature-Preserving (Minimal)")
    print("=" * 60)
    results['minimal'] = apply_feature_preserving_correction(
        data.copy(), frame_rate, pelvis_index,
        drift_correction_strength='minimal'
    )

    # Feature-preserving (moderate)
    print("\n" + "=" * 60)
    print("Method 2: Feature-Preserving (Moderate)")
    print("=" * 60)
    results['moderate'] = apply_feature_preserving_correction(
        data.copy(), frame_rate, pelvis_index,
        drift_correction_strength='moderate'
    )

    # Feature-preserving (aggressive) - similar to standard preprocessing
    print("\n" + "=" * 60)
    print("Method 3: Feature-Preserving (Aggressive)")
    print("=" * 60)
    results['aggressive'] = apply_feature_preserving_correction(
        data.copy(), frame_rate, pelvis_index,
        drift_correction_strength='aggressive'
    )

    return results
