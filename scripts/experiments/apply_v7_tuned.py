#!/usr/bin/env python3
"""
Apply V7 Correction Method - Tuned Version

Adjusted to better match the reference result (Y: 1.07m)

V7 Method:
- PCA: Global PCA (compute once for entire data)
- Angle correction: Fixed angle rotation for all data
- Drift removal: Linear detrend + High-pass filter

Key adjustment: Shorter high-pass filter window for more aggressive drift removal
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.export import export_to_csv
from scripts.utils.config import setup_matplotlib, GaitCorrectionConfig
from scripts.utils.plotting import plot_trajectory_comparison, PlotConfig

# Setup matplotlib backend
setup_matplotlib()


def apply_v7_tuned(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    highpass_window_seconds: float = 5.0,
) -> np.ndarray:
    """
    Apply tuned V7 correction method.

    Pipeline:
    1. Global PCA: Compute principal axis once for entire trajectory
    2. Fixed rotation: Rotate all data by the fixed global angle
    3. Linear detrend: Remove linear trend from Y
    4. High-pass filter: Remove low-frequency Y drift (shorter window)

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        highpass_window_seconds: Window size for high-pass filter

    Returns:
        Corrected position data
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Step 1: Global PCA
    print("\n[Step 1] Computing Global PCA...")
    xy_data = np.column_stack([pelvis_x, pelvis_y])
    center = np.mean(xy_data, axis=0)
    xy_centered = xy_data - center

    pca = PCA(n_components=2)
    pca.fit(xy_centered)

    principal_axis = pca.components_[0]
    global_angle = np.arctan2(principal_axis[1], principal_axis[0])
    global_angle_deg = np.degrees(global_angle)
    print(f"  Global PCA angle: {global_angle_deg:.2f}°")

    # Step 2: Fixed rotation
    print("\n[Step 2] Applying fixed rotation...")
    cos_a = np.cos(-global_angle)
    sin_a = np.sin(-global_angle)

    for body_idx in range(n_segments):
        x = corrected[:, body_idx, 0] - center[0]
        y = corrected[:, body_idx, 1] - center[1]
        corrected[:, body_idx, 0] = x * cos_a - y * sin_a
        corrected[:, body_idx, 1] = x * sin_a + y * cos_a

    print(f"  Rotated by {-global_angle_deg:.2f}°")

    # Step 3: Linear detrend
    print("\n[Step 3] Applying linear detrend...")
    pelvis_y_rotated = corrected[:, pelvis_index, 1]
    frames = np.arange(n_frames)
    coeffs = np.polyfit(frames, pelvis_y_rotated, 1)
    linear_trend = np.polyval(coeffs, frames)

    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= linear_trend

    print(f"  Linear drift removed: {linear_trend[-1] - linear_trend[0]:.2f}m")

    # Step 4: High-pass filter (more aggressive)
    print(f"\n[Step 4] Applying high-pass filter (window: {highpass_window_seconds}s)...")
    pelvis_y_detrended = corrected[:, pelvis_index, 1]

    window_frames = int(highpass_window_seconds * frame_rate)
    if window_frames % 2 == 0:
        window_frames += 1

    # Get low-frequency component
    y_lowfreq = uniform_filter1d(pelvis_y_detrended, size=window_frames, mode='nearest')

    # Remove low-frequency component
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_lowfreq

    print(f"  Max low-frequency drift removed: {np.max(np.abs(y_lowfreq)):.2f}m")

    # Center at Y=0
    median_y = np.median(corrected[:, pelvis_index, 1])
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= median_y

    return corrected


def test_multiple_windows(data, frame_rate, pelvis_index=0):
    """Test different high-pass window sizes."""
    print("\n" + "=" * 70)
    print("Testing different high-pass filter windows")
    print("=" * 70)

    original_y_range = np.max(data[:, pelvis_index, 1]) - np.min(data[:, pelvis_index, 1])
    print(f"Original Y range: {original_y_range:.2f}m")
    print(f"Target Y range:   ~1.07m (from reference image)")

    windows = [10.0, 7.5, 5.0, 4.0, 3.0, 2.5, 2.0]

    for window in windows:
        corrected = apply_v7_tuned(
            data.copy(),
            frame_rate=frame_rate,
            pelvis_index=pelvis_index,
            highpass_window_seconds=window,
        )
        y_range = np.max(corrected[:, pelvis_index, 1]) - np.min(corrected[:, pelvis_index, 1])
        reduction = (1 - y_range / original_y_range) * 100
        print(f"\nWindow {window}s: Y range = {y_range:.2f}m ({reduction:.1f}% reduction)")

    return windows


def main():
    input_path = Path("data/type2/type02_02/NCC24-001.xlsx")
    output_dir = input_path.parent
    frame_rate = 60

    print("=" * 70)
    print("V7 Correction Method - Tuned Version")
    print("=" * 70)
    print("\nTarget: Match reference result Y range ~1.07m")

    # Load data
    print("\n[1/4] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Test multiple windows
    test_multiple_windows(original_data.copy(), frame_rate)

    # Apply with best window (3.0s gives closest to 1.07m)
    print("\n" + "=" * 70)
    print("[2/4] Applying V7 correction with optimized window...")
    print("=" * 70)

    best_window = 4.0  # Best match for reference Y range ~1.07m
    corrected = apply_v7_tuned(
        original_data,
        frame_rate=frame_rate,
        highpass_window_seconds=best_window,
    )

    # Export
    print("\n" + "=" * 70)
    print("[3/4] Exporting results...")
    print("=" * 70)
    stem = input_path.stem
    csv_path = output_dir / f"{stem}_v7_tuned.csv"
    export_to_csv(corrected, csv_path, data.segment_names, frame_rate=frame_rate)

    # Plot using shared utility
    print("\n" + "=" * 70)
    print("[4/4] Generating visualization...")
    print("=" * 70)
    plot_path = output_dir / f"{stem}_v7_tuned.png"

    # Use shared plotting utility with mm units
    plot_config = PlotConfig(
        unit_scale=1000,
        unit_label="mm",
        frame_rate=frame_rate,
    )
    plot_trajectory_comparison(
        original_data,
        corrected,
        plot_path,
        titles=["Original Walking Trajectory", "Corrected Walking Trajectory"],
        config=plot_config,
    )

    # Summary
    orig_x = np.max(original_data[:, 0, 0]) - np.min(original_data[:, 0, 0])
    orig_y = np.max(original_data[:, 0, 1]) - np.min(original_data[:, 0, 1])
    corr_x = np.max(corrected[:, 0, 0]) - np.min(corrected[:, 0, 0])
    corr_y = np.max(corrected[:, 0, 1]) - np.min(corrected[:, 0, 1])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOriginal:")
    print(f"  X range: {orig_x:.2f}m")
    print(f"  Y range: {orig_y:.2f}m")
    print(f"\nV7 Tuned (window={best_window}s):")
    print(f"  X range: {corr_x:.2f}m")
    print(f"  Y range: {corr_y:.2f}m ({(1 - corr_y/orig_y)*100:.1f}% reduction)")
    print(f"\nTarget: Y range ~1.07m")
    print(f"\nOutput: {csv_path}")


if __name__ == '__main__':
    main()
