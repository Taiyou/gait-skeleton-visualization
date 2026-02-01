#!/usr/bin/env python3
"""
Apply V7 Correction Method to type02_02

V7 Method (as shown in the reference image):
- PCA: Global PCA (compute once for entire data)
- Angle correction: Fixed angle rotation for all data
- Drift removal: Linear detrend + High-pass filter (10 second window)

This is different from Smooth PCA which uses sliding windows.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.export import export_to_csv


def apply_v7_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    highpass_window_seconds: float = 10.0,
) -> np.ndarray:
    """
    Apply V7 correction method.

    V7 Pipeline:
    1. Global PCA: Compute principal axis once for entire trajectory
    2. Fixed rotation: Rotate all data by the fixed global angle
    3. Linear detrend: Remove linear trend from Y
    4. High-pass filter: Remove low-frequency Y drift (10s window)

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

    # Get pelvis trajectory
    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # =========================================================================
    # Step 1: Global PCA - compute principal axis once for entire trajectory
    # =========================================================================
    print("\n[Step 1] Computing Global PCA...")

    # Use all XY positions for PCA
    xy_data = np.column_stack([pelvis_x, pelvis_y])

    # Center the data
    center = np.mean(xy_data, axis=0)
    xy_centered = xy_data - center

    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(xy_centered)

    # Principal axis angle
    principal_axis = pca.components_[0]
    global_angle = np.arctan2(principal_axis[1], principal_axis[0])
    global_angle_deg = np.degrees(global_angle)

    print(f"  Global PCA angle: {global_angle_deg:.2f}°")

    # =========================================================================
    # Step 2: Fixed rotation - rotate all data by global angle
    # =========================================================================
    print("\n[Step 2] Applying fixed rotation...")

    # Rotation matrix to align principal axis with X
    cos_a = np.cos(-global_angle)
    sin_a = np.sin(-global_angle)

    for body_idx in range(n_segments):
        x = corrected[:, body_idx, 0] - center[0]
        y = corrected[:, body_idx, 1] - center[1]

        corrected[:, body_idx, 0] = x * cos_a - y * sin_a
        corrected[:, body_idx, 1] = x * sin_a + y * cos_a

    print(f"  Rotated by {-global_angle_deg:.2f}°")

    # =========================================================================
    # Step 3: Linear detrend - remove linear trend from Y
    # =========================================================================
    print("\n[Step 3] Applying linear detrend...")

    pelvis_y_rotated = corrected[:, pelvis_index, 1]

    # Fit linear trend
    frames = np.arange(n_frames)
    coeffs = np.polyfit(frames, pelvis_y_rotated, 1)
    linear_trend = np.polyval(coeffs, frames)

    # Remove trend from all segments
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= linear_trend

    slope_per_sec = coeffs[0] * frame_rate
    print(f"  Linear trend slope: {slope_per_sec:.4f} m/s")
    print(f"  Total linear drift removed: {linear_trend[-1] - linear_trend[0]:.2f}m")

    # =========================================================================
    # Step 4: High-pass filter - remove low-frequency Y drift
    # =========================================================================
    print("\n[Step 4] Applying high-pass filter...")

    pelvis_y_detrended = corrected[:, pelvis_index, 1]

    # Design high-pass filter
    # Cutoff frequency = 1 / window_seconds
    cutoff_freq = 1.0 / highpass_window_seconds
    nyquist = frame_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist

    # Ensure cutoff is valid
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.95
    if normalized_cutoff <= 0:
        normalized_cutoff = 0.01

    print(f"  Cutoff frequency: {cutoff_freq:.4f} Hz ({highpass_window_seconds}s window)")

    # Butterworth high-pass filter
    b, a = butter(2, normalized_cutoff, btype='high')

    # Apply filter to Y coordinate
    # Use filtfilt for zero-phase filtering
    y_lowfreq = pelvis_y_detrended - filtfilt(b, a, pelvis_y_detrended)

    # Alternative: use uniform filter to get low-frequency component
    window_frames = int(highpass_window_seconds * frame_rate)
    if window_frames % 2 == 0:
        window_frames += 1
    y_lowfreq_smooth = uniform_filter1d(pelvis_y_detrended, size=window_frames, mode='nearest')

    # Remove low-frequency component from all segments
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_lowfreq_smooth

    max_lowfreq_removed = np.max(np.abs(y_lowfreq_smooth))
    print(f"  Max low-frequency drift removed: {max_lowfreq_removed:.2f}m")

    # =========================================================================
    # Center at Y=0
    # =========================================================================
    median_y = np.median(corrected[:, pelvis_index, 1])
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= median_y

    return corrected


def plot_v7_comparison(
    original: np.ndarray,
    corrected: np.ndarray,
    output_path: Path,
    frame_rate: int = 60,
    pelvis_index: int = 0,
):
    """Create comparison plot for V7 correction."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    n_frames = len(original)
    time = np.arange(n_frames) / frame_rate / 60  # Time in minutes

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, n_frames))

    def calc_range(d, axis):
        return np.max(d[:, pelvis_index, axis]) - np.min(d[:, pelvis_index, axis])

    # Row 1: Trajectories
    ax = axes[0, 0]
    scatter = ax.scatter(
        original[:, pelvis_index, 0] * 1000,  # Convert to mm
        original[:, pelvis_index, 1] * 1000,
        c=time, cmap='viridis', s=0.5, alpha=0.5
    )
    ax.plot(original[0, pelvis_index, 0] * 1000, original[0, pelvis_index, 1] * 1000,
            'go', markersize=10, label='Start')
    ax.plot(original[-1, pelvis_index, 0] * 1000, original[-1, pelvis_index, 1] * 1000,
            'rs', markersize=10, label='End')
    ax.set_title(f'Original Trajectory\nX: {calc_range(original, 0):.2f}m, Y: {calc_range(original, 1):.2f}m')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(
        corrected[:, pelvis_index, 0] * 1000,
        corrected[:, pelvis_index, 1] * 1000,
        c=time, cmap='viridis', s=0.5, alpha=0.5
    )
    ax.plot(corrected[0, pelvis_index, 0] * 1000, corrected[0, pelvis_index, 1] * 1000,
            'go', markersize=10, label='Start')
    ax.plot(corrected[-1, pelvis_index, 0] * 1000, corrected[-1, pelvis_index, 1] * 1000,
            'rs', markersize=10, label='End')
    ax.set_title(f'Corrected Walking Trajectory\nX: {calc_range(corrected, 0):.2f}m, Y: {calc_range(corrected, 1):.2f}m')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Time (min)')

    # Overlay
    ax = axes[0, 2]
    ax.plot(original[:, pelvis_index, 0], original[:, pelvis_index, 1],
            'b-', alpha=0.3, linewidth=0.3, label='Original')
    ax.plot(corrected[:, pelvis_index, 0], corrected[:, pelvis_index, 1],
            'r-', alpha=0.6, linewidth=0.3, label='V7 Corrected')
    ax.set_title('Comparison Overlay')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Row 2: Time series
    step = max(1, n_frames // 5000)
    t = time[::step]

    ax = axes[1, 0]
    ax.plot(t, original[::step, pelvis_index, 0], 'b-', linewidth=0.5)
    ax.set_title('X Coordinate over Time')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('X (m)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, original[::step, pelvis_index, 1], 'b-', alpha=0.5, linewidth=0.5, label='Original')
    ax.plot(t, corrected[::step, pelvis_index, 1], 'r-', alpha=0.7, linewidth=0.5, label='V7 Corrected')
    ax.set_title('Y Coordinate over Time')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoomed Y (first 2 minutes)
    ax = axes[1, 2]
    zoom_frames = min(int(2 * 60 * frame_rate), n_frames)
    t_zoom = np.arange(zoom_frames) / frame_rate
    ax.plot(t_zoom, original[:zoom_frames, pelvis_index, 1], 'b-', alpha=0.5, linewidth=0.8, label='Original')
    ax.plot(t_zoom, corrected[:zoom_frames, pelvis_index, 1], 'r-', alpha=0.7, linewidth=0.8, label='V7 Corrected')
    ax.set_title('Y Coordinate (First 2 min)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to {output_path}")


def main():
    input_path = Path("data/type2/type02_02/NCC24-001.xlsx")
    output_dir = input_path.parent
    frame_rate = 60

    print("=" * 70)
    print("V7 Correction Method for type02_02")
    print("=" * 70)
    print("\nV7 Method:")
    print("  - PCA: Global PCA (compute once for entire data)")
    print("  - Angle: Fixed angle rotation for all data")
    print("  - Drift: Linear detrend + High-pass filter (10s window)")

    # Load data
    print("\n" + "=" * 70)
    print("[1/4] Loading data...")
    print("=" * 70)
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Apply V7 correction
    print("\n" + "=" * 70)
    print("[2/4] Applying V7 correction...")
    print("=" * 70)
    corrected = apply_v7_correction(
        original_data,
        frame_rate=frame_rate,
        highpass_window_seconds=10.0,
    )

    # Export
    print("\n" + "=" * 70)
    print("[3/4] Exporting results...")
    print("=" * 70)
    stem = input_path.stem
    csv_path = output_dir / f"{stem}_v7_corrected.csv"
    export_to_csv(corrected, csv_path, data.segment_names, frame_rate=frame_rate)

    # Plot
    print("\n" + "=" * 70)
    print("[4/4] Generating visualization...")
    print("=" * 70)
    plot_path = output_dir / f"{stem}_v7_comparison.png"
    plot_v7_comparison(original_data, corrected, plot_path, frame_rate=frame_rate)

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
    print(f"\nV7 Corrected:")
    print(f"  X range: {corr_x:.2f}m")
    print(f"  Y range: {corr_y:.2f}m ({(1 - corr_y/orig_y)*100:.1f}% reduction)")
    print(f"\nOutput: {csv_path}")


if __name__ == '__main__':
    main()
