#!/usr/bin/env python3
"""
Apply Strong Y Correction for Diagonal Offset Removal

This script applies aggressive Y-axis correction to eliminate
diagonal offset between walking passes.

Strategy:
1. Apply baseline correction (Smooth PCA + Drift)
2. Detect turnarounds
3. For each segment, flatten Y to its median value
4. Result: All walking passes aligned to same Y baseline
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction, SmoothPCAParams
from scripts.gait_correction.drift_removal import apply_full_drift_correction
from scripts.gait_correction.turnaround import detect_turnarounds_adaptive
from scripts.gait_correction.export import export_to_csv, export_to_excel


def apply_strong_y_correction(
    data: np.ndarray,
    original_data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """
    Apply strong Y correction to flatten each segment.

    For each walking segment:
    - Calculate the linear trend of Y coordinates
    - Remove the trend to make each segment horizontal
    - Align all segments to Y=0
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Detect turnarounds using original data
    ref_pelvis_x = original_data[:, pelvis_index, 0]
    turnarounds = detect_turnarounds_adaptive(ref_pelvis_x, frame_rate=frame_rate)

    print(f"Detected {len(turnarounds.segments)} segments")

    # Global baseline: Y=0
    target_y = 0.0

    # Build Y correction for each frame
    y_correction = np.zeros(n_frames)

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        seg_length = end - start + 1
        if seg_length < 10:
            continue

        # Get segment Y values
        seg_y = pelvis_y[start:end + 1]
        seg_x = pelvis_x[start:end + 1]

        # Calculate linear trend (Y = a*X + b)
        # This captures the diagonal drift within the segment
        if len(seg_x) > 2:
            coeffs = np.polyfit(seg_x - seg_x[0], seg_y, 1)
            trend = np.polyval(coeffs, seg_x - seg_x[0])
        else:
            trend = seg_y

        # Calculate median Y (after removing trend)
        detrended_y = seg_y - trend
        median_detrend = np.median(detrended_y)

        # Correction = trend + (median - target)
        # This removes both the diagonal drift and the Y offset
        segment_baseline = np.median(seg_y)
        correction = seg_y - target_y

        y_correction[start:end + 1] = correction

    # Smooth corrections at segment boundaries
    smooth_window = int(0.3 * frame_rate)
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_correction_smooth = uniform_filter1d(y_correction, size=smooth_window, mode='nearest')

    # Apply correction
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_correction_smooth

    return corrected


def apply_detrend_per_segment(
    data: np.ndarray,
    original_data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """
    Remove linear Y trend within each segment and align to Y=0.
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Detect turnarounds
    ref_pelvis_x = original_data[:, pelvis_index, 0]
    turnarounds = detect_turnarounds_adaptive(ref_pelvis_x, frame_rate=frame_rate)

    # Process each segment
    y_correction = np.zeros(n_frames)

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        if end - start < 10:
            continue

        seg_y = pelvis_y[start:end + 1]
        seg_frames = np.arange(end - start + 1)

        # Fit linear trend
        if len(seg_frames) > 2:
            coeffs = np.polyfit(seg_frames, seg_y, 1)
            trend = np.polyval(coeffs, seg_frames)

            # Remove trend and center at Y=0
            detrended = seg_y - trend
            median_y = np.median(seg_y)

            # Total correction: remove original Y, keep only detrended oscillation around 0
            y_correction[start:end + 1] = seg_y - np.median(detrended)
        else:
            y_correction[start:end + 1] = seg_y

    # Smooth at boundaries
    smooth_window = int(0.2 * frame_rate)
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_correction_smooth = uniform_filter1d(y_correction, size=smooth_window, mode='nearest')

    # Apply
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_correction_smooth

    return corrected


def plot_comparison(
    original: np.ndarray,
    baseline: np.ndarray,
    final: np.ndarray,
    output_path: Path,
    pelvis_index: int = 0,
):
    """Create comparison plot."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original
    ax = axes[0]
    ax.plot(original[:, pelvis_index, 0], original[:, pelvis_index, 1],
            'b-', alpha=0.5, linewidth=0.3)
    y_range = np.max(original[:, pelvis_index, 1]) - np.min(original[:, pelvis_index, 1])
    ax.set_title(f'Original\nY range: {y_range:.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # After baseline
    ax = axes[1]
    ax.plot(baseline[:, pelvis_index, 0], baseline[:, pelvis_index, 1],
            'g-', alpha=0.5, linewidth=0.3)
    y_range = np.max(baseline[:, pelvis_index, 1]) - np.min(baseline[:, pelvis_index, 1])
    ax.set_title(f'After Baseline\nY range: {y_range:.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Final
    ax = axes[2]
    ax.plot(final[:, pelvis_index, 0], final[:, pelvis_index, 1],
            'r-', alpha=0.5, linewidth=0.3)
    y_range = np.max(final[:, pelvis_index, 1]) - np.min(final[:, pelvis_index, 1])
    ax.set_title(f'After Strong Y Correction\nY range: {y_range:.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Overlay
    ax = axes[3]
    ax.plot(original[:, pelvis_index, 0], original[:, pelvis_index, 1],
            'b-', alpha=0.2, linewidth=0.3, label='Original')
    ax.plot(final[:, pelvis_index, 0], final[:, pelvis_index, 1],
            'r-', alpha=0.6, linewidth=0.3, label='Final')
    ax.set_title('Comparison')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    input_path = Path("data/type2/type02_02/NCC24-001.xlsx")
    output_dir = input_path.parent
    frame_rate = 60

    print("=" * 70)
    print("Strong Y Correction for Diagonal Offset Removal")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Baseline correction
    print("\n[2/5] Applying baseline correction...")
    params = SmoothPCAParams(
        window_seconds=30.0,
        sample_interval_seconds=5.0,
        smoothing_factor=0.1,
        frame_rate=frame_rate,
    )

    baseline, _, _ = apply_smooth_pca_correction(
        data.positions,
        skip_start_seconds=5.0,
        skip_end_seconds=5.0,
        params=params,
    )

    baseline = apply_full_drift_correction(baseline, frame_rate=frame_rate)

    # Strong Y correction
    print("\n[3/5] Applying strong Y correction...")
    final = apply_detrend_per_segment(
        baseline,
        original_data,
        frame_rate=frame_rate,
    )

    # Export
    print("\n[4/5] Exporting results...")
    stem = input_path.stem

    csv_path = output_dir / f"{stem}_strong_corrected.csv"
    xlsx_path = output_dir / f"{stem}_strong_corrected.xlsx"
    plot_path = output_dir / f"{stem}_strong_comparison.png"

    export_to_csv(final, csv_path, data.segment_names, frame_rate=frame_rate)
    export_to_excel(final, xlsx_path, data.segment_names, frame_rate=frame_rate)

    # Plot
    print("\n[5/5] Generating visualization...")
    plot_comparison(original_data, baseline, final, plot_path)

    # Summary
    orig_y = np.max(original_data[:, 0, 1]) - np.min(original_data[:, 0, 1])
    base_y = np.max(baseline[:, 0, 1]) - np.min(baseline[:, 0, 1])
    final_y = np.max(final[:, 0, 1]) - np.min(final[:, 0, 1])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original Y range:     {orig_y:.2f}m")
    print(f"After baseline:       {base_y:.2f}m ({(1-base_y/orig_y)*100:.1f}% reduction)")
    print(f"After strong Y corr:  {final_y:.2f}m ({(1-final_y/orig_y)*100:.1f}% total reduction)")
    print(f"\nOutput: {csv_path}")


if __name__ == '__main__':
    main()
