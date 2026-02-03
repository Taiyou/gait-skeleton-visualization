#!/usr/bin/env python3
"""
Apply Balanced Y Correction

Removes diagonal offset between walking passes while preserving
natural Y-axis oscillation (body sway during walking).

Strategy:
1. Apply baseline correction (Smooth PCA + Drift)
2. Detect turnarounds
3. For each segment:
   - Calculate median Y as segment baseline
   - Shift segment so median Y = 0 (global baseline)
   - Preserve Y oscillation within segment
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction, SmoothPCAParams
from scripts.gait_correction.drift_removal import apply_full_drift_correction
from scripts.gait_correction.turnaround import detect_turnarounds_adaptive
from scripts.gait_correction.export import export_to_csv, export_to_excel
from scripts.utils.config import setup_matplotlib, GaitCorrectionConfig
from scripts.utils.plotting import (
    plot_multi_method_comparison,
    plot_time_series_comparison,
    calc_range,
    PlotConfig,
)

# Setup matplotlib backend
setup_matplotlib()


def apply_balanced_y_correction(
    data: np.ndarray,
    original_data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """
    Apply balanced Y correction.

    - Removes offset between segments (diagonal drift)
    - Preserves natural Y oscillation within each segment
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_y = data[:, pelvis_index, 1]

    # Detect turnarounds using original data
    ref_pelvis_x = original_data[:, pelvis_index, 0]
    turnarounds = detect_turnarounds_adaptive(ref_pelvis_x, frame_rate=frame_rate)

    print(f"Processing {len(turnarounds.segments)} segments")

    # Global baseline: median Y of entire trajectory
    global_median = np.median(pelvis_y)

    # Build Y correction: shift each segment's median to global median
    y_offset = np.zeros(n_frames)

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        if end - start < 10:
            continue

        seg_y = pelvis_y[start:end + 1]

        # Segment median
        seg_median = np.median(seg_y)

        # Offset to align segment median with global median (then to 0)
        offset = seg_median - global_median
        y_offset[start:end + 1] = offset

    # Smooth transitions at segment boundaries
    smooth_window = int(0.5 * frame_rate)
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_offset_smooth = uniform_filter1d(y_offset, size=smooth_window, mode='nearest')

    # Apply offset (shift each segment to common baseline)
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_offset_smooth

    # Center entire trajectory at Y=0
    final_median = np.median(corrected[:, pelvis_index, 1])
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= final_median

    return corrected


def apply_aggressive_alignment(
    data: np.ndarray,
    original_data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """
    More aggressive alignment: remove linear drift within each segment.
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Detect turnarounds
    ref_pelvis_x = original_data[:, pelvis_index, 0]
    turnarounds = detect_turnarounds_adaptive(ref_pelvis_x, frame_rate=frame_rate)

    # Build correction
    y_correction = np.zeros(n_frames)

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        if end - start < 10:
            continue

        seg_y = pelvis_y[start:end + 1]
        seg_x = pelvis_x[start:end + 1]

        # Fit linear trend Y vs X (diagonal drift)
        if len(seg_x) > 10 and (np.max(seg_x) - np.min(seg_x)) > 0.5:
            coeffs = np.polyfit(seg_x, seg_y, 1)
            linear_trend = np.polyval(coeffs, seg_x)

            # Remove only the linear drift, keep residual oscillation
            residual = seg_y - linear_trend
            median_residual = np.median(residual)

            # Correction shifts segment so residual median = 0
            y_correction[start:end + 1] = linear_trend + median_residual
        else:
            # Short segment: just align median to 0
            y_correction[start:end + 1] = np.median(seg_y)

    # Smooth
    smooth_window = int(0.3 * frame_rate)
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_correction_smooth = uniform_filter1d(y_correction, size=smooth_window, mode='nearest')

    # Apply
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_correction_smooth

    return corrected


def main():
    input_path = Path("data/type2/type02_02/NCC24-001.xlsx")
    output_dir = input_path.parent
    frame_rate = 60
    config = GaitCorrectionConfig(frame_rate=frame_rate)

    print("=" * 70)
    print("Balanced Y Correction for Diagonal Offset Removal")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Baseline correction
    print("\n[2/6] Applying baseline correction...")
    params = SmoothPCAParams(
        window_seconds=config.pca_window_seconds,
        sample_interval_seconds=config.pca_sample_interval_seconds,
        smoothing_factor=config.pca_smoothing_factor,
        frame_rate=frame_rate,
    )

    baseline, _, _ = apply_smooth_pca_correction(
        data.positions,
        skip_start_seconds=config.skip_start_seconds,
        skip_end_seconds=config.skip_end_seconds,
        params=params,
    )

    baseline = apply_full_drift_correction(baseline, frame_rate=frame_rate)

    # Balanced Y correction
    print("\n[3/6] Applying balanced Y correction...")
    balanced = apply_balanced_y_correction(
        baseline, original_data, frame_rate=frame_rate
    )

    # Aggressive alignment
    print("\n[4/6] Applying aggressive alignment...")
    aggressive = apply_aggressive_alignment(
        baseline, original_data, frame_rate=frame_rate
    )

    # Export balanced result
    print("\n[5/6] Exporting results...")
    stem = input_path.stem

    csv_path = output_dir / f"{stem}_balanced_corrected.csv"
    xlsx_path = output_dir / f"{stem}_balanced_corrected.xlsx"
    plot_path = output_dir / f"{stem}_balanced_comparison.png"

    export_to_csv(balanced, csv_path, data.segment_names, frame_rate=frame_rate)
    export_to_excel(balanced, xlsx_path, data.segment_names, frame_rate=frame_rate)

    # Also export aggressive
    csv_path_agg = output_dir / f"{stem}_aggressive_corrected.csv"
    export_to_csv(aggressive, csv_path_agg, data.segment_names, frame_rate=frame_rate)

    # Plot using shared utilities
    print("\n[6/6] Generating visualization...")
    plot_config = PlotConfig(frame_rate=frame_rate)
    plot_results = [
        {"name": "Balanced Y Correction", "data": balanced},
        {"name": "Aggressive Alignment", "data": aggressive},
    ]
    plot_multi_method_comparison(
        plot_results,
        original_data,
        plot_path,
        config=plot_config,
    )

    # Summary
    orig_y = calc_range(original_data, 1, config.pelvis_index)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original Y range:     {orig_y:.2f}m")
    print(f"Balanced:             {calc_range(balanced, 1):.2f}m ({(1-calc_range(balanced, 1)/orig_y)*100:.1f}% reduction)")
    print(f"Aggressive:           {calc_range(aggressive, 1):.2f}m ({(1-calc_range(aggressive, 1)/orig_y)*100:.1f}% reduction)")
    print(f"\nRecommended: Balanced correction (preserves natural sway)")
    print(f"Output: {csv_path}")


if __name__ == '__main__':
    main()
