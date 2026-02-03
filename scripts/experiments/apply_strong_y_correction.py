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
    calc_range,
    PlotConfig,
)

# Setup matplotlib backend
setup_matplotlib()


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

        # Correction shifts segment so its median = target_y
        y_correction[start:end + 1] = seg_y - target_y

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


def main():
    input_path = Path("data/type2/type02_02/NCC24-001.xlsx")
    output_dir = input_path.parent
    frame_rate = 60
    config = GaitCorrectionConfig(frame_rate=frame_rate)

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

    # Plot using shared utilities
    print("\n[5/5] Generating visualization...")
    plot_config = PlotConfig(frame_rate=frame_rate)
    plot_results = [
        {"name": "After Baseline", "data": baseline},
        {"name": "After Strong Y Correction", "data": final},
    ]
    plot_multi_method_comparison(
        plot_results,
        original_data,
        plot_path,
        config=plot_config,
    )

    # Summary
    orig_y = calc_range(original_data, 1, config.pelvis_index)
    base_y = calc_range(baseline, 1, config.pelvis_index)
    final_y = calc_range(final, 1, config.pelvis_index)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original Y range:     {orig_y:.2f}m")
    print(f"After baseline:       {base_y:.2f}m ({(1-base_y/orig_y)*100:.1f}% reduction)")
    print(f"After strong Y corr:  {final_y:.2f}m ({(1-final_y/orig_y)*100:.1f}% total reduction)")
    print(f"\nOutput: {csv_path}")


if __name__ == '__main__':
    main()
