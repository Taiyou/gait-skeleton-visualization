#!/usr/bin/env python3
"""
Plot X, Y Time Series for Both Datasets

Shows the X and Y coordinates over time for original and corrected data.
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


def apply_aggressive_alignment(
    data: np.ndarray,
    original_data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """Remove linear Y trend within each segment."""
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    ref_pelvis_x = original_data[:, pelvis_index, 0]
    turnarounds = detect_turnarounds_adaptive(ref_pelvis_x, frame_rate=frame_rate)

    y_correction = np.zeros(n_frames)

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        if end - start < 10:
            continue

        seg_y = pelvis_y[start:end + 1]
        seg_x = pelvis_x[start:end + 1]

        if len(seg_x) > 10 and (np.max(seg_x) - np.min(seg_x)) > 0.5:
            coeffs = np.polyfit(seg_x, seg_y, 1)
            linear_trend = np.polyval(coeffs, seg_x)
            residual = seg_y - linear_trend
            median_residual = np.median(residual)
            y_correction[start:end + 1] = linear_trend + median_residual
        else:
            y_correction[start:end + 1] = np.median(seg_y)

    smooth_window = int(0.3 * frame_rate)
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_correction_smooth = uniform_filter1d(y_correction, size=smooth_window, mode='nearest')

    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_correction_smooth

    return corrected


def process_dataset(input_path: Path, frame_rate: int = 60):
    """Process a dataset through the correction pipeline."""
    print(f"\nProcessing: {input_path.name}")

    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

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

    aggressive = apply_aggressive_alignment(baseline, original_data, frame_rate=frame_rate)

    return {
        'name': input_path.stem,
        'original': original_data,
        'baseline': baseline,
        'aggressive': aggressive,
        'frame_rate': frame_rate,
    }


def plot_xy_timeseries(results1, results2, output_path: Path):
    """Create X, Y time series plots for both datasets."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))

    pelvis_index = 0
    fr1 = results1['frame_rate']
    fr2 = results2['frame_rate']

    # Time in seconds
    n1 = len(results1['original'])
    n2 = len(results2['original'])
    time1 = np.arange(n1) / fr1
    time2 = np.arange(n2) / fr2

    # Subsample for plotting
    step1 = max(1, n1 // 10000)
    step2 = max(1, n2 // 10000)

    t1 = time1[::step1]
    t2 = time2[::step2]

    # ===== type02_01 =====
    # Row 0: X coordinate
    ax = axes[0, 0]
    ax.plot(t1, results1['original'][::step1, pelvis_index, 0], 'b-', alpha=0.7, linewidth=0.5, label='Original')
    ax.plot(t1, results1['baseline'][::step1, pelvis_index, 0], 'g-', alpha=0.7, linewidth=0.5, label='Baseline')
    ax.plot(t1, results1['aggressive'][::step1, pelvis_index, 0], 'r-', alpha=0.7, linewidth=0.5, label='Aggressive')
    ax.set_title('type02_01 - X Coordinate over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 1: Y coordinate
    ax = axes[1, 0]
    ax.plot(t1, results1['original'][::step1, pelvis_index, 1], 'b-', alpha=0.7, linewidth=0.5, label='Original')
    ax.plot(t1, results1['baseline'][::step1, pelvis_index, 1], 'g-', alpha=0.7, linewidth=0.5, label='Baseline')
    ax.plot(t1, results1['aggressive'][::step1, pelvis_index, 1], 'r-', alpha=0.7, linewidth=0.5, label='Aggressive')
    ax.set_title('type02_01 - Y Coordinate over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # ===== type02_02 =====
    # Row 0: X coordinate
    ax = axes[0, 1]
    ax.plot(t2, results2['original'][::step2, pelvis_index, 0], 'b-', alpha=0.7, linewidth=0.5, label='Original')
    ax.plot(t2, results2['baseline'][::step2, pelvis_index, 0], 'g-', alpha=0.7, linewidth=0.5, label='Baseline')
    ax.plot(t2, results2['aggressive'][::step2, pelvis_index, 0], 'r-', alpha=0.7, linewidth=0.5, label='Aggressive')
    ax.set_title('type02_02 - X Coordinate over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 1: Y coordinate
    ax = axes[1, 1]
    ax.plot(t2, results2['original'][::step2, pelvis_index, 1], 'b-', alpha=0.7, linewidth=0.5, label='Original')
    ax.plot(t2, results2['baseline'][::step2, pelvis_index, 1], 'g-', alpha=0.7, linewidth=0.5, label='Baseline')
    ax.plot(t2, results2['aggressive'][::step2, pelvis_index, 1], 'r-', alpha=0.7, linewidth=0.5, label='Aggressive')
    ax.set_title('type02_02 - Y Coordinate over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 2: Zoomed view (first 60 seconds)
    zoom_end1 = min(60 * fr1, n1)
    zoom_end2 = min(60 * fr2, n2)
    step_zoom = 1

    tz1 = time1[:zoom_end1:step_zoom]
    tz2 = time2[:zoom_end2:step_zoom]

    ax = axes[2, 0]
    ax.plot(tz1, results1['original'][:zoom_end1:step_zoom, pelvis_index, 1], 'b-', alpha=0.7, linewidth=0.8, label='Original')
    ax.plot(tz1, results1['baseline'][:zoom_end1:step_zoom, pelvis_index, 1], 'g-', alpha=0.7, linewidth=0.8, label='Baseline')
    ax.plot(tz1, results1['aggressive'][:zoom_end1:step_zoom, pelvis_index, 1], 'r-', alpha=0.7, linewidth=0.8, label='Aggressive')
    ax.set_title('type02_01 - Y Coordinate (First 60s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(tz2, results2['original'][:zoom_end2:step_zoom, pelvis_index, 1], 'b-', alpha=0.7, linewidth=0.8, label='Original')
    ax.plot(tz2, results2['baseline'][:zoom_end2:step_zoom, pelvis_index, 1], 'g-', alpha=0.7, linewidth=0.8, label='Baseline')
    ax.plot(tz2, results2['aggressive'][:zoom_end2:step_zoom, pelvis_index, 1], 'r-', alpha=0.7, linewidth=0.8, label='Aggressive')
    ax.set_title('type02_02 - Y Coordinate (First 60s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 3: Trajectory (X vs Y)
    ax = axes[3, 0]
    ax.plot(results1['original'][:, pelvis_index, 0], results1['original'][:, pelvis_index, 1],
            'b-', alpha=0.3, linewidth=0.3, label='Original')
    ax.plot(results1['aggressive'][:, pelvis_index, 0], results1['aggressive'][:, pelvis_index, 1],
            'r-', alpha=0.6, linewidth=0.3, label='Corrected')
    ax.set_title('type02_01 - Trajectory (X vs Y)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.plot(results2['original'][:, pelvis_index, 0], results2['original'][:, pelvis_index, 1],
            'b-', alpha=0.3, linewidth=0.3, label='Original')
    ax.plot(results2['aggressive'][:, pelvis_index, 0], results2['aggressive'][:, pelvis_index, 1],
            'r-', alpha=0.6, linewidth=0.3, label='Corrected')
    ax.set_title('type02_02 - Trajectory (X vs Y)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


def print_statistics(results, name):
    """Print statistics for a dataset."""
    pelvis_index = 0
    fr = results['frame_rate']

    orig = results['original']
    base = results['baseline']
    aggr = results['aggressive']

    print(f"\n{name}:")
    print(f"  Duration: {len(orig) / fr:.1f} seconds ({len(orig)} frames)")
    print(f"  Frame rate: {fr} Hz")

    print(f"\n  X Coordinate:")
    print(f"    Original:   min={np.min(orig[:, pelvis_index, 0]):.2f}, max={np.max(orig[:, pelvis_index, 0]):.2f}, range={np.max(orig[:, pelvis_index, 0]) - np.min(orig[:, pelvis_index, 0]):.2f}m")
    print(f"    Baseline:   min={np.min(base[:, pelvis_index, 0]):.2f}, max={np.max(base[:, pelvis_index, 0]):.2f}, range={np.max(base[:, pelvis_index, 0]) - np.min(base[:, pelvis_index, 0]):.2f}m")
    print(f"    Aggressive: min={np.min(aggr[:, pelvis_index, 0]):.2f}, max={np.max(aggr[:, pelvis_index, 0]):.2f}, range={np.max(aggr[:, pelvis_index, 0]) - np.min(aggr[:, pelvis_index, 0]):.2f}m")

    print(f"\n  Y Coordinate:")
    print(f"    Original:   min={np.min(orig[:, pelvis_index, 1]):.2f}, max={np.max(orig[:, pelvis_index, 1]):.2f}, range={np.max(orig[:, pelvis_index, 1]) - np.min(orig[:, pelvis_index, 1]):.2f}m")
    print(f"    Baseline:   min={np.min(base[:, pelvis_index, 1]):.2f}, max={np.max(base[:, pelvis_index, 1]):.2f}, range={np.max(base[:, pelvis_index, 1]) - np.min(base[:, pelvis_index, 1]):.2f}m")
    print(f"    Aggressive: min={np.min(aggr[:, pelvis_index, 1]):.2f}, max={np.max(aggr[:, pelvis_index, 1]):.2f}, range={np.max(aggr[:, pelvis_index, 1]) - np.min(aggr[:, pelvis_index, 1]):.2f}m")


def main():
    print("=" * 70)
    print("X, Y Time Series Visualization")
    print("=" * 70)

    path1 = Path("data/type2/type02_01/trial-001_sub012.xlsx")
    path2 = Path("data/type2/type02_02/NCC24-001.xlsx")

    results1 = process_dataset(path1)
    results2 = process_dataset(path2)

    output_path = Path("data/type2/xy_timeseries_comparison.png")
    plot_xy_timeseries(results1, results2, output_path)

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print_statistics(results1, "type02_01 (trial-001_sub012)")
    print_statistics(results2, "type02_02 (NCC24-001)")


if __name__ == '__main__':
    main()
