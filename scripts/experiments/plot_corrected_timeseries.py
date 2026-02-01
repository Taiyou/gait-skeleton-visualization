#!/usr/bin/env python3
"""
Plot Corrected X, Y Time Series

Shows only the final corrected data for both datasets.
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


def apply_aggressive_alignment(data, original_data, frame_rate=60, pelvis_index=0):
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


def process_type02_01(input_path, frame_rate=60):
    """Process type02_01 with baseline correction only."""
    print(f"\nProcessing type02_01: {input_path.name}")

    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    params = SmoothPCAParams(
        window_seconds=30.0,
        sample_interval_seconds=5.0,
        smoothing_factor=0.1,
        frame_rate=frame_rate,
    )

    corrected, _, _ = apply_smooth_pca_correction(
        data.positions,
        skip_start_seconds=5.0,
        skip_end_seconds=5.0,
        params=params,
    )
    corrected = apply_full_drift_correction(corrected, frame_rate=frame_rate)

    return {
        'name': 'type02_01',
        'original': original_data,
        'corrected': corrected,
        'frame_rate': frame_rate,
        'method': 'Baseline (Smooth PCA + Drift)',
    }


def process_type02_02(input_path, frame_rate=60):
    """Process type02_02 with baseline + aggressive correction."""
    print(f"\nProcessing type02_02: {input_path.name}")

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

    corrected = apply_aggressive_alignment(baseline, original_data, frame_rate=frame_rate)

    return {
        'name': 'type02_02',
        'original': original_data,
        'corrected': corrected,
        'frame_rate': frame_rate,
        'method': 'Baseline + Aggressive Y Alignment',
    }


def plot_corrected_timeseries(results1, results2, output_path):
    """Plot corrected X, Y time series for both datasets."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 14))

    pelvis_index = 0

    # Time arrays
    n1 = len(results1['corrected'])
    n2 = len(results2['corrected'])
    fr1, fr2 = results1['frame_rate'], results2['frame_rate']

    time1 = np.arange(n1) / fr1
    time2 = np.arange(n2) / fr2

    # Subsample for plotting
    step1 = max(1, n1 // 8000)
    step2 = max(1, n2 // 8000)

    t1, t2 = time1[::step1], time2[::step2]
    c1 = results1['corrected'][::step1]
    c2 = results2['corrected'][::step2]

    # ===== Row 0: X coordinate =====
    ax = axes[0, 0]
    ax.plot(t1, c1[:, pelvis_index, 0], 'b-', linewidth=0.5)
    ax.set_title(f"type02_01 - X Coordinate (Corrected)\nMethod: {results1['method']}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X (m)')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t2, c2[:, pelvis_index, 0], 'b-', linewidth=0.5)
    ax.set_title(f"type02_02 - X Coordinate (Corrected)\nMethod: {results2['method']}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X (m)')
    ax.grid(True, alpha=0.3)

    # ===== Row 1: Y coordinate =====
    ax = axes[1, 0]
    ax.plot(t1, c1[:, pelvis_index, 1], 'r-', linewidth=0.5)
    y_range1 = np.max(results1['corrected'][:, pelvis_index, 1]) - np.min(results1['corrected'][:, pelvis_index, 1])
    ax.set_title(f"type02_01 - Y Coordinate (Corrected)\nY range: {y_range1:.2f}m")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t2, c2[:, pelvis_index, 1], 'r-', linewidth=0.5)
    y_range2 = np.max(results2['corrected'][:, pelvis_index, 1]) - np.min(results2['corrected'][:, pelvis_index, 1])
    ax.set_title(f"type02_02 - Y Coordinate (Corrected)\nY range: {y_range2:.2f}m")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    # ===== Row 2: Zoomed Y (first 120 seconds) =====
    zoom1 = min(120 * fr1, n1)
    zoom2 = min(120 * fr2, n2)

    ax = axes[2, 0]
    ax.plot(time1[:zoom1], results1['corrected'][:zoom1, pelvis_index, 1], 'r-', linewidth=0.8)
    ax.set_title("type02_01 - Y Coordinate (First 120s)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(time2[:zoom2], results2['corrected'][:zoom2, pelvis_index, 1], 'r-', linewidth=0.8)
    ax.set_title("type02_02 - Y Coordinate (First 120s)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    # ===== Row 3: Trajectory (X vs Y) =====
    ax = axes[3, 0]
    ax.plot(results1['corrected'][:, pelvis_index, 0],
            results1['corrected'][:, pelvis_index, 1],
            'b-', alpha=0.5, linewidth=0.3)
    ax.set_title(f"type02_01 - Corrected Trajectory\nY range: {y_range1:.2f}m")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.plot(results2['corrected'][:, pelvis_index, 0],
            results2['corrected'][:, pelvis_index, 1],
            'b-', alpha=0.5, linewidth=0.3)
    ax.set_title(f"type02_02 - Corrected Trajectory\nY range: {y_range2:.2f}m")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved to {output_path}")


def main():
    print("=" * 70)
    print("Corrected Data - X, Y Time Series")
    print("=" * 70)

    path1 = Path("data/type2/type02_01/trial-001_sub012.xlsx")
    path2 = Path("data/type2/type02_02/NCC24-001.xlsx")

    results1 = process_type02_01(path1)
    results2 = process_type02_02(path2)

    output_path = Path("data/type2/corrected_xy_timeseries.png")
    plot_corrected_timeseries(results1, results2, output_path)

    # Print statistics
    print("\n" + "=" * 70)
    print("CORRECTED DATA STATISTICS")
    print("=" * 70)

    for name, results in [("type02_01", results1), ("type02_02", results2)]:
        c = results['corrected']
        fr = results['frame_rate']
        print(f"\n{name}:")
        print(f"  Method: {results['method']}")
        print(f"  Duration: {len(c) / fr:.1f}s")
        print(f"  X: min={np.min(c[:, 0, 0]):.2f}, max={np.max(c[:, 0, 0]):.2f}, range={np.max(c[:, 0, 0]) - np.min(c[:, 0, 0]):.2f}m")
        print(f"  Y: min={np.min(c[:, 0, 1]):.2f}, max={np.max(c[:, 0, 1]):.2f}, range={np.max(c[:, 0, 1]) - np.min(c[:, 0, 1]):.2f}m")


if __name__ == '__main__':
    main()
