#!/usr/bin/env python3
"""
Adaptive Drift Correction

Addresses time-dependent drift that starts small and grows over time.

Strategy:
1. Use initial period (first 1-2 minutes) as reference for "correct" Y behavior
2. Track drift accumulation over time
3. Apply time-varying correction that increases as drift accumulates

This preserves natural walking patterns while removing cumulative IMU drift.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.export import export_to_csv
from scripts.gait_correction.turnaround import detect_turnarounds_adaptive
from scripts.utils.config import setup_matplotlib, GaitCorrectionConfig
from scripts.utils.plotting import (
    plot_multi_method_comparison,
    calc_range,
    PlotConfig,
)

# Setup matplotlib backend
setup_matplotlib()
import matplotlib.pyplot as plt


def apply_adaptive_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    reference_seconds: float = 60.0,
    drift_window_seconds: float = 10.0,
) -> tuple[np.ndarray, dict]:
    """
    Apply adaptive drift correction.

    Strategy:
    1. Global PCA rotation to align walking direction with X axis
    2. Use initial period to establish baseline Y behavior
    3. Estimate cumulative drift by comparing current Y to expected Y
    4. Apply time-varying correction

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        reference_seconds: Duration of reference period
        drift_window_seconds: Window for drift smoothing

    Returns:
        Tuple of (corrected_data, info_dict)
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape
    info = {}

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Step 1: Global PCA rotation
    print("\n[Step 1] Global PCA rotation...")

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

    print(f"  Rotated by {np.degrees(-global_angle):.2f}°")
    info['rotation_angle'] = np.degrees(-global_angle)

    # Step 2: Establish reference Y behavior from initial period
    print(f"\n[Step 2] Establishing reference from first {reference_seconds}s...")

    ref_frames = int(reference_seconds * frame_rate)
    pelvis_y_rotated = corrected[:, pelvis_index, 1]

    ref_y = pelvis_y_rotated[:ref_frames]
    ref_median = np.median(ref_y)
    ref_std = np.std(ref_y)
    ref_range = ref_y.max() - ref_y.min()

    print(f"  Reference Y: median={ref_median:.3f}m, std={ref_std:.3f}m, range={ref_range:.2f}m")
    info['ref_median'] = ref_median
    info['ref_std'] = ref_std
    info['ref_range'] = ref_range

    # Step 3: Estimate cumulative drift over time
    print("\n[Step 3] Estimating cumulative drift...")

    window_frames = int(drift_window_seconds * frame_rate)
    if window_frames % 2 == 0:
        window_frames += 1

    y_drift = uniform_filter1d(pelvis_y_rotated, size=window_frames, mode='nearest')
    drift_from_ref = y_drift - ref_median

    print(f"  Max drift from reference: {np.max(np.abs(drift_from_ref)):.2f}m")
    info['max_drift'] = np.max(np.abs(drift_from_ref))

    # Step 4: Apply adaptive correction
    print("\n[Step 4] Applying adaptive correction...")

    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= drift_from_ref

    final_median = np.median(corrected[:, pelvis_index, 1])
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= final_median

    final_y = corrected[:, pelvis_index, 1]
    final_range = final_y.max() - final_y.min()

    print(f"  Final Y range: {final_range:.2f}m")
    info['final_range'] = final_range

    return corrected, info


def apply_turnaround_based_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Turnaround-based correction: align each walking pass to same Y baseline.

    This method:
    1. Detects turnaround points
    2. For each walking segment, calculates median Y
    3. Aligns all segments to have same median Y
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape
    info = {}

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # Step 1: Global PCA rotation
    print("\n[Step 1] Global PCA rotation...")

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

    print(f"  Rotated by {np.degrees(-global_angle):.2f}°")

    # Step 2: Detect turnarounds
    print("\n[Step 2] Detecting turnarounds...")

    pelvis_x_rotated = corrected[:, pelvis_index, 0]
    pelvis_y_rotated = corrected[:, pelvis_index, 1]

    turnarounds = detect_turnarounds_adaptive(pelvis_x_rotated, frame_rate=frame_rate)
    print(f"  Found {len(turnarounds.segments)} walking segments")
    info['n_segments'] = len(turnarounds.segments)

    # Step 3: Align each segment's median Y to common baseline
    print("\n[Step 3] Aligning segments...")

    first_seg = turnarounds.segments[0]
    ref_median = np.median(pelvis_y_rotated[first_seg[0]:first_seg[1]+1])

    y_correction = np.zeros(n_frames)

    for seg_idx, (start, end) in enumerate(turnarounds.segments):
        if end - start < 10:
            continue

        seg_y = pelvis_y_rotated[start:end+1]
        seg_median = np.median(seg_y)
        offset = seg_median - ref_median
        y_correction[start:end+1] = offset

    smooth_window = int(0.5 * frame_rate)
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_correction_smooth = uniform_filter1d(y_correction, size=smooth_window, mode='nearest')

    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_correction_smooth

    final_median = np.median(corrected[:, pelvis_index, 1])
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= final_median

    final_y = corrected[:, pelvis_index, 1]
    info['final_range'] = final_y.max() - final_y.min()

    print(f"  Final Y range: {info['final_range']:.2f}m")

    return corrected, info


def plot_comparison(
    original: np.ndarray,
    adaptive: np.ndarray,
    turnaround: np.ndarray,
    v7: np.ndarray,
    output_path: Path,
    frame_rate: int = 60,
    pelvis_index: int = 0,
):
    """Create comparison plot using shared utilities."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    n_frames = len(original)
    time = np.arange(n_frames) / frame_rate / 60

    methods = [
        ('Original', original),
        ('V7 (High-pass)', v7),
        ('Adaptive (Time-based)', adaptive),
        ('Turnaround-based', turnaround),
    ]

    # Row 1: Trajectories
    for i, (name, data) in enumerate(methods):
        ax = axes[0, i]
        x = data[:, pelvis_index, 0]
        y = data[:, pelvis_index, 1]

        scatter = ax.scatter(x*1000, y*1000, c=time, cmap='viridis', s=0.5, alpha=0.5)
        ax.plot(x[0]*1000, y[0]*1000, 'go', markersize=8, label='Start')
        ax.plot(x[-1]*1000, y[-1]*1000, 'rs', markersize=8, label='End')

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        ax.set_title(f'{name}\nX: {x_range:.2f}m, Y: {y_range:.2f}m')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)

    # Row 2: Y over time
    for i, (name, data) in enumerate(methods):
        ax = axes[1, i]
        y = data[:, pelvis_index, 1]

        ax.plot(time, y, linewidth=0.5)
        ax.set_title(f'{name} - Y over Time')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

        # Show Y range per minute as bars
        ax2 = ax.twinx()
        segment_duration = 60 * frame_rate
        n_segs = n_frames // segment_duration
        y_ranges = []
        for j in range(n_segs):
            start = j * segment_duration
            end = (j + 1) * segment_duration
            seg_y = y[start:end]
            y_ranges.append(seg_y.max() - seg_y.min())
        ax2.bar(np.arange(n_segs) + 0.5, y_ranges, width=0.8, alpha=0.3, color='red')
        ax2.set_ylabel('Y Range/min (m)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved: {output_path}")


def main():
    input_path = Path("data/type2/type02_02/NCC24-001.xlsx")
    output_dir = input_path.parent
    frame_rate = 60

    print("=" * 70)
    print("Adaptive Drift Correction for Time-Dependent Drift")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Load V7 for comparison
    v7_df = pd.read_csv(output_dir / "NCC24-001_v7_final.csv")
    v7_data = original_data.copy()
    v7_data[:, 0, 0] = v7_df['Pelvis_X'].values
    v7_data[:, 0, 1] = v7_df['Pelvis_Y'].values

    # Apply adaptive correction
    print("\n[2/5] Applying adaptive (time-based) correction...")
    adaptive_data, adaptive_info = apply_adaptive_correction(
        original_data.copy(),
        frame_rate=frame_rate,
        reference_seconds=60.0,
        drift_window_seconds=10.0,
    )

    # Apply turnaround-based correction
    print("\n[3/5] Applying turnaround-based correction...")
    turnaround_data, turnaround_info = apply_turnaround_based_correction(
        original_data.copy(),
        frame_rate=frame_rate,
    )

    # Export
    print("\n[4/5] Exporting results...")
    stem = input_path.stem

    export_to_csv(adaptive_data, output_dir / f"{stem}_adaptive.csv",
                  data.segment_names, frame_rate=frame_rate)
    export_to_csv(turnaround_data, output_dir / f"{stem}_turnaround.csv",
                  data.segment_names, frame_rate=frame_rate)

    # Plot
    print("\n[5/5] Generating comparison...")
    plot_path = output_dir / f"{stem}_adaptive_comparison.png"
    plot_comparison(original_data, adaptive_data, turnaround_data, v7_data,
                    plot_path, frame_rate=frame_rate)

    # Summary
    orig_y = original_data[:, 0, 1].max() - original_data[:, 0, 1].min()
    v7_y = v7_data[:, 0, 1].max() - v7_data[:, 0, 1].min()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOriginal:           Y range = {orig_y:.2f}m")
    print(f"V7 (High-pass):     Y range = {v7_y:.2f}m ({(1-v7_y/orig_y)*100:.1f}% reduction)")
    print(f"Adaptive:           Y range = {adaptive_info['final_range']:.2f}m ({(1-adaptive_info['final_range']/orig_y)*100:.1f}% reduction)")
    print(f"Turnaround-based:   Y range = {turnaround_info['final_range']:.2f}m ({(1-turnaround_info['final_range']/orig_y)*100:.1f}% reduction)")

    # Check Y range consistency over time
    print("\n--- Y Range per Minute (consistency check) ---")
    segment_duration = 60 * frame_rate
    n_segs = len(original_data) // segment_duration

    print(f"{'Time':<10} {'Original':<12} {'V7':<12} {'Adaptive':<12} {'Turnaround':<12}")
    print("-" * 58)

    for i in range(n_segs):
        start = i * segment_duration
        end = (i + 1) * segment_duration

        orig_r = original_data[start:end, 0, 1].max() - original_data[start:end, 0, 1].min()
        v7_r = v7_data[start:end, 0, 1].max() - v7_data[start:end, 0, 1].min()
        adap_r = adaptive_data[start:end, 0, 1].max() - adaptive_data[start:end, 0, 1].min()
        turn_r = turnaround_data[start:end, 0, 1].max() - turnaround_data[start:end, 0, 1].min()

        print(f"{i}-{i+1}min     {orig_r:>6.2f}m      {v7_r:>6.2f}m      {adap_r:>6.2f}m      {turn_r:>6.2f}m")


if __name__ == '__main__':
    main()
