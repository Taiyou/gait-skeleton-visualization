#!/usr/bin/env python3
"""
Segment-wise PCA Correction

Applies PCA correction per walking segment to maintain consistent
8m walking distance throughout the recording.

Problem: Global PCA doesn't account for time-varying drift, causing
later segments to have shorter apparent walking distances.

Solution: Apply PCA alignment per segment to ensure each segment
maintains the expected ~8m walking distance.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.export import export_to_csv


def detect_turnarounds(x_data, frame_rate=60, min_distance_seconds=3.0):
    """Detect turnaround points from X coordinate."""
    # Smooth X
    x_smooth = uniform_filter1d(x_data, size=int(2 * frame_rate), mode='nearest')

    min_distance = int(min_distance_seconds * frame_rate)

    # Find peaks and valleys
    peaks, _ = find_peaks(x_smooth, distance=min_distance, prominence=0.5)
    valleys, _ = find_peaks(-x_smooth, distance=min_distance, prominence=0.5)

    # Combine and sort
    turnarounds = np.sort(np.concatenate([peaks, valleys]))

    return turnarounds


def apply_segment_pca_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
    expected_distance: float = 8.0,
    highpass_window_seconds: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """
    Apply segment-wise PCA correction.

    Strategy:
    1. First pass: Global PCA to roughly align data
    2. Detect turnaround points
    3. For each segment: apply local PCA to align walking direction
    4. Scale each segment to expected distance (8m)
    5. Apply high-pass filter to Y to remove remaining drift

    Args:
        data: Position data (n_frames, n_segments, 3)
        frame_rate: Frame rate in Hz
        pelvis_index: Index of pelvis segment
        expected_distance: Expected walking distance per segment
        highpass_window_seconds: Window for Y high-pass filter

    Returns:
        Tuple of (corrected_data, info_dict)
    """
    corrected = data.copy()
    n_frames, n_segments, _ = data.shape
    info = {'segment_x_ranges': [], 'segment_scales': []}

    pelvis_x = data[:, pelvis_index, 0]
    pelvis_y = data[:, pelvis_index, 1]

    # =========================================================================
    # Step 1: Initial global PCA alignment
    # =========================================================================
    print("\n[Step 1] Initial global PCA alignment...")

    xy_data = np.column_stack([pelvis_x, pelvis_y])
    center = np.mean(xy_data, axis=0)

    pca = PCA(n_components=2)
    pca.fit(xy_data - center)

    global_angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    cos_a, sin_a = np.cos(-global_angle), np.sin(-global_angle)

    for body_idx in range(n_segments):
        x = corrected[:, body_idx, 0] - center[0]
        y = corrected[:, body_idx, 1] - center[1]
        corrected[:, body_idx, 0] = x * cos_a - y * sin_a
        corrected[:, body_idx, 1] = x * sin_a + y * cos_a

    print(f"  Global rotation: {np.degrees(-global_angle):.2f}°")

    # =========================================================================
    # Step 2: Detect turnarounds
    # =========================================================================
    print("\n[Step 2] Detecting turnarounds...")

    pelvis_x_aligned = corrected[:, pelvis_index, 0]
    turnarounds = detect_turnarounds(pelvis_x_aligned, frame_rate)

    # Add start and end
    if turnarounds[0] > 0:
        turnarounds = np.concatenate([[0], turnarounds])
    if turnarounds[-1] < n_frames - 1:
        turnarounds = np.concatenate([turnarounds, [n_frames - 1]])

    n_segs = len(turnarounds) - 1
    print(f"  Found {n_segs} walking segments")

    # =========================================================================
    # Step 3: Per-segment PCA alignment and scaling
    # =========================================================================
    print("\n[Step 3] Per-segment PCA alignment and scaling...")

    # Process each segment
    for seg_idx in range(n_segs):
        start = turnarounds[seg_idx]
        end = turnarounds[seg_idx + 1]

        if end - start < 30:  # Skip very short segments
            continue

        # Get segment data
        seg_x = corrected[start:end+1, pelvis_index, 0]
        seg_y = corrected[start:end+1, pelvis_index, 1]

        # Calculate current X range
        current_x_range = seg_x.max() - seg_x.min()
        info['segment_x_ranges'].append(current_x_range)

        # Local PCA for this segment
        seg_xy = np.column_stack([seg_x, seg_y])
        seg_center = np.mean(seg_xy, axis=0)

        pca_local = PCA(n_components=2)
        pca_local.fit(seg_xy - seg_center)

        local_angle = np.arctan2(pca_local.components_[0, 1], pca_local.components_[0, 0])

        # Only apply small corrections (avoid large rotations)
        if abs(local_angle) < np.radians(30):
            cos_l, sin_l = np.cos(-local_angle), np.sin(-local_angle)

            for body_idx in range(n_segments):
                x = corrected[start:end+1, body_idx, 0] - seg_center[0]
                y = corrected[start:end+1, body_idx, 1] - seg_center[1]
                corrected[start:end+1, body_idx, 0] = x * cos_l - y * sin_l + seg_center[0]
                corrected[start:end+1, body_idx, 1] = x * sin_l + y * cos_l + seg_center[1]

        # Recalculate X range after local PCA
        seg_x_new = corrected[start:end+1, pelvis_index, 0]
        new_x_range = seg_x_new.max() - seg_x_new.min()

        # Scale to expected distance if significantly different
        if new_x_range > 0.5 * expected_distance:  # Only scale reasonable segments
            scale = expected_distance / new_x_range

            # Limit scaling to reasonable range (0.8 to 1.3)
            scale = np.clip(scale, 0.8, 1.3)
            info['segment_scales'].append(scale)

            # Apply scaling centered on segment center
            seg_center_x = np.mean(corrected[start:end+1, pelvis_index, 0])

            for body_idx in range(n_segments):
                corrected[start:end+1, body_idx, 0] = (
                    (corrected[start:end+1, body_idx, 0] - seg_center_x) * scale + seg_center_x
                )

    # =========================================================================
    # Step 4: Smooth segment boundaries
    # =========================================================================
    print("\n[Step 4] Smoothing segment boundaries...")

    # Apply small smoothing at boundaries to avoid discontinuities
    boundary_window = int(0.5 * frame_rate)  # 0.5 second

    for i in range(1, len(turnarounds) - 1):
        t = turnarounds[i]
        start = max(0, t - boundary_window)
        end = min(n_frames, t + boundary_window)

        for body_idx in range(n_segments):
            for axis in range(3):
                corrected[start:end, body_idx, axis] = uniform_filter1d(
                    corrected[start:end, body_idx, axis],
                    size=min(boundary_window, end-start),
                    mode='nearest'
                )

    # =========================================================================
    # Step 5: High-pass filter for Y drift
    # =========================================================================
    print(f"\n[Step 5] High-pass filter for Y (window={highpass_window_seconds}s)...")

    # Linear detrend
    pelvis_y_new = corrected[:, pelvis_index, 1]
    frames = np.arange(n_frames)
    coeffs = np.polyfit(frames, pelvis_y_new, 1)
    trend = np.polyval(coeffs, frames)

    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= trend

    # High-pass filter
    window = int(highpass_window_seconds * frame_rate)
    if window % 2 == 0:
        window += 1

    y_lowfreq = uniform_filter1d(corrected[:, pelvis_index, 1], size=window, mode='nearest')

    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= y_lowfreq

    # Center at Y=0
    median_y = np.median(corrected[:, pelvis_index, 1])
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= median_y

    # Final statistics
    info['final_x_range'] = corrected[:, pelvis_index, 0].max() - corrected[:, pelvis_index, 0].min()
    info['final_y_range'] = corrected[:, pelvis_index, 1].max() - corrected[:, pelvis_index, 1].min()

    return corrected, info


def analyze_segment_distances(corrected, frame_rate=60, pelvis_index=0):
    """Analyze walking distance per segment."""
    x = corrected[:, pelvis_index, 0]
    turnarounds = detect_turnarounds(x, frame_rate)

    if turnarounds[0] > 0:
        turnarounds = np.concatenate([[0], turnarounds])
    if turnarounds[-1] < len(x) - 1:
        turnarounds = np.concatenate([turnarounds, [len(x) - 1]])

    x_ranges = []
    for i in range(len(turnarounds) - 1):
        start = turnarounds[i]
        end = turnarounds[i + 1]
        seg_x = x[start:end]
        x_ranges.append(seg_x.max() - seg_x.min())

    return x_ranges, turnarounds


def plot_comparison(original, v7_data, segment_pca_data, output_path, frame_rate=60):
    """Create comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    methods = [
        ('Original', original),
        ('V7 (Global PCA)', v7_data),
        ('Segment PCA', segment_pca_data),
    ]

    # Row 1: Trajectories
    for i, (name, data) in enumerate(methods):
        ax = axes[0, i]
        x = data[:, 0, 0]
        y = data[:, 0, 1]
        time = np.arange(len(x)) / frame_rate / 60

        scatter = ax.scatter(x*1000, y*1000, c=time, cmap='viridis', s=0.5, alpha=0.5)
        ax.plot(x[0]*1000, y[0]*1000, 'go', markersize=8, label='Start')
        ax.plot(x[-1]*1000, y[-1]*1000, 'rs', markersize=8, label='End')

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        ax.set_title(f'{name}\nX: {x_range:.2f}m, Y: {y_range:.2f}m')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    # Row 2: Segment X ranges
    for i, (name, data) in enumerate(methods):
        ax = axes[1, i]
        x_ranges, _ = analyze_segment_distances(data, frame_rate)

        ax.bar(range(len(x_ranges)), x_ranges, alpha=0.7)
        ax.axhline(y=8.0, color='r', linestyle='--', label='Expected (8m)')
        ax.set_title(f'{name} - Walking Distance per Segment')
        ax.set_xlabel('Segment')
        ax.set_ylabel('X Range (m)')
        ax.legend()
        ax.set_ylim(0, 10)

        # Statistics
        mean_x = np.mean(x_ranges)
        std_x = np.std(x_ranges)
        ax.text(0.95, 0.95, f'Mean: {mean_x:.2f}m\nStd: {std_x:.2f}m',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved: {output_path}")


def main():
    input_path = Path("data/type2/type02_02/NCC24-001.xlsx")
    output_dir = input_path.parent
    frame_rate = 60

    print("=" * 70)
    print("Segment-wise PCA Correction for Consistent Walking Distance")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Load V7 for comparison
    import pandas as pd
    v7_df = pd.read_csv(output_dir / "NCC24-001_v7_final.csv")
    v7_data = original_data.copy()
    for i, name in enumerate(data.segment_names):
        v7_data[:, i, 0] = v7_df[f'{name}_X'].values
        v7_data[:, i, 1] = v7_df[f'{name}_Y'].values
        v7_data[:, i, 2] = v7_df[f'{name}_Z'].values

    # Apply segment PCA correction
    print("\n[2/4] Applying segment-wise PCA correction...")
    segment_pca_data, info = apply_segment_pca_correction(
        original_data.copy(),
        frame_rate=frame_rate,
        expected_distance=8.0,
        highpass_window_seconds=4.0,
    )

    # Export
    print("\n[3/4] Exporting results...")
    stem = input_path.stem
    export_to_csv(segment_pca_data, output_dir / f"{stem}_segment_pca.csv",
                  data.segment_names, frame_rate=frame_rate)

    # Plot
    print("\n[4/4] Generating comparison...")
    plot_path = output_dir / f"{stem}_segment_pca_comparison.png"
    plot_comparison(original_data, v7_data, segment_pca_data, plot_path, frame_rate)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    orig_x_ranges, _ = analyze_segment_distances(original_data, frame_rate)
    v7_x_ranges, _ = analyze_segment_distances(v7_data, frame_rate)
    seg_x_ranges, _ = analyze_segment_distances(segment_pca_data, frame_rate)

    print(f"\nWalking distance per segment (expected: 8m):")
    print(f"  Original:     mean={np.mean(orig_x_ranges):.2f}m, std={np.std(orig_x_ranges):.2f}m")
    print(f"  V7:           mean={np.mean(v7_x_ranges):.2f}m, std={np.std(v7_x_ranges):.2f}m")
    print(f"  Segment PCA:  mean={np.mean(seg_x_ranges):.2f}m, std={np.std(seg_x_ranges):.2f}m")

    print(f"\nFinal ranges:")
    print(f"  Segment PCA: X={info['final_x_range']:.2f}m, Y={info['final_y_range']:.2f}m")


if __name__ == '__main__':
    main()
