#!/usr/bin/env python3
"""
Compare Both Datasets with Same Correction Pipeline

Applies aggressive Y alignment to both type02_01 and type02_02
and creates a side-by-side comparison.
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
from scripts.gait_correction.export import export_to_csv


def apply_aggressive_alignment(
    data: np.ndarray,
    original_data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """Remove linear Y trend within each segment and align to Y=0."""
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
    """Process a single dataset through the full pipeline."""
    print(f"\nProcessing: {input_path.name}")

    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Baseline correction
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

    # Aggressive alignment
    aggressive = apply_aggressive_alignment(baseline, original_data, frame_rate=frame_rate)

    return {
        'name': input_path.stem,
        'original': original_data,
        'baseline': baseline,
        'aggressive': aggressive,
        'segment_names': data.segment_names,
    }


def y_range(data, pelvis_index=0):
    return np.max(data[:, pelvis_index, 1]) - np.min(data[:, pelvis_index, 1])


def create_comparison_plot(results1, results2, output_path: Path):
    """Create side-by-side comparison of both datasets."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    pelvis_index = 0

    # Dataset names
    name1 = "type02_01"
    name2 = "type02_02"

    # Row 0: Original
    ax = axes[0, 0]
    ax.plot(results1['original'][:, pelvis_index, 0],
            results1['original'][:, pelvis_index, 1],
            'b-', alpha=0.5, linewidth=0.3)
    ax.set_title(f'{name1} - Original\nY range: {y_range(results1["original"]):.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(results2['original'][:, pelvis_index, 0],
            results2['original'][:, pelvis_index, 1],
            'b-', alpha=0.5, linewidth=0.3)
    ax.set_title(f'{name2} - Original\nY range: {y_range(results2["original"]):.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Row 0: Baseline
    ax = axes[0, 2]
    ax.plot(results1['baseline'][:, pelvis_index, 0],
            results1['baseline'][:, pelvis_index, 1],
            'g-', alpha=0.5, linewidth=0.3)
    yr = y_range(results1['baseline'])
    red = (1 - yr / y_range(results1['original'])) * 100
    ax.set_title(f'{name1} - Baseline\nY: {yr:.2f}m ({red:.1f}% reduction)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 3]
    ax.plot(results2['baseline'][:, pelvis_index, 0],
            results2['baseline'][:, pelvis_index, 1],
            'g-', alpha=0.5, linewidth=0.3)
    yr = y_range(results2['baseline'])
    red = (1 - yr / y_range(results2['original'])) * 100
    ax.set_title(f'{name2} - Baseline\nY: {yr:.2f}m ({red:.1f}% reduction)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Row 1: Aggressive
    ax = axes[1, 0]
    ax.plot(results1['aggressive'][:, pelvis_index, 0],
            results1['aggressive'][:, pelvis_index, 1],
            'r-', alpha=0.5, linewidth=0.3)
    yr = y_range(results1['aggressive'])
    red = (1 - yr / y_range(results1['original'])) * 100
    ax.set_title(f'{name1} - Aggressive\nY: {yr:.2f}m ({red:.1f}% reduction)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(results2['aggressive'][:, pelvis_index, 0],
            results2['aggressive'][:, pelvis_index, 1],
            'r-', alpha=0.5, linewidth=0.3)
    yr = y_range(results2['aggressive'])
    red = (1 - yr / y_range(results2['original'])) * 100
    ax.set_title(f'{name2} - Aggressive\nY: {yr:.2f}m ({red:.1f}% reduction)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Row 1: Overlays
    ax = axes[1, 2]
    ax.plot(results1['original'][:, pelvis_index, 0],
            results1['original'][:, pelvis_index, 1],
            'b-', alpha=0.2, linewidth=0.3, label='Original')
    ax.plot(results1['aggressive'][:, pelvis_index, 0],
            results1['aggressive'][:, pelvis_index, 1],
            'r-', alpha=0.6, linewidth=0.3, label='Aggressive')
    ax.set_title(f'{name1} - Comparison')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 3]
    ax.plot(results2['original'][:, pelvis_index, 0],
            results2['original'][:, pelvis_index, 1],
            'b-', alpha=0.2, linewidth=0.3, label='Original')
    ax.plot(results2['aggressive'][:, pelvis_index, 0],
            results2['aggressive'][:, pelvis_index, 1],
            'r-', alpha=0.6, linewidth=0.3, label='Aggressive')
    ax.set_title(f'{name2} - Comparison')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Row 2: Y coordinate over time
    ax = axes[2, 0]
    n1 = len(results1['original'])
    step1 = max(1, n1 // 5000)
    frames1 = np.arange(0, n1, step1)
    ax.plot(frames1, results1['original'][::step1, pelvis_index, 1], 'b-', alpha=0.5, linewidth=0.5, label='Original')
    ax.plot(frames1, results1['aggressive'][::step1, pelvis_index, 1], 'r-', alpha=0.7, linewidth=0.5, label='Aggressive')
    ax.set_title(f'{name1} - Y over Time')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    n2 = len(results2['original'])
    step2 = max(1, n2 // 5000)
    frames2 = np.arange(0, n2, step2)
    ax.plot(frames2, results2['original'][::step2, pelvis_index, 1], 'b-', alpha=0.5, linewidth=0.5, label='Original')
    ax.plot(frames2, results2['aggressive'][::step2, pelvis_index, 1], 'r-', alpha=0.7, linewidth=0.5, label='Aggressive')
    ax.set_title(f'{name2} - Y over Time')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[2, 2]
    ax.axis('off')

    summary_text = f"""
SUMMARY COMPARISON

{name1} (trial-001_sub012):
  Original:    {y_range(results1['original']):.2f}m
  Baseline:    {y_range(results1['baseline']):.2f}m ({(1-y_range(results1['baseline'])/y_range(results1['original']))*100:.1f}%)
  Aggressive:  {y_range(results1['aggressive']):.2f}m ({(1-y_range(results1['aggressive'])/y_range(results1['original']))*100:.1f}%)

{name2} (NCC24-001):
  Original:    {y_range(results2['original']):.2f}m
  Baseline:    {y_range(results2['baseline']):.2f}m ({(1-y_range(results2['baseline'])/y_range(results2['original']))*100:.1f}%)
  Aggressive:  {y_range(results2['aggressive']):.2f}m ({(1-y_range(results2['aggressive'])/y_range(results2['original']))*100:.1f}%)

Key Differences:
- type02_01: Radial drift pattern (heading drift dominant)
- type02_02: Parallel offset pattern (Y drift between passes)

Recommended:
- type02_01: Baseline correction sufficient
- type02_02: Aggressive alignment needed
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    ax = axes[2, 3]
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison to {output_path}")


def main():
    print("=" * 70)
    print("Comparing Both Datasets")
    print("=" * 70)

    # Process both datasets
    path1 = Path("data/type2/type02_01/trial-001_sub012.xlsx")
    path2 = Path("data/type2/type02_02/NCC24-001.xlsx")

    results1 = process_dataset(path1)
    results2 = process_dataset(path2)

    # Create comparison
    output_path = Path("data/type2/both_datasets_comparison.png")
    create_comparison_plot(results1, results2, output_path)

    # Export aggressive corrected for type02_01
    csv_path = path1.parent / f"{path1.stem}_aggressive_corrected.csv"
    export_to_csv(results1['aggressive'], csv_path, results1['segment_names'])

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\ntype02_01 (trial-001_sub012):")
    print(f"  Original Y range:    {y_range(results1['original']):.2f}m")
    print(f"  Baseline Y range:    {y_range(results1['baseline']):.2f}m ({(1-y_range(results1['baseline'])/y_range(results1['original']))*100:.1f}% reduction)")
    print(f"  Aggressive Y range:  {y_range(results1['aggressive']):.2f}m ({(1-y_range(results1['aggressive'])/y_range(results1['original']))*100:.1f}% reduction)")

    print("\ntype02_02 (NCC24-001):")
    print(f"  Original Y range:    {y_range(results2['original']):.2f}m")
    print(f"  Baseline Y range:    {y_range(results2['baseline']):.2f}m ({(1-y_range(results2['baseline'])/y_range(results2['original']))*100:.1f}% reduction)")
    print(f"  Aggressive Y range:  {y_range(results2['aggressive']):.2f}m ({(1-y_range(results2['aggressive'])/y_range(results2['original']))*100:.1f}% reduction)")

    print(f"\nComparison plot: {output_path}")


if __name__ == '__main__':
    main()
