#!/usr/bin/env python3
"""
Apply Optimal Gait Correction

Applies the best correction method for each dataset based on analysis results.

type02_01: Baseline (Smooth PCA + Drift) only - 72.6% Y range reduction
type02_02: Baseline + Method 2 (Cumulative Y) - 38.4% additional reduction

Usage:
    python scripts/apply_optimal_correction.py <input_file> [--method <method>]
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction, SmoothPCAParams
from scripts.gait_correction.drift_removal import apply_full_drift_correction
from scripts.gait_correction.advanced_correction import method2_cumulative_y_correction
from scripts.gait_correction.export import export_to_csv, export_to_excel


def plot_final_comparison(
    original_data: np.ndarray,
    corrected_data: np.ndarray,
    output_path: Path,
    title: str,
    pelvis_index: int = 0,
):
    """Create final comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    orig_y_range = np.max(original_data[:, pelvis_index, 1]) - np.min(original_data[:, pelvis_index, 1])
    corr_y_range = np.max(corrected_data[:, pelvis_index, 1]) - np.min(corrected_data[:, pelvis_index, 1])
    reduction = (1 - corr_y_range / orig_y_range) * 100

    # Original
    ax = axes[0]
    ax.plot(
        original_data[:, pelvis_index, 0],
        original_data[:, pelvis_index, 1],
        'b-', alpha=0.5, linewidth=0.3,
    )
    ax.set_title(f'Original\nY range: {orig_y_range:.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Corrected
    ax = axes[1]
    ax.plot(
        corrected_data[:, pelvis_index, 0],
        corrected_data[:, pelvis_index, 1],
        'g-', alpha=0.5, linewidth=0.3,
    )
    ax.set_title(f'Corrected (Optimal)\nY range: {corr_y_range:.2f}m ({reduction:.1f}% reduction)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Overlay
    ax = axes[2]
    ax.plot(
        original_data[:, pelvis_index, 0],
        original_data[:, pelvis_index, 1],
        'b-', alpha=0.3, linewidth=0.3, label='Original',
    )
    ax.plot(
        corrected_data[:, pelvis_index, 0],
        corrected_data[:, pelvis_index, 1],
        'g-', alpha=0.6, linewidth=0.3, label='Corrected',
    )
    ax.set_title('Comparison Overlay')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def apply_baseline_correction(
    data: np.ndarray,
    frame_rate: int = 60,
    skip_start: float = 5.0,
    skip_end: float = 5.0,
) -> np.ndarray:
    """Apply baseline correction (Smooth PCA + Drift removal)."""
    params = SmoothPCAParams(
        window_seconds=30.0,
        sample_interval_seconds=5.0,
        smoothing_factor=0.1,
        frame_rate=frame_rate,
    )

    corrected, _, _ = apply_smooth_pca_correction(
        data,
        skip_start_seconds=skip_start,
        skip_end_seconds=skip_end,
        params=params,
        pelvis_index=0,
    )

    corrected = apply_full_drift_correction(
        corrected,
        drift_window_seconds=30.0,
        frame_rate=frame_rate,
        pelvis_index=0,
    )

    return corrected


def apply_optimal_type02_01(
    input_path: Path,
    output_dir: Path,
    frame_rate: int = 60,
):
    """
    Apply optimal correction for type02_01.

    Best method: Baseline only (Smooth PCA + Drift)
    - Original Y range: 12.55m
    - Corrected Y range: 3.44m
    - Reduction: 72.6%
    """
    print("=" * 70)
    print("Applying OPTIMAL correction for type02_01")
    print("Method: Baseline (Smooth PCA + Drift removal)")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Apply baseline correction only
    print("\n[2/4] Applying Smooth PCA correction...")
    corrected = apply_baseline_correction(data.positions, frame_rate=frame_rate)

    # Calculate metrics
    orig_y_range = np.max(original_data[:, 0, 1]) - np.min(original_data[:, 0, 1])
    corr_y_range = np.max(corrected[:, 0, 1]) - np.min(corrected[:, 0, 1])
    reduction = (1 - corr_y_range / orig_y_range) * 100

    # Export
    print("\n[3/4] Exporting results...")
    stem = input_path.stem

    csv_path = output_dir / f"{stem}_optimal.csv"
    xlsx_path = output_dir / f"{stem}_optimal.xlsx"
    plot_path = output_dir / f"{stem}_optimal_comparison.png"

    export_to_csv(corrected, csv_path, data.segment_names, frame_rate=frame_rate)
    export_to_excel(corrected, xlsx_path, data.segment_names, frame_rate=frame_rate)

    # Plot
    print("\n[4/4] Generating visualization...")
    plot_final_comparison(
        original_data,
        corrected,
        plot_path,
        title=f"type02_01: Optimal Correction (Baseline Only)",
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - type02_01")
    print("=" * 70)
    print(f"Method: Baseline (Smooth PCA + Drift removal)")
    print(f"Original Y range: {orig_y_range:.2f}m")
    print(f"Corrected Y range: {corr_y_range:.2f}m")
    print(f"Reduction: {reduction:.1f}%")
    print(f"\nOutput files:")
    print(f"  CSV: {csv_path}")
    print(f"  Excel: {xlsx_path}")
    print(f"  Plot: {plot_path}")

    return corrected


def apply_optimal_type02_02(
    input_path: Path,
    output_dir: Path,
    frame_rate: int = 60,
):
    """
    Apply optimal correction for type02_02.

    Best method: Baseline + Method 2 (Cumulative Y)
    - Original Y range: 7.73m
    - After baseline: 6.66m (13.8% reduction)
    - After Method 2: 4.10m (38.4% total reduction)
    """
    print("=" * 70)
    print("Applying OPTIMAL correction for type02_02")
    print("Method: Baseline + Cumulative Y Correction")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Apply baseline correction
    print("\n[2/5] Applying Smooth PCA correction...")
    baseline_corrected = apply_baseline_correction(data.positions, frame_rate=frame_rate)

    # Apply Method 2 (Cumulative Y)
    print("\n[3/5] Applying Cumulative Y correction...")
    result = method2_cumulative_y_correction(
        baseline_corrected,
        frame_rate=frame_rate,
        pelvis_index=0,
        turnaround_reference=original_data,
    )
    corrected = result.data

    # Calculate metrics
    orig_y_range = np.max(original_data[:, 0, 1]) - np.min(original_data[:, 0, 1])
    corr_y_range = np.max(corrected[:, 0, 1]) - np.min(corrected[:, 0, 1])
    reduction = (1 - corr_y_range / orig_y_range) * 100

    # Export
    print("\n[4/5] Exporting results...")
    stem = input_path.stem

    csv_path = output_dir / f"{stem}_optimal.csv"
    xlsx_path = output_dir / f"{stem}_optimal.xlsx"
    plot_path = output_dir / f"{stem}_optimal_comparison.png"

    export_to_csv(corrected, csv_path, data.segment_names, frame_rate=frame_rate)
    export_to_excel(corrected, xlsx_path, data.segment_names, frame_rate=frame_rate)

    # Plot
    print("\n[5/5] Generating visualization...")
    plot_final_comparison(
        original_data,
        corrected,
        plot_path,
        title=f"type02_02: Optimal Correction (Baseline + Cumulative Y)",
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - type02_02")
    print("=" * 70)
    print(f"Method: Baseline + Cumulative Y Correction")
    print(f"Original Y range: {orig_y_range:.2f}m")
    print(f"Corrected Y range: {corr_y_range:.2f}m")
    print(f"Reduction: {reduction:.1f}%")
    print(f"\nOutput files:")
    print(f"  CSV: {csv_path}")
    print(f"  Excel: {xlsx_path}")
    print(f"  Plot: {plot_path}")

    return corrected


def main():
    parser = argparse.ArgumentParser(
        description='Apply optimal gait correction',
    )
    parser.add_argument('input', type=Path, help='Input Excel file path')
    parser.add_argument(
        '--output', '-o', type=Path, default=None,
        help='Output directory (default: same as input)',
    )
    parser.add_argument(
        '--frame-rate', '-f', type=int, default=60,
        help='Frame rate in Hz (default: 60)',
    )
    parser.add_argument(
        '--method', '-m', type=str, default='auto',
        choices=['auto', 'baseline', 'cumulative'],
        help='Correction method: auto (detect from filename), baseline, or cumulative',
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = args.output or args.input.parent

    # Auto-detect method based on filename/path
    input_str = str(args.input).lower()

    if args.method == 'auto':
        if 'type02_01' in input_str or 'sub012' in input_str:
            apply_optimal_type02_01(args.input, output_dir, args.frame_rate)
        elif 'type02_02' in input_str or 'ncc24' in input_str:
            apply_optimal_type02_02(args.input, output_dir, args.frame_rate)
        else:
            print("Could not auto-detect data type. Using baseline method.")
            apply_optimal_type02_01(args.input, output_dir, args.frame_rate)
    elif args.method == 'baseline':
        apply_optimal_type02_01(args.input, output_dir, args.frame_rate)
    else:  # cumulative
        apply_optimal_type02_02(args.input, output_dir, args.frame_rate)


if __name__ == '__main__':
    main()
