#!/usr/bin/env python3
"""
Compare Gait Correction Methods

This script applies multiple correction methods to walking data and
generates comparison visualizations.

Usage:
    python scripts/compare_correction_methods.py <input_file> [--output <output_dir>]
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction, SmoothPCAParams
from scripts.gait_correction.drift_removal import apply_full_drift_correction
from scripts.gait_correction.advanced_correction import (
    apply_all_methods,
    CorrectionResult,
)
from scripts.gait_correction.export import export_to_csv


def plot_all_methods(
    original_data: np.ndarray,
    baseline_data: np.ndarray,
    results: list,
    output_path: Path,
    pelvis_index: int = 0,
):
    """Create comparison plot of all correction methods."""
    n_methods = len(results) + 2  # Original + Baseline + Methods

    # Calculate grid size
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    # Plot original
    ax = axes[0]
    ax.plot(
        original_data[:, pelvis_index, 0],
        original_data[:, pelvis_index, 1],
        'b-', alpha=0.5, linewidth=0.3,
    )
    y_range = np.max(original_data[:, pelvis_index, 1]) - np.min(original_data[:, pelvis_index, 1])
    ax.set_title(f'Original\nY range: {y_range:.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Plot baseline (Smooth PCA + Drift)
    ax = axes[1]
    ax.plot(
        baseline_data[:, pelvis_index, 0],
        baseline_data[:, pelvis_index, 1],
        'g-', alpha=0.5, linewidth=0.3,
    )
    y_range = np.max(baseline_data[:, pelvis_index, 1]) - np.min(baseline_data[:, pelvis_index, 1])
    ax.set_title(f'Baseline (Smooth PCA + Drift)\nY range: {y_range:.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Plot each method
    colors = ['red', 'purple', 'orange', 'cyan', 'magenta']
    for i, result in enumerate(results):
        ax = axes[i + 2]
        ax.plot(
            result.data[:, pelvis_index, 0],
            result.data[:, pelvis_index, 1],
            '-', color=colors[i % len(colors)], alpha=0.5, linewidth=0.3,
        )
        reduction = (1 - result.y_range_corrected / result.y_range_original) * 100
        ax.set_title(f'{result.name}\nY range: {result.y_range_corrected:.2f}m ({reduction:.1f}% reduction)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def plot_overlay_comparison(
    original_data: np.ndarray,
    results: list,
    output_path: Path,
    pelvis_index: int = 0,
):
    """Create overlay comparison of best methods."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot original (faint)
    ax.plot(
        original_data[:, pelvis_index, 0],
        original_data[:, pelvis_index, 1],
        'b-', alpha=0.2, linewidth=0.3, label='Original',
    )

    # Plot each method
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    for i, result in enumerate(results):
        ax.plot(
            result.data[:, pelvis_index, 0],
            result.data[:, pelvis_index, 1],
            '-', color=colors[i % len(colors)], alpha=0.6, linewidth=0.3,
            label=f'{result.name} (Y:{result.y_range_corrected:.2f}m)',
        )

    ax.set_title('Method Comparison Overlay')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved overlay plot to {output_path}")


def print_summary_table(results: list, baseline_y_range: float):
    """Print summary table of all methods."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Method':<35} {'Y Range (m)':<15} {'Reduction':<15}")
    print("-" * 80)
    print(f"{'Original':<35} {results[0].y_range_original:<15.4f} {'-':<15}")
    print(f"{'Baseline (Smooth PCA + Drift)':<35} {baseline_y_range:<15.4f} "
          f"{(1 - baseline_y_range/results[0].y_range_original)*100:.1f}%")

    for result in results:
        reduction = (1 - result.y_range_corrected / result.y_range_original) * 100
        print(f"{result.name:<35} {result.y_range_corrected:<15.4f} {reduction:.1f}%")

    print("=" * 80)

    # Find best method
    best = min(results, key=lambda r: r.y_range_corrected)
    print(f"\nBest method: {best.name}")
    print(f"  Y range: {best.y_range_corrected:.4f}m")
    print(f"  Reduction: {(1 - best.y_range_corrected/best.y_range_original)*100:.1f}%")


def run_comparison(
    input_path: Path,
    output_dir: Path,
    frame_rate: int = 60,
):
    """
    Run comparison of all correction methods.

    Args:
        input_path: Path to input Excel file
        output_dir: Directory for output files
        frame_rate: Frame rate in Hz
    """
    print("=" * 80)
    print(f"Gait Correction Method Comparison")
    print(f"Input: {input_path}")
    print("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Step 2: Apply baseline correction (Smooth PCA + Drift)
    print("\n[Step 2] Applying baseline correction (Smooth PCA + Drift)...")
    params = SmoothPCAParams(
        window_seconds=30.0,
        sample_interval_seconds=5.0,
        smoothing_factor=0.1,
        frame_rate=frame_rate,
    )

    baseline_corrected, _, _ = apply_smooth_pca_correction(
        data.positions,
        skip_start_seconds=5.0,
        skip_end_seconds=5.0,
        params=params,
        pelvis_index=0,
    )

    baseline_corrected = apply_full_drift_correction(
        baseline_corrected,
        drift_window_seconds=30.0,
        frame_rate=frame_rate,
        pelvis_index=0,
    )

    baseline_y_range = (
        np.max(baseline_corrected[:, 0, 1]) - np.min(baseline_corrected[:, 0, 1])
    )

    # Step 3: Apply all advanced methods (on baseline-corrected data)
    # Pass original data for turnaround detection
    print("\n[Step 3] Applying advanced correction methods...")
    results = apply_all_methods(
        baseline_corrected,
        frame_rate=frame_rate,
        pelvis_index=0,
        original_data=original_data,
    )

    # Step 4: Generate visualizations
    print("\n[Step 4] Generating visualizations...")

    stem = input_path.stem
    comparison_path = output_dir / f"{stem}_method_comparison.png"
    overlay_path = output_dir / f"{stem}_method_overlay.png"

    plot_all_methods(
        original_data,
        baseline_corrected,
        results,
        comparison_path,
        pelvis_index=0,
    )

    plot_overlay_comparison(
        original_data,
        results,
        overlay_path,
        pelvis_index=0,
    )

    # Step 5: Export best result
    print("\n[Step 5] Exporting results...")

    best = min(results, key=lambda r: r.y_range_corrected)
    best_csv_path = output_dir / f"{stem}_best_corrected.csv"

    export_to_csv(
        best.data,
        best_csv_path,
        data.segment_names,
        frame_rate=frame_rate,
    )

    # Print summary
    print_summary_table(results, baseline_y_range)

    print(f"\nOutput files:")
    print(f"  Comparison: {comparison_path}")
    print(f"  Overlay: {overlay_path}")
    print(f"  Best result: {best_csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare gait correction methods',
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

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = args.output or args.input.parent

    run_comparison(
        input_path=args.input,
        output_dir=output_dir,
        frame_rate=args.frame_rate,
    )


if __name__ == '__main__':
    main()
