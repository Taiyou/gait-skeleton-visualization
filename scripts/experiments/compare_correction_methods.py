#!/usr/bin/env python3
"""
Compare Gait Correction Methods

This script applies multiple correction methods to walking data and
generates comparison visualizations.

Usage:
    python scripts/compare_correction_methods.py <input_file> [--output <output_dir>]
"""

import argparse
from pathlib import Path
import numpy as np

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction, SmoothPCAParams
from scripts.gait_correction.drift_removal import apply_full_drift_correction
from scripts.gait_correction.advanced_correction import apply_all_methods
from scripts.gait_correction.export import export_to_csv
from scripts.utils.config import setup_matplotlib, GaitCorrectionConfig
from scripts.utils.plotting import (
    plot_multi_method_comparison,
    plot_overlay_trajectories,
    print_summary_table,
    calc_range,
    PlotConfig,
)

# Setup matplotlib backend
setup_matplotlib()


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
    config = GaitCorrectionConfig(frame_rate=frame_rate)

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
        window_seconds=config.pca_window_seconds,
        sample_interval_seconds=config.pca_sample_interval_seconds,
        smoothing_factor=config.pca_smoothing_factor,
        frame_rate=frame_rate,
    )

    baseline_corrected, _, _ = apply_smooth_pca_correction(
        data.positions,
        skip_start_seconds=config.skip_start_seconds,
        skip_end_seconds=config.skip_end_seconds,
        params=params,
        pelvis_index=config.pelvis_index,
    )

    baseline_corrected = apply_full_drift_correction(
        baseline_corrected,
        drift_window_seconds=config.drift_window_seconds,
        frame_rate=frame_rate,
        pelvis_index=config.pelvis_index,
    )

    baseline_y_range = calc_range(baseline_corrected, 1, config.pelvis_index)

    # Step 3: Apply all advanced methods (on baseline-corrected data)
    print("\n[Step 3] Applying advanced correction methods...")
    results = apply_all_methods(
        baseline_corrected,
        frame_rate=frame_rate,
        pelvis_index=config.pelvis_index,
        original_data=original_data,
    )

    # Step 4: Generate visualizations
    print("\n[Step 4] Generating visualizations...")

    stem = input_path.stem
    comparison_path = output_dir / f"{stem}_method_comparison.png"
    overlay_path = output_dir / f"{stem}_method_overlay.png"

    # Prepare results for plotting
    plot_results = [
        {
            "name": "Baseline (Smooth PCA + Drift)",
            "data": baseline_corrected,
            "y_range_corrected": baseline_y_range,
            "y_range_original": calc_range(original_data, 1, config.pelvis_index),
        }
    ]
    for r in results:
        plot_results.append({
            "name": r.name,
            "data": r.data,
            "y_range_corrected": r.y_range_corrected,
            "y_range_original": r.y_range_original,
        })

    # Multi-method grid comparison
    plot_config = PlotConfig(frame_rate=frame_rate)
    plot_multi_method_comparison(
        plot_results,
        original_data,
        comparison_path,
        config=plot_config,
    )

    # Overlay comparison
    overlay_datasets = [{"name": "Original", "data": original_data, "alpha": 0.2}]
    for r in plot_results:
        overlay_datasets.append({
            "name": f"{r['name']} (Y:{r['y_range_corrected']:.2f}m)",
            "data": r["data"],
            "alpha": 0.6,
        })

    plot_overlay_trajectories(
        overlay_datasets,
        overlay_path,
        config=plot_config,
        title="Method Comparison Overlay",
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
    original_y_range = calc_range(original_data, 1, config.pelvis_index)
    summary_results = [
        {"name": r.name, "y_range_corrected": r.y_range_corrected, "y_range_original": r.y_range_original}
        for r in results
    ]
    print_summary_table(summary_results, original_y_range, baseline_y_range)

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
        return 1

    output_dir = args.output or args.input.parent

    run_comparison(
        input_path=args.input,
        output_dir=output_dir,
        frame_rate=args.frame_rate,
    )

    return 0


if __name__ == '__main__':
    exit(main())
