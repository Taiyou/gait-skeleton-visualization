#!/usr/bin/env python3
"""
Noise Removal Script for Xsens Motion Capture Data

This script applies the full noise removal pipeline:
1. Load Xsens Excel data
2. Apply Smooth PCA correction (heading drift)
3. Apply Y-drift removal
4. Align horizontally
5. Export corrected data

Usage:
    python scripts/run_noise_removal.py <input_file> [--output <output_file>]
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
from scripts.gait_correction.export import export_to_csv, export_to_excel


def plot_comparison(
    original_data: np.ndarray,
    corrected_data: np.ndarray,
    segment_names: list,
    output_path: Path,
    pelvis_index: int = 0,
):
    """Create comparison plot of original vs corrected trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original trajectory
    ax1 = axes[0]
    ax1.plot(
        original_data[:, pelvis_index, 0],
        original_data[:, pelvis_index, 1],
        'b-', alpha=0.7, linewidth=0.5,
    )
    ax1.set_title('Original Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # Corrected trajectory
    ax2 = axes[1]
    ax2.plot(
        corrected_data[:, pelvis_index, 0],
        corrected_data[:, pelvis_index, 1],
        'g-', alpha=0.7, linewidth=0.5,
    )
    ax2.set_title('Corrected Trajectory')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # Overlay comparison
    ax3 = axes[2]
    ax3.plot(
        original_data[:, pelvis_index, 0],
        original_data[:, pelvis_index, 1],
        'b-', alpha=0.5, linewidth=0.5, label='Original',
    )
    ax3.plot(
        corrected_data[:, pelvis_index, 0],
        corrected_data[:, pelvis_index, 1],
        'g-', alpha=0.7, linewidth=0.5, label='Corrected',
    )
    ax3.set_title('Comparison')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def run_noise_removal(
    input_path: Path,
    output_dir: Path,
    frame_rate: int = 60,
    skip_start_seconds: float = 5.0,
    skip_end_seconds: float = 5.0,
    pca_window_seconds: float = 30.0,
    drift_window_seconds: float = 30.0,
):
    """
    Run the full noise removal pipeline.

    Args:
        input_path: Path to input Excel file
        output_dir: Directory for output files
        frame_rate: Frame rate in Hz
        skip_start_seconds: Seconds to skip at start
        skip_end_seconds: Seconds to skip at end
        pca_window_seconds: Window size for PCA correction
        drift_window_seconds: Window size for drift removal
    """
    print("=" * 60)
    print(f"Noise Removal Pipeline")
    print(f"Input: {input_path}")
    print("=" * 60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    data = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = data.positions.copy()

    # Step 2: Apply Smooth PCA correction
    print("\n[Step 2] Applying Smooth PCA correction...")
    params = SmoothPCAParams(
        window_seconds=pca_window_seconds,
        sample_interval_seconds=5.0,
        smoothing_factor=0.1,
        frame_rate=frame_rate,
    )

    corrected, orig_x, orig_y = apply_smooth_pca_correction(
        data.positions,
        skip_start_seconds=skip_start_seconds,
        skip_end_seconds=skip_end_seconds,
        params=params,
        pelvis_index=0,
    )

    # Step 3: Apply drift correction
    print("\n[Step 3] Applying drift correction...")
    corrected = apply_full_drift_correction(
        corrected,
        drift_window_seconds=drift_window_seconds,
        frame_rate=frame_rate,
        pelvis_index=0,
    )

    # Step 4: Export results
    print("\n[Step 4] Exporting results...")

    # Create output filename
    stem = input_path.stem
    csv_path = output_dir / f"{stem}_corrected.csv"
    xlsx_path = output_dir / f"{stem}_corrected.xlsx"
    plot_path = output_dir / f"{stem}_comparison.png"

    export_to_csv(
        corrected,
        csv_path,
        data.segment_names,
        frame_rate=frame_rate,
    )

    export_to_excel(
        corrected,
        xlsx_path,
        data.segment_names,
        frame_rate=frame_rate,
    )

    # Create comparison plot
    plot_comparison(
        original_data,
        corrected,
        data.segment_names,
        plot_path,
        pelvis_index=0,
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    pelvis_idx = 0
    orig_y_range = np.max(original_data[:, pelvis_idx, 1]) - np.min(original_data[:, pelvis_idx, 1])
    corr_y_range = np.max(corrected[:, pelvis_idx, 1]) - np.min(corrected[:, pelvis_idx, 1])

    print(f"Original Y range: {orig_y_range:.4f} m")
    print(f"Corrected Y range: {corr_y_range:.4f} m")
    print(f"Y range reduction: {(1 - corr_y_range/orig_y_range)*100:.1f}%")

    print(f"\nOutput files:")
    print(f"  CSV: {csv_path}")
    print(f"  Excel: {xlsx_path}")
    print(f"  Plot: {plot_path}")

    return corrected


def main():
    parser = argparse.ArgumentParser(
        description='Apply noise removal to Xsens motion capture data',
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
        '--skip-start', type=float, default=5.0,
        help='Seconds to skip at start (default: 5)',
    )
    parser.add_argument(
        '--skip-end', type=float, default=5.0,
        help='Seconds to skip at end (default: 5)',
    )
    parser.add_argument(
        '--pca-window', type=float, default=30.0,
        help='PCA window size in seconds (default: 30)',
    )
    parser.add_argument(
        '--drift-window', type=float, default=30.0,
        help='Drift removal window size in seconds (default: 30)',
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = args.output or args.input.parent

    run_noise_removal(
        input_path=args.input,
        output_dir=output_dir,
        frame_rate=args.frame_rate,
        skip_start_seconds=args.skip_start,
        skip_end_seconds=args.skip_end,
        pca_window_seconds=args.pca_window,
        drift_window_seconds=args.drift_window,
    )


if __name__ == '__main__':
    main()
