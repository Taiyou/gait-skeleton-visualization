#!/usr/bin/env python3
"""
Compare Preprocessing Methods for LSTM Analysis

This script compares:
1. Standard preprocessing (for visualization)
2. Feature-preserving preprocessing (for LSTM)

Visualizes the differences in:
- Trajectory shape
- Y-axis time series
- Frequency spectrum (gait rhythm preservation)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction, SmoothPCAParams
from scripts.gait_correction.drift_removal import apply_full_drift_correction
from scripts.gait_analysis.feature_preserving_correction import (
    apply_feature_preserving_correction,
    compare_preprocessing_methods
)


def compute_spectrum(signal: np.ndarray, frame_rate: int):
    """Compute power spectrum of a signal."""
    n = len(signal)
    signal_centered = signal - np.mean(signal)
    freqs = fftfreq(n, 1/frame_rate)
    spectrum = np.abs(fft(signal_centered))

    # Only positive frequencies
    pos_mask = freqs > 0
    return freqs[pos_mask], spectrum[pos_mask]


def apply_standard_preprocessing(data: np.ndarray, frame_rate: int = 60) -> np.ndarray:
    """Apply standard preprocessing (for visualization)."""
    # Step 1: Smooth PCA correction
    params = SmoothPCAParams(
        frame_rate=frame_rate,
        window_seconds=30.0,
        sample_interval_seconds=5.0,
    )
    corrected, _, _ = apply_smooth_pca_correction(
        data.copy(),
        params=params,
    )

    # Step 2: Full drift correction
    corrected = apply_full_drift_correction(
        corrected,
        drift_window_seconds=30.0,
        frame_rate=frame_rate,
    )

    return corrected


def plot_comparison(
    original: np.ndarray,
    standard: np.ndarray,
    feature_preserving: dict,
    frame_rate: int,
    output_path: Path,
    pelvis_index: int = 0,
):
    """Create comprehensive comparison plot."""
    fig = plt.figure(figsize=(20, 16))

    n_frames = len(original)
    time = np.arange(n_frames) / frame_rate

    # Get data
    orig_x = original[:, pelvis_index, 0]
    orig_y = original[:, pelvis_index, 1]
    std_x = standard[:, pelvis_index, 0]
    std_y = standard[:, pelvis_index, 1]
    min_x = feature_preserving['minimal'].data[:, pelvis_index, 0]
    min_y = feature_preserving['minimal'].data[:, pelvis_index, 1]
    mod_x = feature_preserving['moderate'].data[:, pelvis_index, 0]
    mod_y = feature_preserving['moderate'].data[:, pelvis_index, 1]

    # =========================================================================
    # Row 1: Trajectories
    # =========================================================================
    methods = [
        ('Original', orig_x, orig_y),
        ('Standard (Visualization)', std_x, std_y),
        ('Feature-Preserving (Minimal)', min_x, min_y),
        ('Feature-Preserving (Moderate)', mod_x, mod_y),
    ]

    for i, (name, x, y) in enumerate(methods):
        ax = fig.add_subplot(4, 4, i + 1)
        colors = np.arange(len(x)) / frame_rate / 60  # Time in minutes
        scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=1, alpha=0.5)
        ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
        ax.plot(x[-1], y[-1], 'rs', markersize=8, label='End')

        y_range = y.max() - y.min()
        ax.set_title(f'{name}\nY range: {y_range:.2f}m', fontsize=10)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    # =========================================================================
    # Row 2: Y-axis time series (full)
    # =========================================================================
    ax = fig.add_subplot(4, 1, 2)
    ax.plot(time, orig_y, 'gray', alpha=0.5, linewidth=0.5, label='Original')
    ax.plot(time, std_y, 'r', alpha=0.7, linewidth=0.5, label='Standard')
    ax.plot(time, min_y, 'b', alpha=0.7, linewidth=0.5, label='Feature-Preserving (Minimal)')
    ax.plot(time, mod_y, 'g', alpha=0.7, linewidth=0.5, label='Feature-Preserving (Moderate)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Y-axis Time Series Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Row 3: Y-axis time series (zoomed - 10 seconds)
    # =========================================================================
    ax = fig.add_subplot(4, 1, 3)

    # Zoom to middle 10 seconds
    start_frame = n_frames // 2 - 5 * frame_rate
    end_frame = n_frames // 2 + 5 * frame_rate
    zoom_time = time[start_frame:end_frame]

    ax.plot(zoom_time, orig_y[start_frame:end_frame], 'gray', alpha=0.7, linewidth=1, label='Original')
    ax.plot(zoom_time, std_y[start_frame:end_frame], 'r', alpha=0.7, linewidth=1, label='Standard')
    ax.plot(zoom_time, min_y[start_frame:end_frame], 'b', alpha=0.7, linewidth=1, label='Feature-Preserving (Minimal)')
    ax.plot(zoom_time, mod_y[start_frame:end_frame], 'g', alpha=0.7, linewidth=1, label='Feature-Preserving (Moderate)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Y-axis Time Series (10-second zoom) - Gait Oscillations', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation about gait rhythm
    ax.annotate('Gait oscillations\n(should be preserved)',
                xy=(zoom_time[len(zoom_time)//4], min_y[start_frame + len(zoom_time)//4]),
                xytext=(zoom_time[len(zoom_time)//4] + 1, min_y[start_frame:end_frame].max()),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9, color='blue')

    # =========================================================================
    # Row 4: Frequency spectrum
    # =========================================================================
    ax = fig.add_subplot(4, 1, 4)

    # Compute spectra
    freq_orig, spec_orig = compute_spectrum(orig_y, frame_rate)
    freq_std, spec_std = compute_spectrum(std_y, frame_rate)
    freq_min, spec_min = compute_spectrum(min_y, frame_rate)
    freq_mod, spec_mod = compute_spectrum(mod_y, frame_rate)

    # Plot up to 5 Hz
    freq_mask = freq_orig < 5

    ax.semilogy(freq_orig[freq_mask], spec_orig[freq_mask], 'gray', alpha=0.5, linewidth=1, label='Original')
    ax.semilogy(freq_std[freq_mask], spec_std[freq_mask], 'r', alpha=0.7, linewidth=1, label='Standard')
    ax.semilogy(freq_min[freq_mask], spec_min[freq_mask], 'b', alpha=0.7, linewidth=1, label='Feature-Preserving (Minimal)')
    ax.semilogy(freq_mod[freq_mask], spec_mod[freq_mask], 'g', alpha=0.7, linewidth=1, label='Feature-Preserving (Moderate)')

    # Highlight gait frequency band
    ax.axvspan(0.5, 3.0, alpha=0.2, color='yellow', label='Gait frequency band (0.5-3 Hz)')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (log scale)')
    ax.set_title('Frequency Spectrum - Gait Rhythm Preservation', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot to: {output_path}")


def create_summary_table(
    original: np.ndarray,
    standard: np.ndarray,
    feature_preserving: dict,
    frame_rate: int,
    pelvis_index: int = 0,
) -> str:
    """Create summary table of preprocessing results."""

    def calc_metrics(data, original_data, name):
        y = data[:, pelvis_index, 1]
        orig_y = original_data[:, pelvis_index, 1]

        y_range = y.max() - y.min()
        orig_y_range = orig_y.max() - orig_y.min()
        reduction = (1 - y_range / orig_y_range) * 100

        # Gait power preservation
        freqs, spec = compute_spectrum(y, frame_rate)
        freqs_orig, spec_orig = compute_spectrum(orig_y, frame_rate)

        gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
        gait_power = np.sum(spec[gait_mask]**2)
        orig_gait_power = np.sum(spec_orig[gait_mask]**2)
        power_pres = gait_power / orig_gait_power if orig_gait_power > 0 else 1.0

        return {
            'name': name,
            'y_range': y_range,
            'reduction': reduction,
            'gait_power_preservation': power_pres
        }

    orig_y = original[:, pelvis_index, 1]
    orig_y_range = orig_y.max() - orig_y.min()

    metrics = [
        {'name': 'Original', 'y_range': orig_y_range, 'reduction': 0, 'gait_power_preservation': 1.0},
        calc_metrics(standard, original, 'Standard (Visualization)'),
        calc_metrics(feature_preserving['minimal'].data, original, 'Feature-Preserving (Minimal)'),
        calc_metrics(feature_preserving['moderate'].data, original, 'Feature-Preserving (Moderate)'),
        calc_metrics(feature_preserving['aggressive'].data, original, 'Feature-Preserving (Aggressive)'),
    ]

    # Create table
    lines = []
    lines.append("=" * 90)
    lines.append("PREPROCESSING COMPARISON SUMMARY")
    lines.append("=" * 90)
    lines.append(f"{'Method':<35} {'Y Range (m)':<15} {'Reduction %':<15} {'Gait Power Pres.':<15}")
    lines.append("-" * 90)

    for m in metrics:
        lines.append(f"{m['name']:<35} {m['y_range']:<15.2f} {m['reduction']:<15.1f} {m['gait_power_preservation']:<15.2%}")

    lines.append("=" * 90)
    lines.append("\nRecommendation:")
    lines.append("  - For VISUALIZATION: Use 'Standard' (maximum drift removal)")
    lines.append("  - For LSTM/PCA: Use 'Feature-Preserving (Minimal)' or 'Moderate'")
    lines.append("    (preserves gait patterns while removing global drift)")

    return "\n".join(lines)


def main():
    # Select a sample file
    project_root = Path(__file__).parent.parent
    sample_files = [
        project_root / "data/type2/datatype2/NCC03-001.xlsx",  # Good performer
        project_root / "data/type2/datatype2/NCC13-002.xlsx",  # Best performer
    ]

    for input_path in sample_files:
        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue

        print("\n" + "=" * 70)
        print(f"Processing: {input_path.name}")
        print("=" * 70)

        output_dir = project_root / "data/type2/lstm_preprocessing_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_rate = 60

        # Load data
        print("\n[1/4] Loading data...")
        loader = load_xsens_data(input_path, frame_rate=frame_rate)
        original_data = loader.positions.copy()

        # Apply standard preprocessing
        print("\n[2/4] Applying standard preprocessing...")
        standard_data = apply_standard_preprocessing(original_data.copy(), frame_rate)

        # Apply feature-preserving preprocessing
        print("\n[3/4] Applying feature-preserving preprocessing...")
        fp_results = compare_preprocessing_methods(original_data.copy(), frame_rate)

        # Create comparison plot
        print("\n[4/4] Creating comparison visualization...")
        output_path = output_dir / f"{input_path.stem}_lstm_preprocessing_comparison.png"
        plot_comparison(
            original_data, standard_data, fp_results,
            frame_rate, output_path
        )

        # Print summary
        summary = create_summary_table(
            original_data, standard_data, fp_results, frame_rate
        )
        print("\n" + summary)

        # Save summary
        summary_path = output_dir / f"{input_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
