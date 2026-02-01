#!/usr/bin/env python3
"""
LSTM Preprocessing Pipeline

Complete pipeline for LSTM/PCA gait analysis:
1. Load raw data
2. Apply Feature-Preserving correction (drift removal while preserving gait rhythm)
3. Extract straight-line walking segments (velocity-based)
4. Visualize results

This integrates gait-skeleton-visualization preprocessing with ms9_gait-main's
segment extraction approach.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_analysis.feature_preserving_correction import (
    apply_feature_preserving_correction,
    CorrectionResult
)


def extract_straight_segments(
    data: np.ndarray,
    velocity_data: np.ndarray,
    frame_rate: int = 60,
    velocity_threshold: float = 0.5,
    min_segment_frames: int = 100,
) -> list:
    """
    Extract straight-line walking segments based on velocity.

    Args:
        data: Position data (n_frames, n_segments, 3)
        velocity_data: Velocity data (n_frames, 2) - X and Y velocity of reference point
        frame_rate: Frame rate in Hz
        velocity_threshold: Minimum velocity to consider as walking
        min_segment_frames: Minimum frames for valid segment

    Returns:
        List of (start_frame, end_frame, segment_data) tuples
    """
    velocity_magnitude = np.linalg.norm(velocity_data, axis=1)

    # Find segments where velocity exceeds threshold
    above_threshold = velocity_magnitude > velocity_threshold

    # Find segment boundaries
    changes = np.diff(above_threshold.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases
    if above_threshold[0]:
        starts = np.concatenate([[0], starts])
    if above_threshold[-1]:
        ends = np.concatenate([ends, [len(velocity_magnitude)]])

    segments = []
    for start, end in zip(starts, ends):
        if end - start >= min_segment_frames:
            segment_data = data[start:end].copy()
            segments.append({
                'start': start,
                'end': end,
                'data': segment_data,
                'duration_sec': (end - start) / frame_rate,
                'mean_velocity': np.mean(velocity_magnitude[start:end]),
            })

    return segments, velocity_magnitude


def compute_velocity_from_positions(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """Compute velocity from position data."""
    pelvis_pos = data[:, pelvis_index, :2]  # X, Y only
    velocity = np.gradient(pelvis_pos, 1/frame_rate, axis=0)
    return velocity


def visualize_pipeline_results(
    original_data: np.ndarray,
    corrected_result: CorrectionResult,
    segments: list,
    velocity: np.ndarray,
    velocity_threshold: float,
    frame_rate: int,
    output_path: Path,
    file_name: str,
    pelvis_index: int = 0,
):
    """Create comprehensive visualization of the pipeline results."""
    fig = plt.figure(figsize=(20, 20))

    n_frames = len(original_data)
    time = np.arange(n_frames) / frame_rate

    corrected_data = corrected_result.data

    # ==========================================================================
    # Row 1: Original vs Corrected Trajectories
    # ==========================================================================
    ax1 = fig.add_subplot(4, 2, 1)
    orig_x = original_data[:, pelvis_index, 0]
    orig_y = original_data[:, pelvis_index, 1]
    colors = time / 60  # Time in minutes
    sc = ax1.scatter(orig_x, orig_y, c=colors, cmap='viridis', s=1, alpha=0.5)
    ax1.plot(orig_x[0], orig_y[0], 'go', markersize=10, label='Start')
    ax1.plot(orig_x[-1], orig_y[-1], 'rs', markersize=10, label='End')
    ax1.set_title(f'Original Trajectory\nY range: {orig_y.max()-orig_y.min():.2f}m', fontsize=12)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(4, 2, 2)
    corr_x = corrected_data[:, pelvis_index, 0]
    corr_y = corrected_data[:, pelvis_index, 1]
    sc = ax2.scatter(corr_x, corr_y, c=colors, cmap='viridis', s=1, alpha=0.5)
    ax2.plot(corr_x[0], corr_y[0], 'go', markersize=10, label='Start')
    ax2.plot(corr_x[-1], corr_y[-1], 'rs', markersize=10, label='End')
    ax2.set_title(f'Feature-Preserving Corrected\nY range: {corr_y.max()-corr_y.min():.2f}m', fontsize=12)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ==========================================================================
    # Row 2: Velocity and Segment Extraction
    # ==========================================================================
    ax3 = fig.add_subplot(4, 1, 2)
    velocity_mag = np.linalg.norm(velocity, axis=1)
    ax3.plot(time, velocity_mag, 'b-', linewidth=0.5, alpha=0.7, label='Velocity')
    ax3.axhline(velocity_threshold, color='red', linestyle='--', label=f'Threshold ({velocity_threshold} m/s)')

    # Highlight segments
    for i, seg in enumerate(segments):
        start_t = seg['start'] / frame_rate
        end_t = seg['end'] / frame_rate
        ax3.axvspan(start_t, end_t, alpha=0.3, color='green', label='Segment' if i == 0 else None)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title(f'Velocity-based Segment Extraction ({len(segments)} segments)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # ==========================================================================
    # Row 3: Extracted Segments Visualization
    # ==========================================================================
    n_segments_to_show = min(4, len(segments))
    for i in range(n_segments_to_show):
        ax = fig.add_subplot(4, 4, 9 + i)
        seg = segments[i]
        seg_x = seg['data'][:, pelvis_index, 0]
        seg_y = seg['data'][:, pelvis_index, 1]
        seg_frames = np.arange(len(seg_x))

        ax.scatter(seg_x, seg_y, c=seg_frames, cmap='viridis', s=2)
        ax.plot(seg_x[0], seg_y[0], 'go', markersize=8)
        ax.plot(seg_x[-1], seg_y[-1], 'rs', markersize=8)
        ax.set_title(f'Segment {i+1}\n{seg["duration_sec"]:.1f}s, {seg["mean_velocity"]:.2f}m/s', fontsize=10)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    # ==========================================================================
    # Row 4: Gait Rhythm Analysis (Y-axis of first segment)
    # ==========================================================================
    if segments:
        ax_y = fig.add_subplot(4, 2, 7)
        seg = segments[0]
        seg_y = seg['data'][:, pelvis_index, 1]
        seg_time = np.arange(len(seg_y)) / frame_rate

        ax_y.plot(seg_time, seg_y, 'b-', linewidth=1)
        ax_y.set_xlabel('Time (s)')
        ax_y.set_ylabel('Y position (m)')
        ax_y.set_title('Segment 1: Y-axis Time Series (Gait Oscillations)', fontsize=10)
        ax_y.grid(True, alpha=0.3)

        # Frequency spectrum
        ax_fft = fig.add_subplot(4, 2, 8)
        n = len(seg_y)
        freqs = fftfreq(n, 1/frame_rate)
        spectrum = np.abs(fft(seg_y - np.mean(seg_y)))

        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        spec_pos = spectrum[pos_mask]

        freq_limit = freqs_pos < 5
        ax_fft.semilogy(freqs_pos[freq_limit], spec_pos[freq_limit], 'b-', linewidth=1)
        ax_fft.axvspan(0.5, 3.0, alpha=0.2, color='yellow', label='Gait band (0.5-3 Hz)')
        ax_fft.set_xlabel('Frequency (Hz)')
        ax_fft.set_ylabel('Power')
        ax_fft.set_title('Segment 1: Frequency Spectrum', fontsize=10)
        ax_fft.legend()
        ax_fft.grid(True, alpha=0.3)

    plt.suptitle(f'LSTM Preprocessing Pipeline: {file_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {output_path}")


def create_summary_report(
    file_name: str,
    original_data: np.ndarray,
    corrected_result: CorrectionResult,
    segments: list,
    frame_rate: int,
    pelvis_index: int = 0,
) -> str:
    """Create text summary of preprocessing results."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"LSTM PREPROCESSING PIPELINE REPORT: {file_name}")
    lines.append("=" * 70)

    # Original data stats
    orig_y = original_data[:, pelvis_index, 1]
    orig_y_range = orig_y.max() - orig_y.min()

    lines.append(f"\n[1] Original Data:")
    lines.append(f"    Total frames: {len(original_data)}")
    lines.append(f"    Duration: {len(original_data)/frame_rate:.1f} seconds")
    lines.append(f"    Y-axis range: {orig_y_range:.2f} m (drift indicator)")

    # Correction results
    info = corrected_result.info
    lines.append(f"\n[2] Feature-Preserving Correction:")
    lines.append(f"    Rotation applied: {info.get('rotation_angle_deg', 0):.2f}°")
    lines.append(f"    Y-range after correction: {info.get('corrected_y_range', 0):.2f} m")
    lines.append(f"    Y-range reduction: {info.get('y_range_reduction_pct', 0):.1f}%")
    lines.append(f"    Gait rhythm preservation: {info.get('gait_power_preservation', 0):.1%}")

    # Segment extraction
    lines.append(f"\n[3] Segment Extraction:")
    lines.append(f"    Total segments: {len(segments)}")

    if segments:
        durations = [s['duration_sec'] for s in segments]
        velocities = [s['mean_velocity'] for s in segments]
        lines.append(f"    Duration range: {min(durations):.1f} - {max(durations):.1f} s")
        lines.append(f"    Mean duration: {np.mean(durations):.1f} s")
        lines.append(f"    Velocity range: {min(velocities):.2f} - {max(velocities):.2f} m/s")

        total_frames = sum(s['end'] - s['start'] for s in segments)
        lines.append(f"    Total usable frames: {total_frames}")
        lines.append(f"    Data utilization: {total_frames/len(original_data)*100:.1f}%")

    lines.append("\n" + "=" * 70)
    lines.append("Recommendation: Use 'moderate' strength for LSTM training")
    lines.append("=" * 70)

    return "\n".join(lines)


def process_single_file(
    input_path: Path,
    output_dir: Path,
    frame_rate: int = 60,
    velocity_threshold: float = 0.5,
    correction_strength: str = 'moderate',
):
    """Process a single file through the full pipeline."""
    print(f"\n{'='*70}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*70}")

    # 1. Load data
    print("\n[1/4] Loading data...")
    loader = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = loader.positions.copy()

    # 2. Apply Feature-Preserving correction
    print("\n[2/4] Applying Feature-Preserving correction...")
    corrected_result = apply_feature_preserving_correction(
        original_data.copy(),
        frame_rate=frame_rate,
        drift_correction_strength=correction_strength,
    )

    # 3. Compute velocity and extract segments
    print("\n[3/4] Extracting straight-line segments...")
    velocity = compute_velocity_from_positions(
        corrected_result.data,
        frame_rate=frame_rate,
    )

    segments, velocity_mag = extract_straight_segments(
        corrected_result.data,
        velocity,
        frame_rate=frame_rate,
        velocity_threshold=velocity_threshold,
    )

    print(f"    Found {len(segments)} valid segments")

    # 4. Visualize and save
    print("\n[4/4] Creating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_path = output_dir / f"{input_path.stem}_pipeline_results.png"
    visualize_pipeline_results(
        original_data,
        corrected_result,
        segments,
        velocity,
        velocity_threshold,
        frame_rate,
        viz_path,
        input_path.stem,
    )

    # Save summary
    summary = create_summary_report(
        input_path.stem,
        original_data,
        corrected_result,
        segments,
        frame_rate,
    )
    print("\n" + summary)

    summary_path = output_dir / f"{input_path.stem}_report.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)

    # Save segments as CSV (compatible with ms9_gait-main)
    segments_dir = output_dir / "segments" / input_path.stem
    segments_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments, start=1):
        seg_data = seg['data']
        n_frames, n_segments, n_coords = seg_data.shape

        # Flatten to DataFrame (segment_x, segment_y, segment_z format)
        columns = []
        flat_data = []
        segment_names = loader.segment_names if hasattr(loader, 'segment_names') else [f"Seg{j}" for j in range(n_segments)]

        for j, name in enumerate(segment_names):
            columns.extend([f"{name} x", f"{name} y", f"{name} z"])

        flat_data = seg_data.reshape(n_frames, -1)
        df = pd.DataFrame(flat_data, columns=columns)

        csv_path = segments_dir / f"{str(i).zfill(3)}.csv"
        df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(segments)} segments to: {segments_dir}")

    return {
        'file': input_path.stem,
        'n_segments': len(segments),
        'y_reduction': corrected_result.info.get('y_range_reduction_pct', 0),
        'gait_preservation': corrected_result.info.get('gait_power_preservation', 0),
    }


def main():
    project_root = Path(__file__).parent.parent

    # Process sample files
    sample_files = [
        project_root / "data/type2/datatype2/NCC03-001.xlsx",
        project_root / "data/type2/datatype2/NCC13-002.xlsx",
    ]

    output_dir = project_root / "data/type2/lstm_pipeline_output"

    results = []
    for input_path in sample_files:
        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue

        result = process_single_file(
            input_path,
            output_dir,
            frame_rate=60,
            velocity_threshold=0.5,
            correction_strength='moderate',
        )
        results.append(result)

    # Print overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"{'File':<20} {'Segments':<12} {'Y Reduction':<15} {'Gait Pres.':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['file']:<20} {r['n_segments']:<12} {r['y_reduction']:<15.1f}% {r['gait_preservation']:<15.1%}")


if __name__ == "__main__":
    main()
