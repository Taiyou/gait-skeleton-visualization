#!/usr/bin/env python3
"""
Batch LSTM Preprocessing Pipeline

Process all datatype2 files with Feature-Preserving correction
and segment extraction for LSTM/PCA analysis.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_analysis.feature_preserving_correction import (
    apply_feature_preserving_correction,
    CorrectionResult
)


def compute_velocity_from_positions(
    data: np.ndarray,
    frame_rate: int = 60,
    pelvis_index: int = 0,
) -> np.ndarray:
    """Compute velocity from position data."""
    pelvis_pos = data[:, pelvis_index, :2]  # X, Y only
    velocity = np.gradient(pelvis_pos, 1/frame_rate, axis=0)
    return velocity


def extract_straight_segments(
    data: np.ndarray,
    velocity_data: np.ndarray,
    frame_rate: int = 60,
    velocity_threshold: float = 0.5,
    min_segment_frames: int = 100,
) -> list:
    """Extract straight-line walking segments based on velocity."""
    velocity_magnitude = np.linalg.norm(velocity_data, axis=1)

    above_threshold = velocity_magnitude > velocity_threshold

    changes = np.diff(above_threshold.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

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

    return segments


def process_single_file(
    input_path: Path,
    output_dir: Path,
    frame_rate: int = 60,
    velocity_threshold: float = 0.5,
    correction_strength: str = 'moderate',
    save_visualization: bool = True,
):
    """Process a single file through the full pipeline."""
    try:
        # 1. Load data
        loader = load_xsens_data(input_path, frame_rate=frame_rate)
        original_data = loader.positions.copy()

        # 2. Apply Feature-Preserving correction
        corrected_result = apply_feature_preserving_correction(
            original_data.copy(),
            frame_rate=frame_rate,
            drift_correction_strength=correction_strength,
        )

        # 3. Compute velocity and extract segments
        velocity = compute_velocity_from_positions(
            corrected_result.data,
            frame_rate=frame_rate,
        )

        segments, = extract_straight_segments(
            corrected_result.data,
            velocity,
            frame_rate=frame_rate,
            velocity_threshold=velocity_threshold,
        ),

        # 4. Save segments as CSV
        segments_dir = output_dir / "segments" / input_path.stem
        segments_dir.mkdir(parents=True, exist_ok=True)

        segment_names = loader.segment_names if hasattr(loader, 'segment_names') else [f"Seg{j}" for j in range(original_data.shape[1])]

        for i, seg in enumerate(segments, start=1):
            seg_data = seg['data']
            n_frames, n_segments, n_coords = seg_data.shape

            columns = []
            for name in segment_names:
                columns.extend([f"{name} x", f"{name} y", f"{name} z"])

            flat_data = seg_data.reshape(n_frames, -1)
            df = pd.DataFrame(flat_data, columns=columns)

            csv_path = segments_dir / f"{str(i).zfill(3)}.csv"
            df.to_csv(csv_path, index=False)

        # 5. Create quick visualization
        if save_visualization:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original trajectory
            orig_x = original_data[:, 0, 0]
            orig_y = original_data[:, 0, 1]
            axes[0].scatter(orig_x, orig_y, c=np.arange(len(orig_x))/frame_rate/60, cmap='viridis', s=1, alpha=0.5)
            axes[0].set_title(f'Original\nY range: {orig_y.max()-orig_y.min():.2f}m')
            axes[0].set_xlabel('X (m)')
            axes[0].set_ylabel('Y (m)')
            axes[0].axis('equal')

            # Corrected trajectory
            corr_x = corrected_result.data[:, 0, 0]
            corr_y = corrected_result.data[:, 0, 1]
            axes[1].scatter(corr_x, corr_y, c=np.arange(len(corr_x))/frame_rate/60, cmap='viridis', s=1, alpha=0.5)
            axes[1].set_title(f'Corrected\nY range: {corr_y.max()-corr_y.min():.2f}m')
            axes[1].set_xlabel('X (m)')
            axes[1].set_ylabel('Y (m)')
            axes[1].axis('equal')

            # Velocity with segments
            velocity_mag = np.linalg.norm(velocity, axis=1)
            time = np.arange(len(velocity_mag)) / frame_rate
            axes[2].plot(time, velocity_mag, 'b-', linewidth=0.5, alpha=0.7)
            axes[2].axhline(velocity_threshold, color='red', linestyle='--')
            for seg in segments:
                axes[2].axvspan(seg['start']/frame_rate, seg['end']/frame_rate, alpha=0.3, color='green')
            axes[2].set_title(f'{len(segments)} segments extracted')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Velocity (m/s)')

            plt.suptitle(input_path.stem, fontsize=14, fontweight='bold')
            plt.tight_layout()

            viz_path = output_dir / "visualizations" / f"{input_path.stem}.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close()

        return {
            'file': input_path.stem,
            'success': True,
            'n_frames': len(original_data),
            'n_segments': len(segments),
            'y_reduction': corrected_result.info.get('y_range_reduction_pct', 0),
            'gait_preservation': corrected_result.info.get('gait_power_preservation', 0),
            'total_segment_frames': sum(s['end'] - s['start'] for s in segments),
            'data_utilization': sum(s['end'] - s['start'] for s in segments) / len(original_data) * 100,
            'mean_velocity': np.mean([s['mean_velocity'] for s in segments]) if segments else 0,
        }

    except Exception as e:
        return {
            'file': input_path.stem,
            'success': False,
            'error': str(e),
        }


def main():
    project_root = Path(__file__).parent.parent

    # Collect all raw datatype2 files (excluding processed files)
    raw_folders = [
        project_root / "data/type2/datatype2",
        project_root / "data/type2/datatype2 2",
        project_root / "data/type2/datatype2 3",
        project_root / "data/type2/datatype2 4",
    ]

    all_files = []
    for folder in raw_folders:
        if folder.exists():
            files = list(folder.glob("NCC*.xlsx"))
            all_files.extend(files)

    # Remove duplicates and sort
    all_files = sorted(set(all_files), key=lambda x: x.stem)

    print("=" * 70)
    print("BATCH LSTM PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Found {len(all_files)} files to process")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    output_dir = project_root / "data/type2/lstm_pipeline_output"

    results = []
    for i, input_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing: {input_path.stem}...")

        result = process_single_file(
            input_path,
            output_dir,
            frame_rate=60,
            velocity_threshold=0.5,
            correction_strength='moderate',
            save_visualization=True,
        )

        if result['success']:
            print(f"    Segments: {result['n_segments']}, Y reduction: {result['y_reduction']:.1f}%, Gait pres: {result['gait_preservation']:.1%}")
        else:
            print(f"    ERROR: {result.get('error', 'Unknown error')}")

        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for r in failed:
            print(f"  - {r['file']}: {r.get('error', 'Unknown')}")

    # Create summary table
    print("\n" + "-" * 100)
    print(f"{'File':<15} {'Segments':<10} {'Frames':<10} {'Y Red %':<10} {'Gait Pres':<12} {'Data Util %':<12} {'Vel (m/s)':<10}")
    print("-" * 100)

    total_segments = 0
    total_frames = 0
    for r in successful:
        print(f"{r['file']:<15} {r['n_segments']:<10} {r['total_segment_frames']:<10} {r['y_reduction']:<10.1f} {r['gait_preservation']:<12.1%} {r['data_utilization']:<12.1f} {r['mean_velocity']:<10.2f}")
        total_segments += r['n_segments']
        total_frames += r['total_segment_frames']

    print("-" * 100)
    print(f"{'TOTAL':<15} {total_segments:<10} {total_frames:<10}")

    # Save summary to CSV
    summary_df = pd.DataFrame(successful)
    summary_path = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Create overall visualization
    if successful:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Segments per file
        files = [r['file'] for r in successful]
        segments = [r['n_segments'] for r in successful]
        axes[0, 0].barh(files, segments, color='steelblue')
        axes[0, 0].set_xlabel('Number of Segments')
        axes[0, 0].set_title('Segments Extracted per File')
        axes[0, 0].invert_yaxis()

        # 2. Y reduction
        y_reductions = [r['y_reduction'] for r in successful]
        axes[0, 1].barh(files, y_reductions, color='forestgreen')
        axes[0, 1].set_xlabel('Y-axis Drift Reduction (%)')
        axes[0, 1].set_title('Drift Removal Effectiveness')
        axes[0, 1].invert_yaxis()

        # 3. Gait preservation
        gait_pres = [r['gait_preservation'] * 100 for r in successful]
        colors = ['green' if g > 50 else 'orange' if g > 20 else 'red' for g in gait_pres]
        axes[1, 0].barh(files, gait_pres, color=colors)
        axes[1, 0].set_xlabel('Gait Rhythm Preservation (%)')
        axes[1, 0].set_title('Gait Pattern Preservation')
        axes[1, 0].axvline(50, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].invert_yaxis()

        # 4. Data utilization
        utilization = [r['data_utilization'] for r in successful]
        axes[1, 1].barh(files, utilization, color='darkorange')
        axes[1, 1].set_xlabel('Data Utilization (%)')
        axes[1, 1].set_title('Usable Data Percentage')
        axes[1, 1].invert_yaxis()

        plt.suptitle('LSTM Preprocessing Batch Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()

        summary_viz_path = output_dir / "batch_summary.png"
        plt.savefig(summary_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Summary visualization saved to: {summary_viz_path}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
