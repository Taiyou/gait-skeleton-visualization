#!/usr/bin/env python3
"""
Compare Segment Extraction: Raw Data vs Preprocessed Data

Compare:
1. Raw data (NO preprocessing) - only velocity-based segment extraction
2. Preprocessed data (WITH Smooth PCA + drift removal)

This tests if preprocessing helps or hurts segment extraction.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_analysis.feature_preserving_correction import apply_feature_preserving_correction
from scripts.gait_analysis.improved_segment_extraction import (
    SegmentExtractionParams,
    extract_segments_improved
)


def compute_velocity(data: np.ndarray, frame_rate: int = 60) -> np.ndarray:
    """Compute velocity from position data."""
    pelvis_pos = data[:, 0, :2]
    velocity = np.gradient(pelvis_pos, 1/frame_rate, axis=0)
    return velocity


def process_subject(input_path: Path, frame_rate: int = 60):
    """Process a single subject with raw and preprocessed data."""

    # Load data
    loader = load_xsens_data(input_path, frame_rate=frame_rate)
    raw_data = loader.positions.copy()

    # Method 1: RAW DATA - NO preprocessing at all
    # Just use the raw data directly

    # Method 2: PREPROCESSED - With feature preserving correction
    preprocessed_result = apply_feature_preserving_correction(
        raw_data.copy(),
        frame_rate=frame_rate,
        drift_correction_strength='moderate'
    )
    preprocessed_data = preprocessed_result.data

    # Extraction parameters (same for both)
    params = SegmentExtractionParams(
        velocity_threshold=0.4,
        heading_change_threshold=0.1,
        trim_start_seconds=0.5,
        trim_end_seconds=0.3,
        use_overlapping_windows=True,
        window_seconds=5.0,
        window_overlap=0.5,
        min_segment_meters=5.0,
        frame_rate=frame_rate
    )

    # Extract from RAW data
    velocity_raw = compute_velocity(raw_data, frame_rate)
    segments_raw, info_raw = extract_segments_improved(raw_data, velocity_raw, params)

    # Extract from PREPROCESSED data
    velocity_preprocessed = compute_velocity(preprocessed_data, frame_rate)
    segments_preprocessed, info_preprocessed = extract_segments_improved(
        preprocessed_data, velocity_preprocessed, params
    )

    return {
        'raw': {
            'data': raw_data,
            'segments': segments_raw,
            'info': info_raw,
            'velocity': velocity_raw,
        },
        'preprocessed': {
            'data': preprocessed_data,
            'segments': segments_preprocessed,
            'info': info_preprocessed,
            'velocity': velocity_preprocessed,
            'gait_preservation': preprocessed_result.info.get('gait_power_preservation', 0),
            'y_reduction': preprocessed_result.info.get('y_range_reduction_pct', 0),
        }
    }


def visualize_comparison(
    results: dict,
    output_path: Path,
    subject_name: str,
    frame_rate: int = 60
):
    """Visualize raw vs preprocessed comparison."""
    fig = plt.figure(figsize=(20, 14))

    raw = results['raw']
    preprocessed = results['preprocessed']

    time = np.arange(len(raw['data'])) / frame_rate

    # Calculate Y ranges
    raw_y_range = raw['data'][:, 0, 1].max() - raw['data'][:, 0, 1].min()
    prep_y_range = preprocessed['data'][:, 0, 1].max() - preprocessed['data'][:, 0, 1].min()

    # ==========================================================================
    # Row 1: Trajectories
    # ==========================================================================
    # Raw trajectory
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.scatter(raw['data'][:, 0, 0], raw['data'][:, 0, 1],
                c=time/60, cmap='viridis', s=0.5, alpha=0.3)
    for seg in raw['segments']:
        ax1.plot(seg.data[:, 0, 0], seg.data[:, 0, 1], 'r-', linewidth=1.5, alpha=0.7)
    ax1.set_title(f'RAW DATA (No Preprocessing)\n{len(raw["segments"])} segments, Y range: {raw_y_range:.2f}m',
                  fontsize=11, color='red')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # Preprocessed trajectory
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.scatter(preprocessed['data'][:, 0, 0], preprocessed['data'][:, 0, 1],
                c=time/60, cmap='viridis', s=0.5, alpha=0.3)
    for seg in preprocessed['segments']:
        ax2.plot(seg.data[:, 0, 0], seg.data[:, 0, 1], 'g-', linewidth=1.5, alpha=0.7)
    ax2.set_title(f'PREPROCESSED (Smooth PCA + Drift Removal)\n{len(preprocessed["segments"])} segments, Y range: {prep_y_range:.2f}m',
                  fontsize=11, color='green')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # ==========================================================================
    # Row 2: Heading change rate
    # ==========================================================================
    ax3 = fig.add_subplot(3, 2, 3)
    heading_raw = raw['info']['heading_rate']
    ax3.plot(time, np.degrees(heading_raw), 'r-', linewidth=0.5, alpha=0.7)
    ax3.axhline(np.degrees(0.1), color='black', linestyle='--', alpha=0.5, label='Threshold')
    # Highlight raw segments
    for seg in raw['segments']:
        ax3.axvspan(seg.start_frame/frame_rate, seg.end_frame/frame_rate,
                    alpha=0.2, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heading change (deg/frame)')
    ax3.set_title('RAW: Heading Change Rate', fontsize=11, color='red')
    ax3.set_ylim([0, min(30, np.percentile(np.degrees(heading_raw), 99))])
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 4)
    heading_prep = preprocessed['info']['heading_rate']
    ax4.plot(time, np.degrees(heading_prep), 'g-', linewidth=0.5, alpha=0.7)
    ax4.axhline(np.degrees(0.1), color='black', linestyle='--', alpha=0.5, label='Threshold')
    # Highlight preprocessed segments
    for seg in preprocessed['segments']:
        ax4.axvspan(seg.start_frame/frame_rate, seg.end_frame/frame_rate,
                    alpha=0.2, color='green')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heading change (deg/frame)')
    ax4.set_title('PREPROCESSED: Heading Change Rate', fontsize=11, color='green')
    ax4.set_ylim([0, min(30, np.percentile(np.degrees(heading_prep), 99))])
    ax4.grid(True, alpha=0.3)

    # ==========================================================================
    # Row 3: Statistics and segment distribution
    # ==========================================================================
    ax5 = fig.add_subplot(3, 2, 5)

    # Duration histogram
    raw_durations = [s.duration_sec for s in raw['segments']] if raw['segments'] else [0]
    prep_durations = [s.duration_sec for s in preprocessed['segments']] if preprocessed['segments'] else [0]

    max_dur = max(max(raw_durations), max(prep_durations))
    bins = np.linspace(0, max_dur, 15)

    ax5.hist(raw_durations, bins=bins, alpha=0.5, label=f'Raw (n={len(raw["segments"])})', color='red')
    ax5.hist(prep_durations, bins=bins, alpha=0.5, label=f'Preprocessed (n={len(preprocessed["segments"])})', color='green')
    ax5.set_xlabel('Segment Duration (s)')
    ax5.set_ylabel('Count')
    ax5.set_title('Segment Duration Distribution', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Summary
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    raw_frames = sum(s.data.shape[0] for s in raw['segments'])
    prep_frames = sum(s.data.shape[0] for s in preprocessed['segments'])

    diff = len(preprocessed['segments']) - len(raw['segments'])
    better = 'PREPROCESSED' if diff > 0 else ('RAW' if diff < 0 else 'SAME')

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║              COMPARISON SUMMARY                          ║
    ╠══════════════════════════════════════════════════════════╣
    ║                      RAW          PREPROCESSED           ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Segments:           {len(raw['segments']):<12} {len(preprocessed['segments']):<12}         ║
    ║  Total frames:       {raw_frames:<12} {prep_frames:<12}         ║
    ║  Y-axis range:       {raw_y_range:<12.2f} {prep_y_range:<12.2f}         ║
    ║  Straight ratio:     {raw['info']['straight_ratio']*100:<12.1f}% {preprocessed['info']['straight_ratio']*100:<11.1f}%         ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Difference: {diff:+d} segments                                  ║
    ║  Better method: {better:<42} ║
    ╚══════════════════════════════════════════════════════════╝
    """

    ax6.text(0.05, 0.5, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'{subject_name}: Raw Data vs Preprocessed', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'subject': subject_name,
        'raw_segments': len(raw['segments']),
        'preprocessed_segments': len(preprocessed['segments']),
        'raw_frames': raw_frames,
        'preprocessed_frames': prep_frames,
        'raw_y_range': raw_y_range,
        'preprocessed_y_range': prep_y_range,
        'raw_straight_ratio': raw['info']['straight_ratio'],
        'preprocessed_straight_ratio': preprocessed['info']['straight_ratio'],
        'better': better.lower(),
        'difference': diff,
    }


def main():
    project_root = Path(__file__).parent.parent

    # Find all raw data
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

    all_files = sorted(set(all_files), key=lambda x: x.stem)

    print("=" * 70)
    print("COMPARING: RAW DATA vs PREPROCESSED DATA")
    print("=" * 70)
    print(f"Found {len(all_files)} files")
    print("\nThis compares segment extraction on:")
    print("  - RAW: Original data with NO preprocessing")
    print("  - PREPROCESSED: Data with Smooth PCA + drift removal")

    output_dir = project_root / "data/type2/lstm_pipeline_output/raw_vs_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, input_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing: {input_path.stem}...")

        try:
            result = process_subject(input_path, frame_rate=60)

            viz_path = output_dir / f"{input_path.stem}_raw_vs_preprocessed.png"
            stats = visualize_comparison(result, viz_path, input_path.stem, frame_rate=60)
            results.append(stats)

            print(f"    Raw: {stats['raw_segments']} segs, Preprocessed: {stats['preprocessed_segments']} segs ({stats['difference']:+d}) -> {stats['better'].upper()}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Subject':<15} {'Raw':<10} {'Preproc':<10} {'Diff':<10} {'Better':<15}")
    print("-" * 60)

    better_raw = 0
    better_preprocessed = 0
    same = 0
    total_raw = 0
    total_preprocessed = 0

    for r in results:
        print(f"{r['subject']:<15} {r['raw_segments']:<10} {r['preprocessed_segments']:<10} {r['difference']:+d}{'':<6} {r['better'].upper()}")
        total_raw += r['raw_segments']
        total_preprocessed += r['preprocessed_segments']
        if r['better'] == 'raw':
            better_raw += 1
        elif r['better'] == 'preprocessed':
            better_preprocessed += 1
        else:
            same += 1

    print("-" * 60)
    print(f"{'TOTAL':<15} {total_raw:<10} {total_preprocessed:<10} {total_preprocessed - total_raw:+d}")

    print(f"\n" + "=" * 70)
    print(f"Better with RAW data:         {better_raw} subjects")
    print(f"Better with PREPROCESSED:     {better_preprocessed} subjects")
    print(f"Same result:                  {same} subjects")
    print("=" * 70)

    # Save summary
    df = pd.DataFrame(results)
    summary_path = output_dir / "raw_vs_preprocessed_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Create overview
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    subjects = [r['subject'] for r in results]
    raw_segs = [r['raw_segments'] for r in results]
    prep_segs = [r['preprocessed_segments'] for r in results]
    differences = [r['difference'] for r in results]

    x = np.arange(len(subjects))
    width = 0.35

    # Segment count comparison
    ax1 = axes[0]
    ax1.barh(x - width/2, raw_segs, width, label='Raw (No Preprocessing)', color='salmon')
    ax1.barh(x + width/2, prep_segs, width, label='Preprocessed', color='lightgreen')
    ax1.set_yticks(x)
    ax1.set_yticklabels(subjects, fontsize=8)
    ax1.set_xlabel('Number of Segments')
    ax1.set_title('Segment Count: Raw vs Preprocessed', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Difference
    ax2 = axes[1]
    colors = ['green' if d > 0 else ('red' if d < 0 else 'gray') for d in differences]
    ax2.barh(x, differences, color=colors, alpha=0.7)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(subjects, fontsize=8)
    ax2.set_xlabel('Difference (Preprocessed - Raw)')
    ax2.set_title(f'Segment Difference\nGreen=Preprocessing helps, Red=Raw is better', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    fig.suptitle(
        f'Raw vs Preprocessed Comparison\n'
        f'Total: {total_raw} vs {total_preprocessed} ({total_preprocessed - total_raw:+d}), '
        f'Raw better: {better_raw}, Preprocessed better: {better_preprocessed}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    overview_path = output_dir / "raw_vs_preprocessed_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Overview saved to: {overview_path}")


if __name__ == "__main__":
    main()
