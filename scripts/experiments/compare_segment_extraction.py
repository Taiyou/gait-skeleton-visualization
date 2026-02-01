#!/usr/bin/env python3
"""
Compare Segment Extraction Methods

Compares old (velocity-only) vs new (direction + trim + overlap) methods.
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
    extract_segments_improved,
    compare_extraction_methods
)


def compute_velocity(data: np.ndarray, frame_rate: int = 60) -> np.ndarray:
    """Compute velocity from position data."""
    pelvis_pos = data[:, 0, :2]
    velocity = np.gradient(pelvis_pos, 1/frame_rate, axis=0)
    return velocity


def visualize_comparison(
    data: np.ndarray,
    velocity: np.ndarray,
    comparison: dict,
    output_path: Path,
    subject_name: str,
    frame_rate: int = 60
):
    """Visualize old vs new extraction methods."""
    fig = plt.figure(figsize=(20, 16))

    time = np.arange(len(data)) / frame_rate
    pelvis_x = data[:, 0, 0]
    pelvis_y = data[:, 0, 1]

    old_info = comparison['old']['info']
    new_info = comparison['new']['info']
    old_segments = comparison['old']['segments']
    new_segments = comparison['new']['segments']

    # ==========================================================================
    # Row 1: Trajectory with segments highlighted
    # ==========================================================================
    # Old method
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.scatter(pelvis_x, pelvis_y, c=time/60, cmap='viridis', s=0.5, alpha=0.3)

    for seg in old_segments:
        seg_x = seg.data[:, 0, 0]
        seg_y = seg.data[:, 0, 1]
        ax1.plot(seg_x, seg_y, 'r-', linewidth=1.5, alpha=0.7)

    ax1.set_title(f'Old Method: {len(old_segments)} segments\n(velocity threshold only)', fontsize=11)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # New method
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.scatter(pelvis_x, pelvis_y, c=time/60, cmap='viridis', s=0.5, alpha=0.3)

    # Color by parent segment
    parent_ids = list(set(s.parent_segment_id for s in new_segments))
    colors = plt.cm.tab20(np.linspace(0, 1, len(parent_ids)))
    color_map = {pid: colors[i] for i, pid in enumerate(parent_ids)}

    for seg in new_segments:
        seg_x = seg.data[:, 0, 0]
        seg_y = seg.data[:, 0, 1]
        ax2.plot(seg_x, seg_y, color=color_map[seg.parent_segment_id], linewidth=1.5, alpha=0.7)

    ax2.set_title(f'New Method: {len(new_segments)} segments\n(direction + trim + overlap)', fontsize=11)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # ==========================================================================
    # Row 2: Velocity and heading rate analysis
    # ==========================================================================
    ax3 = fig.add_subplot(3, 2, 3)
    velocity_mag = new_info['velocity_mag']
    ax3.plot(time, velocity_mag, 'b-', linewidth=0.5, alpha=0.7, label='Velocity')
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Old threshold (0.5)')
    ax3.axhline(0.4, color='orange', linestyle='--', alpha=0.5, label='New threshold (0.4)')

    # Highlight old segments
    for seg in old_segments:
        ax3.axvspan(seg.start_frame/frame_rate, seg.end_frame/frame_rate,
                    alpha=0.2, color='red')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity with Old Segments', fontsize=11)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, time[-1]])

    ax4 = fig.add_subplot(3, 2, 4)
    heading_rate = new_info['heading_rate']
    ax4.plot(time, np.degrees(heading_rate), 'g-', linewidth=0.5, alpha=0.7, label='Heading change rate')
    ax4.axhline(np.degrees(0.1), color='orange', linestyle='--', alpha=0.5, label='Threshold (0.1 rad)')

    # Highlight new full segments (before windowing)
    straight_mask = new_info['straight_mask']
    for i in range(len(time)-1):
        if straight_mask[i]:
            ax4.axvspan(time[i], time[i+1], alpha=0.1, color='green')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heading change (deg/frame)')
    ax4.set_title('Heading Change Rate (green = straight walking)', fontsize=11)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, time[-1]])
    ax4.set_ylim([0, min(30, np.percentile(np.degrees(heading_rate), 99))])

    # ==========================================================================
    # Row 3: Segment statistics comparison
    # ==========================================================================
    ax5 = fig.add_subplot(3, 2, 5)

    # Duration histogram
    old_durations = [s.duration_sec for s in old_segments]
    new_durations = [s.duration_sec for s in new_segments]

    bins = np.linspace(0, max(max(old_durations) if old_durations else 10,
                              max(new_durations) if new_durations else 10), 20)

    ax5.hist(old_durations, bins=bins, alpha=0.5, label=f'Old (n={len(old_segments)})', color='red')
    ax5.hist(new_durations, bins=bins, alpha=0.5, label=f'New (n={len(new_segments)})', color='green')
    ax5.set_xlabel('Segment Duration (s)')
    ax5.set_ylabel('Count')
    ax5.set_title('Segment Duration Distribution', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(3, 2, 6)

    # Distance histogram
    old_distances = [s.distance_m for s in old_segments]
    new_distances = [s.distance_m for s in new_segments]

    bins = np.linspace(0, max(max(old_distances) if old_distances else 10,
                              max(new_distances) if new_distances else 10), 20)

    ax6.hist(old_distances, bins=bins, alpha=0.5, label=f'Old (n={len(old_segments)})', color='red')
    ax6.hist(new_distances, bins=bins, alpha=0.5, label=f'New (n={len(new_segments)})', color='green')
    ax6.set_xlabel('Segment Distance (m)')
    ax6.set_ylabel('Count')
    ax6.set_title('Segment Distance Distribution', fontsize=11)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Summary text
    old_total_frames = sum(s.data.shape[0] for s in old_segments)
    new_total_frames = sum(s.data.shape[0] for s in new_segments)

    summary = f"""
    OLD METHOD                          NEW METHOD
    ──────────────────────────────────────────────────
    Segments: {len(old_segments):<25} Segments: {len(new_segments)}
    Total frames: {old_total_frames:<21} Total frames: {new_total_frames}
    Avg duration: {np.mean(old_durations) if old_durations else 0:.1f}s{' '*17} Avg duration: {np.mean(new_durations) if new_durations else 0:.1f}s
    Avg distance: {np.mean(old_distances) if old_distances else 0:.1f}m{' '*17} Avg distance: {np.mean(new_distances) if new_distances else 0:.1f}m

    Improvement: {len(new_segments) - len(old_segments):+d} segments ({(len(new_segments)/max(len(old_segments),1)-1)*100:+.1f}%)
    """

    plt.suptitle(f'{subject_name}: Segment Extraction Comparison', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, summary, ha='center', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    return {
        'old_segments': len(old_segments),
        'new_segments': len(new_segments),
        'old_frames': old_total_frames,
        'new_frames': new_total_frames,
        'improvement_pct': (len(new_segments) / max(len(old_segments), 1) - 1) * 100
    }


def process_all_subjects(output_dir: Path):
    """Process all subjects and create summary."""
    project_root = Path(__file__).parent.parent

    # Find all raw data folders
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
    print("COMPARING SEGMENT EXTRACTION METHODS")
    print("=" * 70)
    print(f"Found {len(all_files)} files")

    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "extraction_comparison"
    viz_dir.mkdir(exist_ok=True)

    results = []

    for i, input_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing: {input_path.stem}...")

        try:
            # Load and preprocess
            loader = load_xsens_data(input_path, frame_rate=60)
            original_data = loader.positions.copy()

            # Apply feature-preserving correction
            corrected = apply_feature_preserving_correction(
                original_data,
                frame_rate=60,
                drift_correction_strength='moderate'
            )

            # Compute velocity
            velocity = compute_velocity(corrected.data, frame_rate=60)

            # Compare methods
            comparison = compare_extraction_methods(corrected.data, velocity, frame_rate=60)

            # Visualize
            viz_path = viz_dir / f"{input_path.stem}_extraction_comparison.png"
            result = visualize_comparison(
                corrected.data,
                velocity,
                comparison,
                viz_path,
                input_path.stem,
                frame_rate=60
            )
            result['subject'] = input_path.stem
            results.append(result)

            print(f"    Old: {result['old_segments']} segments, New: {result['new_segments']} segments ({result['improvement_pct']:+.1f}%)")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                'subject': input_path.stem,
                'old_segments': 0,
                'new_segments': 0,
                'old_frames': 0,
                'new_frames': 0,
                'improvement_pct': 0,
                'error': str(e)
            })

    # Create summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid_results = [r for r in results if 'error' not in r]

    print(f"\n{'Subject':<15} {'Old Segs':<12} {'New Segs':<12} {'Change':<12}")
    print("-" * 55)

    total_old = 0
    total_new = 0
    for r in valid_results:
        print(f"{r['subject']:<15} {r['old_segments']:<12} {r['new_segments']:<12} {r['improvement_pct']:+.1f}%")
        total_old += r['old_segments']
        total_new += r['new_segments']

    print("-" * 55)
    print(f"{'TOTAL':<15} {total_old:<12} {total_new:<12} {(total_new/max(total_old,1)-1)*100:+.1f}%")

    # Save summary CSV
    df = pd.DataFrame(valid_results)
    summary_path = output_dir / "extraction_comparison_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Create overview visualization
    create_overview_visualization(valid_results, output_dir)

    return results


def create_overview_visualization(results: list, output_dir: Path):
    """Create overview bar chart comparing methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    subjects = [r['subject'] for r in results]
    old_segs = [r['old_segments'] for r in results]
    new_segs = [r['new_segments'] for r in results]
    improvements = [r['improvement_pct'] for r in results]

    x = np.arange(len(subjects))
    width = 0.35

    # Segment count comparison
    ax1 = axes[0]
    ax1.barh(x - width/2, old_segs, width, label='Old Method', color='salmon')
    ax1.barh(x + width/2, new_segs, width, label='New Method', color='lightgreen')
    ax1.set_yticks(x)
    ax1.set_yticklabels(subjects, fontsize=8)
    ax1.set_xlabel('Number of Segments')
    ax1.set_title('Segment Count: Old vs New Method', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Improvement percentage
    ax2 = axes[1]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.barh(x, improvements, color=colors, alpha=0.7)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(subjects, fontsize=8)
    ax2.set_xlabel('Improvement (%)')
    ax2.set_title('Segment Count Improvement', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # Summary stats
    total_old = sum(old_segs)
    total_new = sum(new_segs)
    avg_improvement = np.mean(improvements)

    fig.suptitle(
        f'Segment Extraction Comparison\n'
        f'Total: {total_old} → {total_new} segments ({(total_new/total_old-1)*100:+.1f}%), '
        f'Avg improvement: {avg_improvement:+.1f}%',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    overview_path = output_dir / "extraction_comparison_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Overview saved to: {overview_path}")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data/type2/lstm_pipeline_output"

    results = process_all_subjects(output_dir)


if __name__ == "__main__":
    main()
