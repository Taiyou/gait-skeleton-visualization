#!/usr/bin/env python3
"""
Visualize Extracted Segments Only

Show trajectories composed only of usable (straight-line walking) segments,
excluding turns and stationary periods.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_segments(segments_dir: Path) -> list:
    """Load all segment CSVs from a directory."""
    segment_files = sorted(segments_dir.glob("*.csv"))
    segments = []

    for f in segment_files:
        df = pd.read_csv(f)
        # Extract Pelvis position (first 3 columns: Pelvis x, Pelvis y, Pelvis z)
        pelvis_cols = [c for c in df.columns if 'Pelvis' in c]
        if len(pelvis_cols) >= 3:
            pelvis_data = df[pelvis_cols].values
            segments.append({
                'file': f.name,
                'data': pelvis_data,
                'n_frames': len(df),
            })

    return segments


def visualize_subject_segments(subject_name: str, segments: list, output_path: Path):
    """Create visualization for a single subject's segments."""
    n_segments = len(segments)

    if n_segments == 0:
        print(f"  No segments found for {subject_name}")
        return

    # Calculate grid size
    n_cols = min(6, n_segments)
    n_rows = (n_segments + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows + 2))

    # Also create a combined trajectory plot
    all_x = []
    all_y = []
    segment_colors = []

    for i, seg in enumerate(segments):
        ax = fig.add_subplot(n_rows + 1, n_cols, i + 1)

        x = seg['data'][:, 0]
        y = seg['data'][:, 1]
        frames = np.arange(len(x))

        # Color by time within segment
        ax.scatter(x, y, c=frames, cmap='viridis', s=2, alpha=0.7)
        ax.plot(x[0], y[0], 'go', markersize=6)
        ax.plot(x[-1], y[-1], 'rs', markersize=6)

        ax.set_title(f'Seg {i+1}\n{seg["n_frames"]} frames', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        # Collect for combined plot
        # Shift each segment to connect to previous
        if all_x:
            x_offset = all_x[-1] - x[0]
            y_offset = all_y[-1] - y[0]
        else:
            x_offset = 0
            y_offset = 0

        all_x.extend(x + x_offset)
        all_y.extend(y + y_offset)
        segment_colors.extend([i] * len(x))

    # Combined trajectory (bottom row, spanning all columns)
    ax_combined = fig.add_subplot(n_rows + 1, 1, n_rows + 1)

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    segment_colors = np.array(segment_colors)

    scatter = ax_combined.scatter(all_x, all_y, c=segment_colors, cmap='tab20', s=1, alpha=0.5)
    ax_combined.plot(all_x[0], all_y[0], 'go', markersize=10, label='Start')
    ax_combined.plot(all_x[-1], all_y[-1], 'rs', markersize=10, label='End')

    total_frames = sum(s['n_frames'] for s in segments)
    ax_combined.set_title(f'Combined: {n_segments} segments, {total_frames} total frames', fontsize=12, fontweight='bold')
    ax_combined.set_xlabel('X (m)')
    ax_combined.set_ylabel('Y (m)')
    ax_combined.set_aspect('equal')
    ax_combined.legend(loc='upper right')
    ax_combined.grid(True, alpha=0.3)

    plt.suptitle(f'{subject_name}: Extracted Straight-Line Segments Only', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'subject': subject_name,
        'n_segments': n_segments,
        'total_frames': total_frames,
    }


def create_all_subjects_overview(all_results: list, output_dir: Path):
    """Create overview visualization of all subjects."""
    n_subjects = len(all_results)

    # Calculate grid
    n_cols = 5
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, result in enumerate(all_results):
        ax = axes[i]

        segments_dir = output_dir / "segments" / result['subject']
        segments = load_segments(segments_dir)

        if not segments:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(result['subject'], fontsize=10)
            continue

        # Combine all segments into continuous trajectory
        all_x = []
        all_y = []

        for seg in segments:
            x = seg['data'][:, 0]
            y = seg['data'][:, 1]

            if all_x:
                x_offset = all_x[-1] - x[0]
                y_offset = all_y[-1] - y[0]
            else:
                x_offset = 0
                y_offset = 0

            all_x.extend(x + x_offset)
            all_y.extend(y + y_offset)

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        # Color by time
        colors = np.arange(len(all_x))
        ax.scatter(all_x, all_y, c=colors, cmap='viridis', s=0.5, alpha=0.5)
        ax.plot(all_x[0], all_y[0], 'go', markersize=5)
        ax.plot(all_x[-1], all_y[-1], 'rs', markersize=5)

        ax.set_title(f"{result['subject']}\n{result['n_segments']} segs, {result['total_frames']} frames", fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(len(all_results), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('All Subjects: Usable Segments Only (Turns Excluded)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    overview_path = output_dir / "all_subjects_segments_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nOverview saved to: {overview_path}")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data/type2/lstm_pipeline_output"
    segments_base = output_dir / "segments"

    if not segments_base.exists():
        print(f"Segments directory not found: {segments_base}")
        return

    # Get all subject directories
    subject_dirs = sorted([d for d in segments_base.iterdir() if d.is_dir()])

    print("=" * 70)
    print("VISUALIZING EXTRACTED SEGMENTS")
    print("=" * 70)
    print(f"Found {len(subject_dirs)} subjects")

    # Create output directory for visualizations
    viz_dir = output_dir / "segment_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        print(f"\nProcessing: {subject_name}...")

        segments = load_segments(subject_dir)

        if segments:
            output_path = viz_dir / f"{subject_name}_segments.png"
            result = visualize_subject_segments(subject_name, segments, output_path)
            all_results.append(result)
            print(f"  {result['n_segments']} segments, {result['total_frames']} frames")
            print(f"  Saved to: {output_path}")
        else:
            print(f"  No segments found")

    # Create overview
    if all_results:
        print("\n" + "=" * 70)
        print("Creating overview visualization...")
        create_all_subjects_overview(all_results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Subject':<15} {'Segments':<12} {'Frames':<12}")
    print("-" * 40)

    total_segments = 0
    total_frames = 0
    for r in all_results:
        print(f"{r['subject']:<15} {r['n_segments']:<12} {r['total_frames']:<12}")
        total_segments += r['n_segments']
        total_frames += r['total_frames']

    print("-" * 40)
    print(f"{'TOTAL':<15} {total_segments:<12} {total_frames:<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
