#!/usr/bin/env python3
"""
Visualize Aligned Segments

- Align each segment to X-axis direction (straight line)
- Filter out segments shorter than 7m
- Show clean, aligned trajectories
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def align_segment_to_x_axis(segment_data: np.ndarray) -> np.ndarray:
    """
    Align a segment so that the main walking direction is along X-axis.

    Args:
        segment_data: (n_frames, 3) array of X, Y, Z positions

    Returns:
        Aligned (n_frames, 3) array
    """
    x = segment_data[:, 0]
    y = segment_data[:, 1]
    z = segment_data[:, 2]

    # Use PCA to find principal direction
    xy_data = np.column_stack([x, y])
    pca = PCA(n_components=2)
    pca.fit(xy_data)

    # Calculate rotation angle to align with X-axis
    principal_axis = pca.components_[0]
    angle = np.arctan2(principal_axis[1], principal_axis[0])

    # Rotate to align with X-axis
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    # Center first
    x_centered = x - x[0]
    y_centered = y - y[0]

    # Rotate
    x_rotated = x_centered * cos_a - y_centered * sin_a
    y_rotated = x_centered * sin_a + y_centered * cos_a

    # Ensure walking direction is positive X
    if x_rotated[-1] < x_rotated[0]:
        x_rotated = -x_rotated
        y_rotated = -y_rotated

    # Center Y around 0
    y_rotated = y_rotated - np.mean(y_rotated)

    return np.column_stack([x_rotated, y_rotated, z - z[0]])


def calculate_segment_length(segment_data: np.ndarray) -> float:
    """Calculate the total walking distance of a segment."""
    x = segment_data[:, 0]
    return abs(x[-1] - x[0])


def filter_and_align_segments(segments: list, min_length_m: float = 7.0) -> list:
    """Filter segments by length and align to X-axis."""
    aligned_segments = []

    for seg in segments:
        # Align first
        aligned_data = align_segment_to_x_axis(seg['data'])

        # Calculate length
        length = calculate_segment_length(aligned_data)

        if length >= min_length_m:
            aligned_segments.append({
                'file': seg['file'],
                'data': aligned_data,
                'n_frames': seg['n_frames'],
                'length_m': length,
            })

    return aligned_segments


def visualize_subject_aligned(subject_name: str, segments: list, output_path: Path):
    """Create visualization for aligned segments."""
    n_segments = len(segments)

    if n_segments == 0:
        print(f"  No valid segments (>= 7m) for {subject_name}")
        return None

    fig = plt.figure(figsize=(16, 10))

    # Top: All segments overlaid
    ax1 = fig.add_subplot(2, 1, 1)

    colors = plt.cm.viridis(np.linspace(0, 1, n_segments))

    for i, seg in enumerate(segments):
        x = seg['data'][:, 0]
        y = seg['data'][:, 1]
        ax1.plot(x, y, color=colors[i], linewidth=0.8, alpha=0.7)

    ax1.set_xlabel('X - Walking Direction (m)', fontsize=12)
    ax1.set_ylabel('Y - Lateral Deviation (m)', fontsize=12)
    ax1.set_title(f'{subject_name}: {n_segments} Aligned Segments (≥7m)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_aspect('equal')

    # Add colorbar to show segment order
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=1, vmax=n_segments))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Segment #')

    # Bottom: Combined continuous trajectory
    ax2 = fig.add_subplot(2, 1, 2)

    all_x = []
    all_y = []
    segment_ids = []

    x_offset = 0
    for i, seg in enumerate(segments):
        x = seg['data'][:, 0] + x_offset
        y = seg['data'][:, 1]

        all_x.extend(x)
        all_y.extend(y)
        segment_ids.extend([i] * len(x))

        x_offset = x[-1] + 0.5  # Small gap between segments

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    segment_ids = np.array(segment_ids)

    ax2.scatter(all_x, all_y, c=segment_ids, cmap='tab20', s=0.5, alpha=0.5)

    total_distance = all_x[-1]
    total_frames = sum(s['n_frames'] for s in segments)

    ax2.set_xlabel('X - Cumulative Walking Distance (m)', fontsize=12)
    ax2.set_ylabel('Y - Lateral Deviation (m)', fontsize=12)
    ax2.set_title(f'Combined: {total_distance:.1f}m total distance, {total_frames} frames', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Set aspect ratio to show more horizontal
    ax2.set_aspect(10)  # Stretch horizontally

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'subject': subject_name,
        'n_segments': n_segments,
        'total_frames': total_frames,
        'total_distance_m': total_distance,
        'avg_length_m': np.mean([s['length_m'] for s in segments]),
    }


def create_overview(all_results: list, all_segments_data: dict, output_dir: Path):
    """Create overview of all subjects with aligned segments."""
    valid_results = [r for r in all_results if r is not None]
    n_subjects = len(valid_results)

    # Grid layout
    n_cols = 5
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, result in enumerate(valid_results):
        ax = axes[i]
        subject = result['subject']
        segments = all_segments_data.get(subject, [])

        if not segments:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(subject, fontsize=9)
            continue

        # Plot all aligned segments overlaid
        colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))

        for j, seg in enumerate(segments):
            x = seg['data'][:, 0]
            y = seg['data'][:, 1]
            ax.plot(x, y, color=colors[j], linewidth=0.5, alpha=0.6)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_title(f"{subject}\n{result['n_segments']} segs, {result['total_distance_m']:.0f}m", fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X (m)', fontsize=6)
        ax.set_ylabel('Y (m)', fontsize=6)

    # Hide unused axes
    for i in range(len(valid_results), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('All Subjects: Aligned Straight-Line Segments (≥7m, Turns Excluded)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    overview_path = output_dir / "all_subjects_aligned_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nOverview saved to: {overview_path}")

    # Also create a combined view showing all subjects' segments
    fig2 = plt.figure(figsize=(20, 12))
    ax = fig2.add_subplot(111)

    y_offset = 0
    subject_labels = []
    subject_y_positions = []

    for result in valid_results:
        subject = result['subject']
        segments = all_segments_data.get(subject, [])

        if not segments:
            continue

        subject_labels.append(subject)
        subject_y_positions.append(y_offset)

        x_offset = 0
        for seg in segments:
            x = seg['data'][:, 0] + x_offset
            y = seg['data'][:, 1] * 0.3 + y_offset  # Scale Y for visibility

            frames = np.arange(len(x))
            ax.scatter(x, y, c=frames, cmap='viridis', s=0.3, alpha=0.5)

            x_offset = x[-1] + 0.3

        y_offset -= 2  # Spacing between subjects

    ax.set_yticks(subject_y_positions)
    ax.set_yticklabels(subject_labels, fontsize=8)
    ax.set_xlabel('Walking Distance (m)', fontsize=12)
    ax.set_title('All Subjects: Aligned Walking Trajectories (≥7m segments)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    combined_path = output_dir / "all_subjects_combined_aligned.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Combined view saved to: {combined_path}")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data/type2/lstm_pipeline_output"
    segments_base = output_dir / "segments"

    MIN_SEGMENT_LENGTH = 7.0  # meters

    if not segments_base.exists():
        print(f"Segments directory not found: {segments_base}")
        return

    subject_dirs = sorted([d for d in segments_base.iterdir() if d.is_dir()])

    print("=" * 70)
    print("VISUALIZING ALIGNED SEGMENTS (≥7m only)")
    print("=" * 70)
    print(f"Found {len(subject_dirs)} subjects")
    print(f"Minimum segment length: {MIN_SEGMENT_LENGTH}m")

    viz_dir = output_dir / "aligned_segment_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_segments_data = {}

    total_original_segments = 0
    total_filtered_segments = 0

    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        print(f"\nProcessing: {subject_name}...")

        # Load original segments
        segments = load_segments(subject_dir)
        total_original_segments += len(segments)

        # Filter and align
        aligned_segments = filter_and_align_segments(segments, min_length_m=MIN_SEGMENT_LENGTH)
        total_filtered_segments += len(aligned_segments)

        all_segments_data[subject_name] = aligned_segments

        if aligned_segments:
            output_path = viz_dir / f"{subject_name}_aligned.png"
            result = visualize_subject_aligned(subject_name, aligned_segments, output_path)
            all_results.append(result)

            removed = len(segments) - len(aligned_segments)
            print(f"  {len(aligned_segments)} segments kept, {removed} removed (<7m)")
            print(f"  Total distance: {result['total_distance_m']:.1f}m")
        else:
            print(f"  All segments removed (all <7m)")
            all_results.append(None)

    # Create overview
    print("\n" + "=" * 70)
    print("Creating overview visualizations...")
    create_overview(all_results, all_segments_data, output_dir)

    # Print summary
    valid_results = [r for r in all_results if r is not None]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original segments: {total_original_segments}")
    print(f"After filtering (≥7m): {total_filtered_segments}")
    print(f"Removed: {total_original_segments - total_filtered_segments} ({(total_original_segments - total_filtered_segments)/total_original_segments*100:.1f}%)")
    print()
    print(f"{'Subject':<15} {'Segments':<12} {'Frames':<12} {'Distance (m)':<15} {'Avg Len (m)':<12}")
    print("-" * 70)

    total_segs = 0
    total_frames = 0
    total_dist = 0

    for r in valid_results:
        print(f"{r['subject']:<15} {r['n_segments']:<12} {r['total_frames']:<12} {r['total_distance_m']:<15.1f} {r['avg_length_m']:<12.1f}")
        total_segs += r['n_segments']
        total_frames += r['total_frames']
        total_dist += r['total_distance_m']

    print("-" * 70)
    print(f"{'TOTAL':<15} {total_segs:<12} {total_frames:<12} {total_dist:<15.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
