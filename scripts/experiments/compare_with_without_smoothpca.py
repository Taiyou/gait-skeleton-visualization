#!/usr/bin/env python3
"""
Compare Segment Extraction: With vs Without Smooth PCA

Test if skipping Smooth PCA correction improves results for some subjects.
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


def simple_drift_correction(data: np.ndarray, frame_rate: int = 60) -> np.ndarray:
    """
    Simple drift correction without Smooth PCA.
    Only applies:
    1. Global rotation alignment (single PCA at end)
    2. Y-drift removal with highpass filter
    """
    from sklearn.decomposition import PCA
    from scipy.signal import butter, filtfilt

    corrected = data.copy()
    n_frames, n_segments, _ = data.shape

    pelvis_x = data[:, 0, 0]
    pelvis_y = data[:, 0, 1]

    # Step 1: Single global PCA rotation (no time-varying correction)
    xy_data = np.column_stack([pelvis_x, pelvis_y])
    center = np.mean(xy_data, axis=0)
    xy_centered = xy_data - center

    pca = PCA(n_components=2)
    pca.fit(xy_centered)

    principal_axis = pca.components_[0]
    global_angle = np.arctan2(principal_axis[1], principal_axis[0])

    cos_a = np.cos(-global_angle)
    sin_a = np.sin(-global_angle)

    for body_idx in range(n_segments):
        x = corrected[:, body_idx, 0] - center[0]
        y = corrected[:, body_idx, 1] - center[1]
        corrected[:, body_idx, 0] = x * cos_a - y * sin_a
        corrected[:, body_idx, 1] = x * sin_a + y * cos_a

    # Step 2: Y-drift removal with highpass filter
    try:
        nyq = 0.5 * frame_rate
        cutoff = 0.1 / nyq
        b, a = butter(2, cutoff, btype='high')

        pelvis_y_rotated = corrected[:, 0, 1].copy()
        y_highpassed = filtfilt(b, a, pelvis_y_rotated)
        y_drift = pelvis_y_rotated - y_highpassed

        for body_idx in range(n_segments):
            corrected[:, body_idx, 1] -= y_drift
    except:
        pass  # Skip if filter fails

    # Center Y at 0
    final_median = np.median(corrected[:, 0, 1])
    for body_idx in range(n_segments):
        corrected[:, body_idx, 1] -= final_median

    return corrected


def process_subject(input_path: Path, frame_rate: int = 60):
    """Process a single subject with both methods."""

    # Load data
    loader = load_xsens_data(input_path, frame_rate=frame_rate)
    original_data = loader.positions.copy()

    # Method 1: With Smooth PCA (feature preserving)
    corrected_with_pca = apply_feature_preserving_correction(
        original_data.copy(),
        frame_rate=frame_rate,
        drift_correction_strength='moderate'
    )

    # Method 2: Without Smooth PCA (simple correction)
    corrected_without_pca = simple_drift_correction(original_data.copy(), frame_rate)

    # Extraction parameters
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

    # Extract segments - With PCA
    velocity_with = compute_velocity(corrected_with_pca.data, frame_rate)
    segments_with, info_with = extract_segments_improved(
        corrected_with_pca.data, velocity_with, params
    )

    # Extract segments - Without PCA
    velocity_without = compute_velocity(corrected_without_pca, frame_rate)
    segments_without, info_without = extract_segments_improved(
        corrected_without_pca, velocity_without, params
    )

    return {
        'original': original_data,
        'with_pca': {
            'data': corrected_with_pca.data,
            'segments': segments_with,
            'info': info_with,
            'velocity': velocity_with,
        },
        'without_pca': {
            'data': corrected_without_pca,
            'segments': segments_without,
            'info': info_without,
            'velocity': velocity_without,
        }
    }


def visualize_comparison(
    results: dict,
    output_path: Path,
    subject_name: str,
    frame_rate: int = 60
):
    """Visualize with vs without Smooth PCA."""
    fig = plt.figure(figsize=(20, 12))

    original = results['original']
    with_pca = results['with_pca']
    without_pca = results['without_pca']

    time = np.arange(len(original)) / frame_rate

    # Row 1: Trajectories
    # Original
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(original[:, 0, 0], original[:, 0, 1],
                c=time/60, cmap='viridis', s=0.5, alpha=0.5)
    orig_y_range = original[:, 0, 1].max() - original[:, 0, 1].min()
    ax1.set_title(f'Original\nY range: {orig_y_range:.2f}m', fontsize=11)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # With Smooth PCA
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(with_pca['data'][:, 0, 0], with_pca['data'][:, 0, 1],
                c=time/60, cmap='viridis', s=0.5, alpha=0.3)
    for seg in with_pca['segments']:
        ax2.plot(seg.data[:, 0, 0], seg.data[:, 0, 1], 'r-', linewidth=1, alpha=0.7)
    with_y_range = with_pca['data'][:, 0, 1].max() - with_pca['data'][:, 0, 1].min()
    ax2.set_title(f'With Smooth PCA: {len(with_pca["segments"])} segs\nY range: {with_y_range:.2f}m', fontsize=11)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # Without Smooth PCA
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(without_pca['data'][:, 0, 0], without_pca['data'][:, 0, 1],
                c=time/60, cmap='viridis', s=0.5, alpha=0.3)
    for seg in without_pca['segments']:
        ax3.plot(seg.data[:, 0, 0], seg.data[:, 0, 1], 'g-', linewidth=1, alpha=0.7)
    without_y_range = without_pca['data'][:, 0, 1].max() - without_pca['data'][:, 0, 1].min()
    ax3.set_title(f'Without Smooth PCA: {len(without_pca["segments"])} segs\nY range: {without_y_range:.2f}m', fontsize=11)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)

    # Row 2: Heading change rate comparison
    ax4 = fig.add_subplot(2, 3, 4)
    heading_with = with_pca['info']['heading_rate']
    heading_without = without_pca['info']['heading_rate']
    ax4.plot(time, np.degrees(heading_with), 'r-', linewidth=0.5, alpha=0.7, label='With PCA')
    ax4.plot(time, np.degrees(heading_without), 'g-', linewidth=0.5, alpha=0.7, label='Without PCA')
    ax4.axhline(np.degrees(0.1), color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heading change (deg/frame)')
    ax4.set_title('Heading Change Rate Comparison', fontsize=11)
    ax4.legend()
    ax4.set_ylim([0, min(30, max(np.percentile(np.degrees(heading_with), 99),
                                 np.percentile(np.degrees(heading_without), 99)))])
    ax4.grid(True, alpha=0.3)

    # Row 2: Straight walking detection
    ax5 = fig.add_subplot(2, 3, 5)
    straight_with = with_pca['info']['straight_mask'].astype(float)
    straight_without = without_pca['info']['straight_mask'].astype(float)
    ax5.fill_between(time, 0, straight_with, alpha=0.5, color='red', label='With PCA')
    ax5.fill_between(time, 0, -straight_without, alpha=0.5, color='green', label='Without PCA')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Straight walking detected')
    ax5.set_title('Straight Walking Detection', fontsize=11)
    ax5.legend()
    ax5.set_ylim([-1.5, 1.5])
    ax5.grid(True, alpha=0.3)

    # Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    with_frames = sum(s.data.shape[0] for s in with_pca['segments'])
    without_frames = sum(s.data.shape[0] for s in without_pca['segments'])

    summary_text = f"""
    COMPARISON SUMMARY
    ══════════════════════════════════════════

                        With PCA    Without PCA
    ──────────────────────────────────────────
    Segments:           {len(with_pca['segments']):<12}{len(without_pca['segments'])}
    Total frames:       {with_frames:<12}{without_frames}
    Y-axis range:       {with_y_range:<12.2f}{without_y_range:.2f}
    Straight ratio:     {with_pca['info']['straight_ratio']*100:<12.1f}{without_pca['info']['straight_ratio']*100:.1f}%

    ══════════════════════════════════════════
    Difference: {len(without_pca['segments']) - len(with_pca['segments']):+d} segments

    Better method: {'WITHOUT PCA' if len(without_pca['segments']) > len(with_pca['segments']) else 'WITH PCA'}
    """

    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{subject_name}: With vs Without Smooth PCA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'subject': subject_name,
        'with_pca_segments': len(with_pca['segments']),
        'without_pca_segments': len(without_pca['segments']),
        'with_pca_frames': with_frames,
        'without_pca_frames': without_frames,
        'with_pca_y_range': with_y_range,
        'without_pca_y_range': without_y_range,
        'better': 'without' if len(without_pca['segments']) > len(with_pca['segments']) else 'with',
        'difference': len(without_pca['segments']) - len(with_pca['segments']),
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
    print("COMPARING: WITH vs WITHOUT SMOOTH PCA")
    print("=" * 70)
    print(f"Found {len(all_files)} files")

    output_dir = project_root / "data/type2/lstm_pipeline_output/pca_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, input_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing: {input_path.stem}...")

        try:
            result = process_subject(input_path, frame_rate=60)

            viz_path = output_dir / f"{input_path.stem}_pca_comparison.png"
            stats = visualize_comparison(result, viz_path, input_path.stem, frame_rate=60)
            results.append(stats)

            print(f"    With PCA: {stats['with_pca_segments']} segs, Without PCA: {stats['without_pca_segments']} segs ({stats['difference']:+d})")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Subject':<15} {'With PCA':<12} {'Without PCA':<12} {'Diff':<10} {'Better':<10}")
    print("-" * 60)

    better_with = 0
    better_without = 0
    total_with = 0
    total_without = 0

    for r in results:
        print(f"{r['subject']:<15} {r['with_pca_segments']:<12} {r['without_pca_segments']:<12} {r['difference']:+d}{'':<6} {r['better'].upper()}")
        total_with += r['with_pca_segments']
        total_without += r['without_pca_segments']
        if r['better'] == 'with':
            better_with += 1
        else:
            better_without += 1

    print("-" * 60)
    print(f"{'TOTAL':<15} {total_with:<12} {total_without:<12} {total_without - total_with:+d}")
    print(f"\nBetter with PCA: {better_with} subjects")
    print(f"Better without PCA: {better_without} subjects")

    # Save summary
    df = pd.DataFrame(results)
    summary_path = output_dir / "pca_comparison_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Create overview
    fig, ax = plt.subplots(figsize=(12, 10))

    subjects = [r['subject'] for r in results]
    with_segs = [r['with_pca_segments'] for r in results]
    without_segs = [r['without_pca_segments'] for r in results]

    x = np.arange(len(subjects))
    width = 0.35

    ax.barh(x - width/2, with_segs, width, label='With Smooth PCA', color='salmon')
    ax.barh(x + width/2, without_segs, width, label='Without Smooth PCA', color='lightgreen')

    ax.set_yticks(x)
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel('Number of Segments')
    ax.set_title(f'Segment Count: With vs Without Smooth PCA\n'
                 f'Total: {total_with} vs {total_without} ({total_without - total_with:+d}), '
                 f'Better without: {better_without}/{len(results)}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    overview_path = output_dir / "pca_comparison_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Overview saved to: {overview_path}")


if __name__ == "__main__":
    main()
