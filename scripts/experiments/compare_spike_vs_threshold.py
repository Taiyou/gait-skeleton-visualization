#!/usr/bin/env python3
"""
Compare Segment Extraction Methods:
1. Threshold-based on RAW data
2. Threshold-based on PREPROCESSED data
3. Spike-based on RAW data (NEW)
4. Spike-based on PREPROCESSED data (NEW)

Spike-based detection uses peak prominence to find turns,
which should be more robust to baseline drift.
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
from scripts.gait_analysis.spike_based_segment_extraction import (
    SpikeExtractionParams,
    extract_segments_spike_based
)


def compute_velocity(data: np.ndarray, frame_rate: int = 60) -> np.ndarray:
    """Compute velocity from position data."""
    pelvis_pos = data[:, 0, :2]
    velocity = np.gradient(pelvis_pos, 1/frame_rate, axis=0)
    return velocity


def process_subject(input_path: Path, frame_rate: int = 60):
    """Process a single subject with all methods."""

    # Load data
    loader = load_xsens_data(input_path, frame_rate=frame_rate)
    raw_data = loader.positions.copy()

    # Preprocess
    preprocessed_result = apply_feature_preserving_correction(
        raw_data.copy(),
        frame_rate=frame_rate,
        drift_correction_strength='moderate'
    )
    preprocessed_data = preprocessed_result.data

    # Velocities
    velocity_raw = compute_velocity(raw_data, frame_rate)
    velocity_preprocessed = compute_velocity(preprocessed_data, frame_rate)

    results = {}

    # =========================================================================
    # Method 1: Threshold-based on RAW data
    # =========================================================================
    params_threshold_raw = SegmentExtractionParams(
        velocity_threshold=0.4,
        heading_change_threshold=0.3,  # Higher for raw data
        trim_start_seconds=0.7,
        trim_end_seconds=0.5,
        use_overlapping_windows=True,
        window_seconds=5.0,
        window_overlap=0.5,
        min_segment_meters=5.0,
        frame_rate=frame_rate
    )
    segments, info = extract_segments_improved(raw_data, velocity_raw, params_threshold_raw)
    results['threshold_raw'] = {
        'data': raw_data,
        'segments': segments,
        'info': info,
        'method': 'Threshold (Raw)',
        'color': 'red',
    }

    # =========================================================================
    # Method 2: Threshold-based on PREPROCESSED data
    # =========================================================================
    params_threshold_prep = SegmentExtractionParams(
        velocity_threshold=0.4,
        heading_change_threshold=0.1,  # Lower for preprocessed
        trim_start_seconds=0.5,
        trim_end_seconds=0.3,
        use_overlapping_windows=True,
        window_seconds=5.0,
        window_overlap=0.5,
        min_segment_meters=5.0,
        frame_rate=frame_rate
    )
    segments, info = extract_segments_improved(preprocessed_data, velocity_preprocessed, params_threshold_prep)
    results['threshold_prep'] = {
        'data': preprocessed_data,
        'segments': segments,
        'info': info,
        'method': 'Threshold (Preprocessed)',
        'color': 'green',
    }

    # =========================================================================
    # Method 3: Spike-based on RAW data
    # =========================================================================
    params_spike_raw = SpikeExtractionParams(
        velocity_threshold=0.4,
        spike_prominence=0.03,  # Lower prominence for raw data (more sensitive)
        spike_width_seconds=0.3,
        spike_distance_seconds=1.0,
        exclude_before_spike_seconds=0.5,
        exclude_after_spike_seconds=0.5,
        trim_start_seconds=0.5,
        trim_end_seconds=0.3,
        use_overlapping_windows=True,
        window_seconds=5.0,
        window_overlap=0.5,
        min_segment_meters=5.0,
        frame_rate=frame_rate
    )
    segments, info = extract_segments_spike_based(raw_data, velocity_raw, params_spike_raw)
    results['spike_raw'] = {
        'data': raw_data,
        'segments': segments,
        'info': info,
        'method': 'Spike (Raw)',
        'color': 'orange',
    }

    # =========================================================================
    # Method 4: Spike-based on PREPROCESSED data
    # =========================================================================
    params_spike_prep = SpikeExtractionParams(
        velocity_threshold=0.4,
        spike_prominence=0.05,  # Standard prominence
        spike_width_seconds=0.3,
        spike_distance_seconds=1.0,
        exclude_before_spike_seconds=0.5,
        exclude_after_spike_seconds=0.5,
        trim_start_seconds=0.5,
        trim_end_seconds=0.3,
        use_overlapping_windows=True,
        window_seconds=5.0,
        window_overlap=0.5,
        min_segment_meters=5.0,
        frame_rate=frame_rate
    )
    segments, info = extract_segments_spike_based(preprocessed_data, velocity_preprocessed, params_spike_prep)
    results['spike_prep'] = {
        'data': preprocessed_data,
        'segments': segments,
        'info': info,
        'method': 'Spike (Preprocessed)',
        'color': 'blue',
    }

    return results


def visualize_comparison(results: dict, output_path: Path, subject_name: str, frame_rate: int = 60):
    """Visualize all methods comparison."""
    fig = plt.figure(figsize=(24, 16))

    methods = ['threshold_raw', 'threshold_prep', 'spike_raw', 'spike_prep']
    method_names = ['Threshold (Raw)', 'Threshold (Prep)', 'Spike (Raw)', 'Spike (Prep)']
    colors = ['red', 'green', 'orange', 'blue']

    # Get time axis
    n_frames = len(results['threshold_raw']['data'])
    time = np.arange(n_frames) / frame_rate

    # Row 1: Trajectories with segments
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = fig.add_subplot(4, 4, i + 1)
        r = results[method]
        data = r['data']

        # Plot full trajectory
        ax.scatter(data[:, 0, 0], data[:, 0, 1], c=time/60, cmap='gray', s=0.3, alpha=0.3)

        # Plot segments
        for seg in r['segments']:
            ax.plot(seg.data[:, 0, 0], seg.data[:, 0, 1], color=color, linewidth=1.5, alpha=0.7)

        n_segs = len(r['segments'])
        ax.set_title(f'{name}\n{n_segs} segments', fontsize=11, color=color, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    # Row 2: Heading change rate with detection visualization
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = fig.add_subplot(4, 4, 4 + i + 1)
        r = results[method]
        heading_rate = r['info']['heading_rate']

        ax.plot(time, np.degrees(heading_rate), color=color, linewidth=0.5, alpha=0.7)

        # Show detected spikes for spike-based methods
        if 'spike' in method and 'spikes' in r['info']:
            spikes = r['info']['spikes']
            if len(spikes) > 0:
                ax.scatter(spikes / frame_rate, np.degrees(heading_rate[spikes]),
                          color='black', s=30, zorder=5, marker='v', label=f'{len(spikes)} spikes')
                ax.legend(loc='upper right', fontsize=8)

        # Highlight segments
        for seg in r['segments']:
            ax.axvspan(seg.start_frame/frame_rate, seg.end_frame/frame_rate,
                      alpha=0.15, color=color)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading rate (deg/frame)')
        ax.set_title(f'{name}: Heading Change', fontsize=10)
        ax.set_ylim([0, min(30, np.percentile(np.degrees(heading_rate), 99))])
        ax.grid(True, alpha=0.3)

    # Row 3: Exclusion zones (for spike-based) or threshold lines (for threshold-based)
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = fig.add_subplot(4, 4, 8 + i + 1)
        r = results[method]

        if 'spike' in method:
            # Show exclusion mask
            if 'exclude_mask' in r['info']:
                exclude = r['info']['exclude_mask'].astype(float)
                ax.fill_between(time, 0, exclude, alpha=0.3, color='red', label='Excluded (turns)')
                ax.fill_between(time, 0, 1-exclude, alpha=0.3, color='green', label='Included')
                ax.set_ylabel('Region')
                ax.set_title(f'{name}: Turn Exclusion Zones', fontsize=10)
                ax.legend(loc='upper right', fontsize=8)
        else:
            # Show threshold comparison
            heading_rate = r['info']['heading_rate']
            if 'raw' in method:
                thresh = 0.3
            else:
                thresh = 0.1

            above = np.degrees(heading_rate) > np.degrees(thresh)
            ax.fill_between(time, 0, above.astype(float), alpha=0.3, color='red', label='Above threshold')
            ax.fill_between(time, 0, (~above).astype(float), alpha=0.3, color='green', label='Below threshold')
            ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Region')
            ax.set_title(f'{name}: Threshold Regions (thresh={np.degrees(thresh):.1f}°)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)

        ax.set_xlabel('Time (s)')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)

    # Row 4: Summary bar chart and statistics
    ax_bar = fig.add_subplot(4, 2, 7)
    segment_counts = [len(results[m]['segments']) for m in methods]
    bars = ax_bar.bar(method_names, segment_counts, color=colors, alpha=0.7)
    ax_bar.set_ylabel('Number of Segments')
    ax_bar.set_title('Segment Count Comparison', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, count in zip(bars, segment_counts):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax_bar.grid(True, alpha=0.3, axis='y')

    # Statistics text
    ax_text = fig.add_subplot(4, 2, 8)
    ax_text.axis('off')

    # Find best method
    best_idx = np.argmax(segment_counts)
    best_method = method_names[best_idx]

    stats_text = f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    COMPARISON SUMMARY                          ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Method                      Segments    Straight %            ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Threshold (Raw)             {len(results['threshold_raw']['segments']):<10}  {results['threshold_raw']['info']['straight_ratio']*100:>6.1f}%              ║
    ║  Threshold (Preprocessed)    {len(results['threshold_prep']['segments']):<10}  {results['threshold_prep']['info']['straight_ratio']*100:>6.1f}%              ║
    ║  Spike (Raw)                 {len(results['spike_raw']['segments']):<10}  {results['spike_raw']['info']['straight_ratio']*100:>6.1f}%              ║
    ║  Spike (Preprocessed)        {len(results['spike_prep']['segments']):<10}  {results['spike_prep']['info']['straight_ratio']*100:>6.1f}%              ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Spikes detected (Raw):      {results['spike_raw']['info'].get('n_spikes_detected', 0):<10}                        ║
    ║  Spikes detected (Prep):     {results['spike_prep']['info'].get('n_spikes_detected', 0):<10}                        ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  BEST METHOD: {best_method:<48} ║
    ╚════════════════════════════════════════════════════════════════╝
    """

    ax_text.text(0.05, 0.5, stats_text, transform=ax_text.transAxes, fontsize=10,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'{subject_name}: Threshold vs Spike-Based Segment Extraction',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'subject': subject_name,
        'threshold_raw': len(results['threshold_raw']['segments']),
        'threshold_prep': len(results['threshold_prep']['segments']),
        'spike_raw': len(results['spike_raw']['segments']),
        'spike_prep': len(results['spike_prep']['segments']),
        'spikes_raw': results['spike_raw']['info'].get('n_spikes_detected', 0),
        'spikes_prep': results['spike_prep']['info'].get('n_spikes_detected', 0),
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
    print("COMPARING: Threshold vs Spike-Based Segment Extraction")
    print("=" * 70)
    print(f"Found {len(all_files)} files")
    print("\nMethods:")
    print("  1. Threshold (Raw): Fixed heading threshold on raw data")
    print("  2. Threshold (Prep): Fixed heading threshold on preprocessed data")
    print("  3. Spike (Raw): Peak detection on raw data")
    print("  4. Spike (Prep): Peak detection on preprocessed data")

    output_dir = project_root / "data/type2/lstm_pipeline_output/spike_vs_threshold"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, input_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing: {input_path.stem}...")

        try:
            results = process_subject(input_path, frame_rate=60)

            viz_path = output_dir / f"{input_path.stem}_comparison.png"
            stats = visualize_comparison(results, viz_path, input_path.stem, frame_rate=60)
            all_results.append(stats)

            print(f"    Threshold: Raw={stats['threshold_raw']}, Prep={stats['threshold_prep']}")
            print(f"    Spike:     Raw={stats['spike_raw']}, Prep={stats['spike_prep']}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(all_results)

    print(f"\n{'Subject':<12} {'Thr(Raw)':<10} {'Thr(Prep)':<10} {'Spk(Raw)':<10} {'Spk(Prep)':<10}")
    print("-" * 55)

    for _, row in df.iterrows():
        print(f"{row['subject']:<12} {row['threshold_raw']:<10} {row['threshold_prep']:<10} "
              f"{row['spike_raw']:<10} {row['spike_prep']:<10}")

    print("-" * 55)
    print(f"{'TOTAL':<12} {df['threshold_raw'].sum():<10} {df['threshold_prep'].sum():<10} "
          f"{df['spike_raw'].sum():<10} {df['spike_prep'].sum():<10}")

    # Find best method per subject
    methods = ['threshold_raw', 'threshold_prep', 'spike_raw', 'spike_prep']
    best_counts = {m: 0 for m in methods}

    for _, row in df.iterrows():
        best = max(methods, key=lambda m: row[m])
        best_counts[best] += 1

    print(f"\n" + "=" * 70)
    print("BEST METHOD COUNTS:")
    for method, count in best_counts.items():
        print(f"  {method:<25}: {count} subjects")
    print("=" * 70)

    # Save summary
    summary_path = output_dir / "spike_vs_threshold_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Create overview chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    subjects = df['subject'].tolist()
    x = np.arange(len(subjects))
    width = 0.2

    # Segment count comparison
    ax1 = axes[0]
    ax1.barh(x - 1.5*width, df['threshold_raw'], width, label='Threshold (Raw)', color='salmon')
    ax1.barh(x - 0.5*width, df['threshold_prep'], width, label='Threshold (Prep)', color='lightgreen')
    ax1.barh(x + 0.5*width, df['spike_raw'], width, label='Spike (Raw)', color='orange', alpha=0.7)
    ax1.barh(x + 1.5*width, df['spike_prep'], width, label='Spike (Prep)', color='lightblue')
    ax1.set_yticks(x)
    ax1.set_yticklabels(subjects, fontsize=8)
    ax1.set_xlabel('Number of Segments')
    ax1.set_title('Segment Count by Method', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Best method by subject
    ax2 = axes[1]
    method_colors = {'threshold_raw': 'salmon', 'threshold_prep': 'lightgreen',
                    'spike_raw': 'orange', 'spike_prep': 'lightblue'}
    method_labels = {'threshold_raw': 'Thr(Raw)', 'threshold_prep': 'Thr(Prep)',
                    'spike_raw': 'Spk(Raw)', 'spike_prep': 'Spk(Prep)'}

    best_methods = []
    best_colors = []
    for _, row in df.iterrows():
        best = max(methods, key=lambda m: row[m])
        best_methods.append(method_labels[best])
        best_colors.append(method_colors[best])

    ax2.barh(x, [1]*len(x), color=best_colors)
    ax2.set_yticks(x)
    ax2.set_yticklabels(subjects, fontsize=8)
    ax2.set_xlim([0, 1])
    ax2.set_xticks([])
    ax2.set_title('Best Method per Subject', fontsize=12, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for m, (c, l) in
                      zip(methods, [(method_colors[m], method_labels[m]) for m in methods])]
    ax2.legend(handles=legend_elements, loc='lower right')
    ax2.invert_yaxis()

    totals = [df[m].sum() for m in methods]
    fig.suptitle(
        f'Threshold vs Spike-Based Comparison\n'
        f'Totals: Thr(Raw)={totals[0]}, Thr(Prep)={totals[1]}, '
        f'Spk(Raw)={totals[2]}, Spk(Prep)={totals[3]}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    overview_path = output_dir / "spike_vs_threshold_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Overview saved to: {overview_path}")


if __name__ == "__main__":
    main()
