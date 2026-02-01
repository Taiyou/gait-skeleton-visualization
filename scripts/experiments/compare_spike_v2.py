#!/usr/bin/env python3
"""
Compare Segment Extraction Methods with Spike v2:
1. Threshold (Preprocessed) - Current best
2. Spike v1 (Preprocessed) - Original spike detection
3. Spike v2 (Preprocessed) - Improved with spike merging
4. Spike v2 (Raw) - To see if v2 works better on raw data

Spike v2 improvements:
- Nearby spikes are merged into single turn regions
- Adaptive prominence based on signal statistics
- Better handling of consecutive turns
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
from scripts.gait_analysis.spike_based_segment_extraction_v2 import (
    SpikeExtractionParamsV2,
    extract_segments_spike_v2
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
    # Method 1: Threshold-based on PREPROCESSED data (current best)
    # =========================================================================
    params_threshold = SegmentExtractionParams(
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
    segments, info = extract_segments_improved(preprocessed_data, velocity_preprocessed, params_threshold)
    results['threshold_prep'] = {
        'data': preprocessed_data,
        'segments': segments,
        'info': info,
        'method': 'Threshold (Prep)',
        'color': 'green',
    }

    # =========================================================================
    # Method 2: Spike v1 on PREPROCESSED data
    # =========================================================================
    params_spike_v1 = SpikeExtractionParams(
        velocity_threshold=0.4,
        spike_prominence=0.05,
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
    segments, info = extract_segments_spike_based(preprocessed_data, velocity_preprocessed, params_spike_v1)
    results['spike_v1_prep'] = {
        'data': preprocessed_data,
        'segments': segments,
        'info': info,
        'method': 'Spike v1 (Prep)',
        'color': 'blue',
    }

    # =========================================================================
    # Method 3: Spike v2 on PREPROCESSED data (NEW - with spike merging)
    # =========================================================================
    # Key insight: Only detect STRONG spikes (real turns), not small fluctuations
    params_spike_v2_prep = SpikeExtractionParamsV2(
        velocity_threshold=0.4,
        spike_prominence_percentile=95,   # Only top 5% of peaks (much stricter)
        spike_min_prominence=0.08,        # Higher minimum prominence
        spike_min_height_percentile=95,   # Only very high peaks
        spike_width_seconds=0.3,
        merge_distance_seconds=1.0,       # Shorter merge window
        exclude_padding_seconds=0.5,
        trim_start_seconds=0.5,
        trim_end_seconds=0.3,
        use_overlapping_windows=True,
        window_seconds=5.0,
        window_overlap=0.5,
        min_segment_meters=5.0,
        frame_rate=frame_rate
    )
    segments, info = extract_segments_spike_v2(preprocessed_data, velocity_preprocessed, params_spike_v2_prep)
    results['spike_v2_prep'] = {
        'data': preprocessed_data,
        'segments': segments,
        'info': info,
        'method': 'Spike v2 (Prep)',
        'color': 'purple',
    }

    # =========================================================================
    # Method 4: Spike v2 on RAW data
    # =========================================================================
    # For raw data, need even stricter spike detection (more noise)
    params_spike_v2_raw = SpikeExtractionParamsV2(
        velocity_threshold=0.4,
        spike_prominence_percentile=97,   # Even stricter for noisy data
        spike_min_prominence=0.10,        # Higher minimum
        spike_min_height_percentile=97,   # Only very strong peaks
        spike_width_seconds=0.3,
        merge_distance_seconds=1.0,
        exclude_padding_seconds=0.5,
        trim_start_seconds=0.7,
        trim_end_seconds=0.5,
        use_overlapping_windows=True,
        window_seconds=5.0,
        window_overlap=0.5,
        min_segment_meters=5.0,
        frame_rate=frame_rate
    )
    segments, info = extract_segments_spike_v2(raw_data, velocity_raw, params_spike_v2_raw)
    results['spike_v2_raw'] = {
        'data': raw_data,
        'segments': segments,
        'info': info,
        'method': 'Spike v2 (Raw)',
        'color': 'orange',
    }

    return results


def visualize_comparison(results: dict, output_path: Path, subject_name: str, frame_rate: int = 60):
    """Visualize all methods comparison with focus on spike merging."""
    fig = plt.figure(figsize=(24, 18))

    methods = ['threshold_prep', 'spike_v1_prep', 'spike_v2_prep', 'spike_v2_raw']
    method_names = ['Threshold (Prep)', 'Spike v1 (Prep)', 'Spike v2 (Prep)', 'Spike v2 (Raw)']
    colors = ['green', 'blue', 'purple', 'orange']

    n_frames = len(results['threshold_prep']['data'])
    time = np.arange(n_frames) / frame_rate

    # Row 1: Trajectories with segments
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = fig.add_subplot(5, 4, i + 1)
        r = results[method]
        data = r['data']

        ax.scatter(data[:, 0, 0], data[:, 0, 1], c=time/60, cmap='gray', s=0.3, alpha=0.3)
        for seg in r['segments']:
            ax.plot(seg.data[:, 0, 0], seg.data[:, 0, 1], color=color, linewidth=1.5, alpha=0.7)

        n_segs = len(r['segments'])
        ax.set_title(f'{name}\n{n_segs} segments', fontsize=11, color=color, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    # Row 2: Heading change rate with spikes marked
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = fig.add_subplot(5, 4, 4 + i + 1)
        r = results[method]
        heading_rate = r['info']['heading_rate']

        ax.plot(time, np.degrees(heading_rate), color=color, linewidth=0.5, alpha=0.7)

        # Mark individual spikes
        if 'spikes' in r['info'] and len(r['info']['spikes']) > 0:
            spikes = r['info']['spikes']
            ax.scatter(spikes / frame_rate, np.degrees(heading_rate[spikes]),
                      color='red', s=20, zorder=5, marker='v',
                      label=f'{len(spikes)} spikes')

        # Highlight segments
        for seg in r['segments']:
            ax.axvspan(seg.start_frame/frame_rate, seg.end_frame/frame_rate,
                      alpha=0.15, color=color)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading rate (deg/frame)')
        ax.set_title(f'{name}: Heading Change', fontsize=10)
        ax.set_ylim([0, min(30, np.percentile(np.degrees(heading_rate), 99))])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 3: Show merged regions for spike v2 methods
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = fig.add_subplot(5, 4, 8 + i + 1)
        r = results[method]

        # Show exclusion mask
        if 'exclude_mask' in r['info']:
            exclude = r['info']['exclude_mask'].astype(float)
            ax.fill_between(time, 0, exclude, alpha=0.4, color='red', label='Excluded')
            ax.fill_between(time, 0, (1-exclude) * r['info'].get('straight_mask', ~r['info']['exclude_mask']).astype(float),
                           alpha=0.4, color='green', label='Straight walking')

        # Mark merged regions for v2 methods
        if 'merged_regions' in r['info'] and len(r['info']['merged_regions']) > 0:
            for start, end, peaks in r['info']['merged_regions']:
                ax.axvspan(start/frame_rate, end/frame_rate, alpha=0.3, color='darkred',
                          linewidth=2)
                # Mark individual peaks within merged region
                for peak in peaks:
                    ax.axvline(peak/frame_rate, color='black', linestyle='--', alpha=0.5, linewidth=0.5)

            n_merged = len(r['info']['merged_regions'])
            ax.set_title(f'{name}: {n_merged} merged turn regions', fontsize=10)
        else:
            ax.set_title(f'{name}: Exclusion zones', fontsize=10)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Region')
        ax.set_ylim([0, 1.1])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 4: Zoomed view of a section with turns (first 60 seconds)
    zoom_end = min(60 * frame_rate, n_frames)
    time_zoom = np.arange(zoom_end) / frame_rate

    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = fig.add_subplot(5, 4, 12 + i + 1)
        r = results[method]
        heading_rate = r['info']['heading_rate'][:zoom_end]

        ax.plot(time_zoom, np.degrees(heading_rate), color=color, linewidth=0.8, alpha=0.8)

        # Mark spikes
        if 'spikes' in r['info']:
            spikes = r['info']['spikes']
            spikes_in_range = spikes[spikes < zoom_end]
            if len(spikes_in_range) > 0:
                ax.scatter(spikes_in_range / frame_rate,
                          np.degrees(r['info']['heading_rate'][spikes_in_range]),
                          color='red', s=40, zorder=5, marker='v')

        # Mark merged regions
        if 'merged_regions' in r['info']:
            for start, end, peaks in r['info']['merged_regions']:
                if start < zoom_end:
                    ax.axvspan(start/frame_rate, min(end, zoom_end)/frame_rate,
                              alpha=0.3, color='yellow', edgecolor='red', linewidth=2)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading rate (deg/frame)')
        ax.set_title(f'{name}: First 60s (zoomed)', fontsize=10)
        ax.set_ylim([0, min(20, np.percentile(np.degrees(heading_rate), 99))])
        ax.grid(True, alpha=0.3)

    # Row 5: Summary bar chart and statistics
    ax_bar = fig.add_subplot(5, 2, 9)
    segment_counts = [len(results[m]['segments']) for m in methods]
    bars = ax_bar.bar(method_names, segment_counts, color=colors, alpha=0.7)
    ax_bar.set_ylabel('Number of Segments')
    ax_bar.set_title('Segment Count Comparison', fontsize=12, fontweight='bold')

    for bar, count in zip(bars, segment_counts):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_bar.grid(True, alpha=0.3, axis='y')

    # Statistics
    ax_text = fig.add_subplot(5, 2, 10)
    ax_text.axis('off')

    best_idx = np.argmax(segment_counts)
    best_method = method_names[best_idx]

    # Get spike/region counts
    def get_counts(method_key):
        info = results[method_key]['info']
        spikes = info.get('n_spikes_detected', info.get('n_spikes', 0))
        merged = info.get('n_merged_regions', '-')
        return spikes, merged

    thresh_spikes, thresh_merged = '-', '-'
    v1_spikes, v1_merged = get_counts('spike_v1_prep')
    v2_prep_spikes, v2_prep_merged = get_counts('spike_v2_prep')
    v2_raw_spikes, v2_raw_merged = get_counts('spike_v2_raw')

    stats_text = f"""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    COMPARISON SUMMARY                              ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Method                  Segments    Spikes    Merged Regions      ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Threshold (Prep)        {len(results['threshold_prep']['segments']):<10}  {thresh_spikes:<10} {thresh_merged:<10}          ║
    ║  Spike v1 (Prep)         {len(results['spike_v1_prep']['segments']):<10}  {v1_spikes:<10} {v1_merged:<10}          ║
    ║  Spike v2 (Prep)         {len(results['spike_v2_prep']['segments']):<10}  {v2_prep_spikes:<10} {v2_prep_merged:<10}          ║
    ║  Spike v2 (Raw)          {len(results['spike_v2_raw']['segments']):<10}  {v2_raw_spikes:<10} {v2_raw_merged:<10}          ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  BEST METHOD: {best_method:<52} ║
    ╚════════════════════════════════════════════════════════════════════╝

    Spike v2 improvements:
    - Nearby spikes merged into single turn regions
    - Adaptive prominence based on signal percentiles
    - Better handling of consecutive direction changes
    """

    ax_text.text(0.02, 0.5, stats_text, transform=ax_text.transAxes, fontsize=10,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'{subject_name}: Spike v2 (with Merging) Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'subject': subject_name,
        'threshold_prep': len(results['threshold_prep']['segments']),
        'spike_v1_prep': len(results['spike_v1_prep']['segments']),
        'spike_v2_prep': len(results['spike_v2_prep']['segments']),
        'spike_v2_raw': len(results['spike_v2_raw']['segments']),
        'v2_prep_spikes': v2_prep_spikes,
        'v2_prep_merged': v2_prep_merged,
        'v2_raw_spikes': v2_raw_spikes,
        'v2_raw_merged': v2_raw_merged,
    }


def main():
    project_root = Path(__file__).parent.parent

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
    print("COMPARING: Spike v2 (with Merging) vs Other Methods")
    print("=" * 70)
    print(f"Found {len(all_files)} files")
    print("\nMethods:")
    print("  1. Threshold (Prep): Fixed threshold on preprocessed data")
    print("  2. Spike v1 (Prep): Original spike detection")
    print("  3. Spike v2 (Prep): Improved with spike merging")
    print("  4. Spike v2 (Raw): v2 on raw data")

    output_dir = project_root / "data/type2/lstm_pipeline_output/spike_v2_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, input_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing: {input_path.stem}...")

        try:
            results = process_subject(input_path, frame_rate=60)

            viz_path = output_dir / f"{input_path.stem}_spike_v2.png"
            stats = visualize_comparison(results, viz_path, input_path.stem, frame_rate=60)
            all_results.append(stats)

            print(f"    Threshold: {stats['threshold_prep']}, Spike v1: {stats['spike_v1_prep']}, "
                  f"Spike v2(Prep): {stats['spike_v2_prep']}, Spike v2(Raw): {stats['spike_v2_raw']}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(all_results)

    print(f"\n{'Subject':<12} {'Thr(P)':<8} {'Spk1(P)':<8} {'Spk2(P)':<8} {'Spk2(R)':<8}")
    print("-" * 50)

    for _, row in df.iterrows():
        print(f"{row['subject']:<12} {row['threshold_prep']:<8} {row['spike_v1_prep']:<8} "
              f"{row['spike_v2_prep']:<8} {row['spike_v2_raw']:<8}")

    print("-" * 50)
    print(f"{'TOTAL':<12} {df['threshold_prep'].sum():<8} {df['spike_v1_prep'].sum():<8} "
          f"{df['spike_v2_prep'].sum():<8} {df['spike_v2_raw'].sum():<8}")

    # Best method per subject
    methods = ['threshold_prep', 'spike_v1_prep', 'spike_v2_prep', 'spike_v2_raw']
    best_counts = {m: 0 for m in methods}

    for _, row in df.iterrows():
        best = max(methods, key=lambda m: row[m])
        best_counts[best] += 1

    print(f"\n" + "=" * 70)
    print("BEST METHOD COUNTS:")
    for method, count in best_counts.items():
        print(f"  {method:<20}: {count} subjects")
    print("=" * 70)

    # Save summary
    summary_path = output_dir / "spike_v2_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Create overview
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    subjects = df['subject'].tolist()
    x = np.arange(len(subjects))
    width = 0.2

    ax1 = axes[0]
    ax1.barh(x - 1.5*width, df['threshold_prep'], width, label='Threshold (Prep)', color='lightgreen')
    ax1.barh(x - 0.5*width, df['spike_v1_prep'], width, label='Spike v1 (Prep)', color='lightblue')
    ax1.barh(x + 0.5*width, df['spike_v2_prep'], width, label='Spike v2 (Prep)', color='plum')
    ax1.barh(x + 1.5*width, df['spike_v2_raw'], width, label='Spike v2 (Raw)', color='orange', alpha=0.7)
    ax1.set_yticks(x)
    ax1.set_yticklabels(subjects, fontsize=8)
    ax1.set_xlabel('Number of Segments')
    ax1.set_title('Segment Count by Method', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Best method
    ax2 = axes[1]
    method_colors = {'threshold_prep': 'lightgreen', 'spike_v1_prep': 'lightblue',
                    'spike_v2_prep': 'plum', 'spike_v2_raw': 'orange'}

    best_colors = []
    for _, row in df.iterrows():
        best = max(methods, key=lambda m: row[m])
        best_colors.append(method_colors[best])

    ax2.barh(x, [1]*len(x), color=best_colors)
    ax2.set_yticks(x)
    ax2.set_yticklabels(subjects, fontsize=8)
    ax2.set_xlim([0, 1])
    ax2.set_xticks([])
    ax2.set_title('Best Method per Subject', fontsize=12, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=method_colors[m], label=m) for m in methods]
    ax2.legend(handles=legend_elements, loc='lower right')
    ax2.invert_yaxis()

    totals = [df[m].sum() for m in methods]
    fig.suptitle(
        f'Spike v2 Comparison\n'
        f'Totals: Thr={totals[0]}, v1={totals[1]}, v2(P)={totals[2]}, v2(R)={totals[3]}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    overview_path = output_dir / "spike_v2_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Overview saved to: {overview_path}")


if __name__ == "__main__":
    main()
