#!/usr/bin/env python3
"""
Generate figures for Wiki documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11


def create_pipeline_flowchart():
    """Create a visual flowchart of the pipeline."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    input_color = '#E3F2FD'  # Light blue
    process_color = '#E8F5E9'  # Light green
    output_color = '#FFF3E0'  # Light orange
    arrow_color = '#455A64'

    def draw_box(x, y, width, height, text, color, fontsize=10):
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color, edgecolor='#333333', linewidth=2
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

    # Title
    ax.text(6, 9.5, 'Segment Extraction Pipeline', ha='center', va='center',
            fontsize=16, fontweight='bold')

    # Input
    draw_box(6, 8.5, 4, 0.7, 'Raw IMU Data (Excel)', input_color)
    draw_arrow(6, 8.1, 6, 7.6)

    # Step 1: Preprocessing
    draw_box(6, 7.2, 5, 1.2, 'Step 1: Preprocessing\n• Global PCA Rotation\n• Y-axis Drift Removal\n• Body-centered Transform', process_color)
    draw_arrow(6, 6.5, 6, 6.0)

    # Step 2: Detection
    draw_box(6, 5.5, 5, 1.0, 'Step 2: Straight Region Detection\n• Velocity > 0.4 m/s\n• Heading change < 0.1 rad/frame', process_color)
    draw_arrow(6, 4.9, 6, 4.4)

    # Step 3: Extraction
    draw_box(6, 3.9, 5, 1.2, 'Step 3: Segment Extraction\n• Find continuous regions\n• Trim start/end\n• Create overlapping windows', process_color)
    draw_arrow(6, 3.2, 6, 2.7)

    # Output
    draw_box(6, 2.3, 4, 0.7, 'List[ExtractedSegment]', output_color)

    # Side annotations
    ax.text(10, 7.2, 'Reduces drift\nby 80%+', ha='left', va='center',
            fontsize=9, style='italic', color='#666666')
    ax.text(10, 5.5, 'Identifies\nstraight walking', ha='left', va='center',
            fontsize=9, style='italic', color='#666666')
    ax.text(10, 3.9, '5s windows\n50% overlap', ha='left', va='center',
            fontsize=9, style='italic', color='#666666')

    plt.tight_layout()
    return fig


def create_preprocessing_comparison():
    """Create before/after preprocessing comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Simulate raw data with drift
    np.random.seed(42)
    n_frames = 600
    time = np.arange(n_frames) / 60

    # Raw data: walking with Y drift
    x_raw = 1.2 * time + 0.02 * np.sin(2 * np.pi * 1.8 * time)
    y_raw = 0.5 * time + 0.02 * np.sin(2 * np.pi * 1.8 * time + np.pi/2)  # Drift + gait

    # Corrected data: Y drift removed
    x_corrected = 1.2 * time + 0.02 * np.sin(2 * np.pi * 1.8 * time)
    y_corrected = 0.02 * np.sin(2 * np.pi * 1.8 * time + np.pi/2)  # Only gait oscillation

    # Raw trajectory
    ax1 = axes[0]
    scatter = ax1.scatter(x_raw, y_raw, c=time, cmap='viridis', s=2, alpha=0.7)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Before Preprocessing\n(Y-axis drift present)', fontsize=13, fontweight='bold', color='red')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Add drift annotation
    ax1.annotate('Y-axis drift', xy=(8, 3), xytext=(5, 4),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')

    # Corrected trajectory
    ax2 = axes[1]
    ax2.scatter(x_corrected, y_corrected, c=time, cmap='viridis', s=2, alpha=0.7)
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title('After Preprocessing\n(Drift removed, gait preserved)', fontsize=13, fontweight='bold', color='green')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Add Y range annotation
    y_range_raw = y_raw.max() - y_raw.min()
    y_range_corrected = y_corrected.max() - y_corrected.min()

    ax1.text(0.05, 0.95, f'Y range: {y_range_raw:.2f}m', transform=ax1.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.05, 0.95, f'Y range: {y_range_corrected:.2f}m', transform=ax2.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.colorbar(scatter, ax=axes, label='Time (s)', shrink=0.8)
    plt.tight_layout()
    return fig


def create_segment_extraction_demo():
    """Create visualization of segment extraction process."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Simulate data
    np.random.seed(42)
    n_frames = 1200  # 20 seconds
    time = np.arange(n_frames) / 60

    # Create velocity with walking and stopping periods
    velocity = np.zeros(n_frames)

    # Walking segments
    walking_periods = [(60, 300), (360, 540), (600, 900), (960, 1140)]
    for start, end in walking_periods:
        velocity[start:end] = 1.2 + 0.1 * np.random.randn(end - start)

    # Turns (direction changes)
    turn_frames = [180, 450, 750, 1050]

    # Create heading change rate
    heading_rate = np.zeros(n_frames)
    for turn in turn_frames:
        # Add spike at turn
        heading_rate[max(0, turn-10):min(n_frames, turn+10)] = 0.3

    # Add small noise
    heading_rate += 0.02 * np.abs(np.random.randn(n_frames))

    # Detect straight regions
    is_walking = velocity > 0.4
    is_straight = heading_rate < 0.1
    straight_mask = is_walking & is_straight

    # Plot 1: Velocity
    ax1 = axes[0]
    ax1.fill_between(time, 0, velocity, alpha=0.3, color='blue')
    ax1.plot(time, velocity, 'b-', linewidth=1, label='Velocity')
    ax1.axhline(0.4, color='red', linestyle='--', linewidth=2, label='Threshold (0.4 m/s)')
    ax1.set_ylabel('Velocity (m/s)', fontsize=12)
    ax1.set_title('Step 1: Velocity Thresholding', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 20)
    ax1.grid(True, alpha=0.3)

    # Highlight walking regions
    for start, end in walking_periods:
        ax1.axvspan(start/60, end/60, alpha=0.15, color='green')

    # Plot 2: Heading change rate
    ax2 = axes[1]
    ax2.fill_between(time, 0, np.degrees(heading_rate), alpha=0.3, color='orange')
    ax2.plot(time, np.degrees(heading_rate), 'orange', linewidth=1, label='Heading change rate')
    ax2.axhline(np.degrees(0.1), color='red', linestyle='--', linewidth=2, label='Threshold (5.7°/frame)')
    ax2.set_ylabel('Heading Rate (°/frame)', fontsize=12)
    ax2.set_title('Step 2: Heading Change Detection (Turns)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 25)
    ax2.grid(True, alpha=0.3)

    # Mark turns
    for turn in turn_frames:
        ax2.axvline(turn/60, color='red', linestyle=':', alpha=0.7, linewidth=2)
        ax2.text(turn/60, 22, 'Turn', ha='center', fontsize=9, color='red')

    # Plot 3: Final segments
    ax3 = axes[2]
    ax3.fill_between(time, 0, straight_mask.astype(float), alpha=0.3, color='green',
                    label='Straight walking')
    ax3.fill_between(time, 0, (~straight_mask).astype(float), alpha=0.2, color='red',
                    label='Excluded')
    ax3.set_ylabel('Segment', fontsize=12)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_title('Step 3: Extracted Straight-Line Segments', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_xlim(0, 20)
    ax3.set_ylim(0, 1.2)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Excluded', 'Walking'])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_experiment_results():
    """Create bar chart of experiment results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [
        'Threshold +\nPreprocessing',
        'Threshold +\nRaw',
        'Spike v1 +\nPreprocessing',
        'Spike v2 +\nRaw',
        'Spike v2 +\nPreprocessing'
    ]
    segments = [1081, 820, 1029, 309, 149]
    colors = ['#4CAF50', '#81C784', '#64B5F6', '#FFB74D', '#FF8A65']

    bars = ax.bar(methods, segments, color=colors, edgecolor='#333333', linewidth=1.5)

    # Add value labels
    for bar, seg in zip(bars, segments):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
               f'{seg}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Highlight best method
    bars[0].set_edgecolor('#2E7D32')
    bars[0].set_linewidth(3)
    ax.text(0, 1100, '★ BEST', ha='center', fontsize=11, fontweight='bold', color='#2E7D32')

    ax.set_ylabel('Total Segments (30 subjects)', fontsize=12)
    ax.set_title('Comparison of Segment Extraction Methods', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1200)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def create_overlapping_windows_diagram():
    """Create diagram explaining overlapping windows."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(10, 4.5, 'Overlapping Windows (5s window, 50% overlap)', ha='center',
            fontsize=14, fontweight='bold')

    # Original segment (15 seconds)
    ax.add_patch(plt.Rectangle((1, 3), 15, 0.8, facecolor='#E3F2FD',
                               edgecolor='#1976D2', linewidth=2))
    ax.text(8.5, 3.4, 'Original Segment (15 seconds)', ha='center', fontsize=11)

    # Windows
    colors = ['#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A', '#4CAF50']
    window_width = 5
    overlap = 2.5

    for i in range(5):
        x_start = 1 + i * overlap
        y_pos = 1.5 - i * 0.25

        # Window box
        ax.add_patch(plt.Rectangle((x_start, y_pos), window_width, 0.5,
                                   facecolor=colors[i], edgecolor='#333333',
                                   linewidth=1.5, alpha=0.8))
        ax.text(x_start + window_width/2, y_pos + 0.25, f'Window {i+1}',
               ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow showing overlap
    ax.annotate('', xy=(3.5, 1.1), xytext=(6, 1.1),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(4.75, 0.9, '2.5s overlap', ha='center', fontsize=10, color='red')

    # Result text
    ax.text(10, 0.3, '→ 5 training samples from 1 segment', ha='center',
            fontsize=12, fontweight='bold', color='#2E7D32')

    plt.tight_layout()
    return fig


def main():
    output_dir = Path(__file__).parent.parent / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Wiki figures...")

    # Generate and save figures
    figures = [
        ("pipeline_flowchart.png", create_pipeline_flowchart()),
        ("preprocessing_comparison.png", create_preprocessing_comparison()),
        ("segment_extraction_demo.png", create_segment_extraction_demo()),
        ("experiment_results.png", create_experiment_results()),
        ("overlapping_windows.png", create_overlapping_windows_diagram()),
    ]

    for filename, fig in figures:
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  Saved: {output_path}")

    print("\nDone! Figures saved to docs/images/")


if __name__ == "__main__":
    main()
