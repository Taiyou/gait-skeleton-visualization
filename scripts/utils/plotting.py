#!/usr/bin/env python3
"""
Common plotting utilities for gait analysis experiments.

This module provides reusable plotting functions to reduce code duplication
across experiment scripts. All functions follow consistent styling and
parameter conventions.

Usage:
    from scripts.utils.plotting import (
        plot_trajectory_comparison,
        plot_multi_method_comparison,
        PlotConfig,
    )

    # Simple trajectory comparison
    plot_trajectory_comparison(
        original_data, corrected_data,
        output_path="comparison.png",
        titles=["Original", "Corrected"],
    )

    # Multiple method comparison
    plot_multi_method_comparison(
        results=[
            {"name": "Method A", "data": data_a},
            {"name": "Method B", "data": data_b},
        ],
        original_data=original,
        output_path="methods.png",
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np

# Lazy import matplotlib to allow backend configuration
_plt = None
_mpl = None


def _get_plt():
    """Lazy import matplotlib.pyplot with Agg backend."""
    global _plt, _mpl
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plt = plt
        _mpl = matplotlib
    return _plt


@dataclass
class PlotConfig:
    """Configuration for plot styling and behavior.

    Attributes:
        figsize: Figure size as (width, height) tuple.
        dpi: Resolution for saved figures.
        colormap: Colormap for time-based coloring.
        scatter_size: Point size for scatter plots.
        scatter_alpha: Transparency for scatter points.
        line_width: Default line width.
        grid_alpha: Grid line transparency.
        title_fontsize: Font size for titles.
        label_fontsize: Font size for axis labels.
        marker_start: Marker style for start point.
        marker_end: Marker style for end point.
        marker_size: Size for start/end markers.
        pelvis_index: Index of pelvis segment in data.
        frame_rate: Frame rate in Hz.
        show_colorbar: Whether to show time colorbar.
        unit_scale: Scale factor for display (1000 for mm, 1 for m).
        unit_label: Unit label string ("mm" or "m").
    """
    figsize: Tuple[float, float] = (14, 6)
    dpi: int = 150
    colormap: str = "viridis"
    scatter_size: float = 0.5
    scatter_alpha: float = 0.5
    line_width: float = 0.5
    grid_alpha: float = 0.3
    title_fontsize: int = 11
    label_fontsize: int = 10
    marker_start: str = "go"
    marker_end: str = "rs"
    marker_size: int = 10
    pelvis_index: int = 0
    frame_rate: int = 60
    show_colorbar: bool = True
    unit_scale: float = 1.0  # 1 for meters, 1000 for mm
    unit_label: str = "m"


def calc_range(data: np.ndarray, axis: int, pelvis_index: int = 0) -> float:
    """Calculate range of pelvis position along specified axis.

    Args:
        data: Position data (n_frames, n_segments, 3)
        axis: Axis index (0=X, 1=Y, 2=Z)
        pelvis_index: Index of pelvis segment

    Returns:
        Range (max - min) in meters
    """
    return np.max(data[:, pelvis_index, axis]) - np.min(data[:, pelvis_index, axis])


def create_figure_grid(
    n_plots: int,
    n_cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (5, 4),
) -> Tuple[Any, np.ndarray]:
    """Create a figure with grid of subplots.

    Args:
        n_plots: Total number of plots needed
        n_cols: Number of columns in grid
        figsize_per_plot: Size per individual plot

    Returns:
        Tuple of (figure, flattened axes array)
    """
    plt = _get_plt()
    n_rows = (n_plots + n_cols - 1) // n_cols
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    return fig, axes.flatten()


def plot_trajectory(
    ax: Any,
    data: np.ndarray,
    title: str = "",
    config: Optional[PlotConfig] = None,
    color_by_time: bool = True,
    show_endpoints: bool = True,
    time_array: Optional[np.ndarray] = None,
) -> Any:
    """Plot a single trajectory on given axes.

    Args:
        ax: Matplotlib axes object
        data: Position data (n_frames, n_segments, 3)
        title: Title for the plot
        config: Plot configuration
        color_by_time: Whether to color points by time
        show_endpoints: Whether to show start/end markers
        time_array: Optional pre-computed time array (in minutes)

    Returns:
        Scatter object (for colorbar creation)
    """
    if config is None:
        config = PlotConfig()

    n_frames = len(data)
    x = data[:, config.pelvis_index, 0] * config.unit_scale
    y = data[:, config.pelvis_index, 1] * config.unit_scale

    if time_array is None:
        time_array = np.arange(n_frames) / config.frame_rate / 60  # Time in minutes

    if color_by_time:
        scatter = ax.scatter(
            x, y,
            c=time_array,
            cmap=config.colormap,
            s=config.scatter_size,
            alpha=config.scatter_alpha,
        )
    else:
        scatter = ax.scatter(
            x, y,
            s=config.scatter_size,
            alpha=config.scatter_alpha,
        )

    if show_endpoints:
        ax.plot(x[0], y[0], config.marker_start,
                markersize=config.marker_size, label='Start')
        ax.plot(x[-1], y[-1], config.marker_end,
                markersize=config.marker_size, label='End')
        ax.legend(loc='upper right', fontsize=8)

    # Calculate ranges for title
    x_range = calc_range(data, 0, config.pelvis_index)
    y_range = calc_range(data, 1, config.pelvis_index)

    if title:
        ax.set_title(
            f'{title}\nX: {x_range:.2f}m, Y: {y_range:.2f}m',
            fontsize=config.title_fontsize,
        )
    else:
        ax.set_title(
            f'X: {x_range:.2f}m, Y: {y_range:.2f}m',
            fontsize=config.title_fontsize,
        )

    ax.set_xlabel(f'X ({config.unit_label})', fontsize=config.label_fontsize)
    ax.set_ylabel(f'Y ({config.unit_label})', fontsize=config.label_fontsize)
    ax.axis('equal')
    ax.grid(True, alpha=config.grid_alpha, linestyle='--')

    return scatter


def plot_trajectory_comparison(
    original: np.ndarray,
    corrected: np.ndarray,
    output_path: Union[str, Path],
    titles: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
    suptitle: Optional[str] = None,
) -> None:
    """Create side-by-side trajectory comparison plot.

    This is the most common comparison pattern used across experiment scripts.

    Args:
        original: Original position data (n_frames, n_segments, 3)
        corrected: Corrected position data (n_frames, n_segments, 3)
        output_path: Path to save the figure
        titles: Optional list of titles [original_title, corrected_title]
        config: Plot configuration
        suptitle: Optional super title for the figure
    """
    plt = _get_plt()

    if config is None:
        config = PlotConfig()

    if titles is None:
        titles = ['Original', 'Corrected']

    n_frames = len(original)
    time = np.arange(n_frames) / config.frame_rate / 60

    fig, axes = plt.subplots(1, 2, figsize=config.figsize)

    # Plot original
    scatter1 = plot_trajectory(
        axes[0], original, titles[0],
        config=config, time_array=time,
    )
    if config.show_colorbar:
        plt.colorbar(scatter1, ax=axes[0], label='Time (min)')

    # Plot corrected
    scatter2 = plot_trajectory(
        axes[1], corrected, titles[1],
        config=config, time_array=time,
    )
    if config.show_colorbar:
        plt.colorbar(scatter2, ax=axes[1], label='Time (min)')

    # Add Y=0 reference line for corrected
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi)
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_multi_method_comparison(
    results: List[Dict[str, Any]],
    original_data: np.ndarray,
    output_path: Union[str, Path],
    config: Optional[PlotConfig] = None,
    include_original: bool = True,
    n_cols: int = 3,
) -> None:
    """Create grid comparison of multiple correction methods.

    Args:
        results: List of dicts with 'name' and 'data' keys.
                 Optional 'y_range_original' and 'y_range_corrected' for stats.
        original_data: Original uncorrected data for reference
        output_path: Path to save the figure
        config: Plot configuration
        include_original: Whether to include original as first plot
        n_cols: Number of columns in the grid

    Example:
        results = [
            {"name": "Method 1", "data": data1},
            {"name": "Method 2", "data": data2, "y_range_corrected": 1.2},
        ]
        plot_multi_method_comparison(results, original, "comparison.png")
    """
    plt = _get_plt()

    if config is None:
        config = PlotConfig()

    n_methods = len(results) + (1 if include_original else 0)
    fig, axes = create_figure_grid(n_methods, n_cols=n_cols)

    n_frames = len(original_data)
    time = np.arange(n_frames) / config.frame_rate / 60

    plot_idx = 0

    # Plot original first if requested
    if include_original:
        ax = axes[plot_idx]
        ax.plot(
            original_data[:, config.pelvis_index, 0],
            original_data[:, config.pelvis_index, 1],
            'b-', alpha=0.5, linewidth=config.line_width,
        )
        y_range = calc_range(original_data, 1, config.pelvis_index)
        ax.set_title(f'Original\nY range: {y_range:.2f}m', fontsize=config.title_fontsize)
        ax.set_xlabel('X (m)', fontsize=config.label_fontsize)
        ax.set_ylabel('Y (m)', fontsize=config.label_fontsize)
        ax.axis('equal')
        ax.grid(True, alpha=config.grid_alpha)
        plot_idx += 1

    # Plot each method
    colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
    for i, result in enumerate(results):
        ax = axes[plot_idx]
        data = result['data']
        name = result['name']

        ax.plot(
            data[:, config.pelvis_index, 0],
            data[:, config.pelvis_index, 1],
            '-', color=colors[i % len(colors)],
            alpha=0.5, linewidth=config.line_width,
        )

        # Calculate or use provided y_range
        y_range = result.get('y_range_corrected', calc_range(data, 1, config.pelvis_index))
        y_range_orig = result.get('y_range_original', calc_range(original_data, 1, config.pelvis_index))

        reduction = (1 - y_range / y_range_orig) * 100
        ax.set_title(
            f'{name}\nY range: {y_range:.2f}m ({reduction:.1f}% reduction)',
            fontsize=config.title_fontsize,
        )
        ax.set_xlabel('X (m)', fontsize=config.label_fontsize)
        ax.set_ylabel('Y (m)', fontsize=config.label_fontsize)
        ax.axis('equal')
        ax.grid(True, alpha=config.grid_alpha)
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def plot_time_series_comparison(
    datasets: List[Dict[str, Any]],
    output_path: Union[str, Path],
    config: Optional[PlotConfig] = None,
    ylabel: str = "Y position (m)",
    zoom_seconds: Optional[Tuple[float, float]] = None,
) -> None:
    """Create time series comparison plot.

    Args:
        datasets: List of dicts with 'name', 'data', and optional 'color' keys
        output_path: Path to save the figure
        config: Plot configuration
        ylabel: Label for Y axis
        zoom_seconds: Optional (start, end) tuple to zoom into specific time range
    """
    plt = _get_plt()

    if config is None:
        config = PlotConfig()

    n_rows = 2 if zoom_seconds else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    default_colors = ['gray', 'red', 'blue', 'green', 'orange', 'purple']

    # Full time series
    ax = axes[0]
    for i, ds in enumerate(datasets):
        data = ds['data']
        n_frames = len(data)
        time = np.arange(n_frames) / config.frame_rate

        y = data[:, config.pelvis_index, 1]
        color = ds.get('color', default_colors[i % len(default_colors)])
        alpha = ds.get('alpha', 0.7)

        ax.plot(time, y, color=color, alpha=alpha,
                linewidth=config.line_width, label=ds['name'])

    ax.set_xlabel('Time (s)', fontsize=config.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=config.label_fontsize)
    ax.set_title('Time Series Comparison', fontsize=config.title_fontsize)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=config.grid_alpha)

    # Zoomed view
    if zoom_seconds:
        ax = axes[1]
        start_frame = int(zoom_seconds[0] * config.frame_rate)
        end_frame = int(zoom_seconds[1] * config.frame_rate)

        for i, ds in enumerate(datasets):
            data = ds['data']
            time = np.arange(len(data)) / config.frame_rate
            y = data[:, config.pelvis_index, 1]
            color = ds.get('color', default_colors[i % len(default_colors)])

            ax.plot(
                time[start_frame:end_frame],
                y[start_frame:end_frame],
                color=color, alpha=0.7,
                linewidth=1.0, label=ds['name'],
            )

        ax.set_xlabel('Time (s)', fontsize=config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=config.label_fontsize)
        ax.set_title(
            f'Zoomed View ({zoom_seconds[0]:.1f}s - {zoom_seconds[1]:.1f}s)',
            fontsize=config.title_fontsize,
        )
        ax.legend(loc='upper right')
        ax.grid(True, alpha=config.grid_alpha)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi)
    plt.close()
    print(f"Saved time series plot to {output_path}")


def plot_segment_distribution(
    segments_list: List[Dict[str, Any]],
    output_path: Union[str, Path],
    config: Optional[PlotConfig] = None,
) -> None:
    """Plot segment duration distribution comparison.

    Args:
        segments_list: List of dicts with 'name', 'segments' (list of segment objects
                       with duration_sec attribute or dict with 'duration_sec')
        output_path: Path to save the figure
        config: Plot configuration
    """
    plt = _get_plt()

    if config is None:
        config = PlotConfig()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['red', 'green', 'blue', 'orange', 'purple']
    all_durations = []

    for seg_data in segments_list:
        segments = seg_data['segments']
        durations = []
        for s in segments:
            if hasattr(s, 'duration_sec'):
                durations.append(s.duration_sec)
            elif isinstance(s, dict) and 'duration_sec' in s:
                durations.append(s['duration_sec'])
        all_durations.extend(durations)
        seg_data['_durations'] = durations

    if not all_durations:
        ax.text(0.5, 0.5, 'No segments found', ha='center', va='center')
        plt.savefig(output_path, dpi=config.dpi)
        plt.close()
        return

    max_dur = max(all_durations)
    bins = np.linspace(0, max_dur, 15)

    for i, seg_data in enumerate(segments_list):
        durations = seg_data['_durations']
        ax.hist(
            durations, bins=bins, alpha=0.5,
            label=f'{seg_data["name"]} (n={len(durations)})',
            color=colors[i % len(colors)],
        )

    ax.set_xlabel('Segment Duration (s)', fontsize=config.label_fontsize)
    ax.set_ylabel('Count', fontsize=config.label_fontsize)
    ax.set_title('Segment Duration Distribution', fontsize=config.title_fontsize)
    ax.legend()
    ax.grid(True, alpha=config.grid_alpha)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi)
    plt.close()
    print(f"Saved segment distribution plot to {output_path}")


def plot_overlay_trajectories(
    datasets: List[Dict[str, Any]],
    output_path: Union[str, Path],
    config: Optional[PlotConfig] = None,
    title: str = "Method Comparison Overlay",
) -> None:
    """Create overlay plot of multiple trajectories.

    Args:
        datasets: List of dicts with 'name', 'data', optional 'color' and 'alpha'
        output_path: Path to save the figure
        config: Plot configuration
        title: Plot title
    """
    plt = _get_plt()

    if config is None:
        config = PlotConfig()

    fig, ax = plt.subplots(figsize=(12, 8))

    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

    for i, ds in enumerate(datasets):
        data = ds['data']
        name = ds['name']
        color = ds.get('color', default_colors[i % len(default_colors)])
        alpha = ds.get('alpha', 0.6)

        y_range = calc_range(data, 1, config.pelvis_index)

        ax.plot(
            data[:, config.pelvis_index, 0],
            data[:, config.pelvis_index, 1],
            '-', color=color, alpha=alpha,
            linewidth=config.line_width,
            label=f'{name} (Y:{y_range:.2f}m)',
        )

    ax.set_title(title, fontsize=config.title_fontsize)
    ax.set_xlabel('X (m)', fontsize=config.label_fontsize)
    ax.set_ylabel('Y (m)', fontsize=config.label_fontsize)
    ax.axis('equal')
    ax.grid(True, alpha=config.grid_alpha)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi)
    plt.close()
    print(f"Saved overlay plot to {output_path}")


def print_summary_table(
    results: List[Dict[str, Any]],
    original_y_range: float,
    baseline_y_range: Optional[float] = None,
) -> None:
    """Print formatted summary table of correction results.

    Args:
        results: List of dicts with 'name', 'y_range_corrected', and optionally
                 'y_range_original'
        original_y_range: Y range of original uncorrected data
        baseline_y_range: Optional Y range of baseline correction
    """
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Method':<35} {'Y Range (m)':<15} {'Reduction':<15}")
    print("-" * 80)
    print(f"{'Original':<35} {original_y_range:<15.4f} {'-':<15}")

    if baseline_y_range is not None:
        reduction = (1 - baseline_y_range / original_y_range) * 100
        print(f"{'Baseline':<35} {baseline_y_range:<15.4f} {reduction:.1f}%")

    for result in results:
        name = result['name']
        y_range = result.get('y_range_corrected', result.get('y_range', 0))
        orig = result.get('y_range_original', original_y_range)
        reduction = (1 - y_range / orig) * 100
        print(f"{name:<35} {y_range:<15.4f} {reduction:.1f}%")

    print("=" * 80)

    # Find best method
    if results:
        best = min(results, key=lambda r: r.get('y_range_corrected', r.get('y_range', float('inf'))))
        best_y = best.get('y_range_corrected', best.get('y_range', 0))
        best_reduction = (1 - best_y / original_y_range) * 100
        print(f"\nBest method: {best['name']}")
        print(f"  Y range: {best_y:.4f}m")
        print(f"  Reduction: {best_reduction:.1f}%")
