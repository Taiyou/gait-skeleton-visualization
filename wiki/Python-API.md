# Python API

This page documents the Python API for programmatic use of the gait skeleton visualization tool.

## Installation

```bash
# Install in development mode (recommended)
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Basic Usage

```python
from src.data_loader import DataLoader
from src.skeleton_model import SkeletonModel
from src.visualizer_3d import Visualizer3D
from src.video_exporter import VideoExporter

# Load marker data
loader = DataLoader()
loader.load_csv("data/sample.csv", frame_rate=100.0)

# Load skeleton configuration
skeleton = SkeletonModel.from_yaml("config/marker_sets.yaml", "simple")
skeleton = skeleton.filter_markers(loader.markers)

# Create visualizer
visualizer = Visualizer3D(skeleton)
visualizer.set_bounds(loader.get_data_bounds())

# Get all frame positions
all_positions = loader.get_all_positions()

# Generate animation frames
frames = visualizer.create_animation_frames(all_positions, frame_rate=100.0)

# Export to video
exporter = VideoExporter(output_fps=30)
exporter.export(frames, "output/video.mp4")
```

---

## Core Classes

### DataLoader

Loads and processes CSV marker data.

```python
loader = DataLoader()
loader.load_csv("data.csv", frame_rate=100.0)

# Available properties
loader.markers          # List of marker names
loader.num_frames       # Total number of frames
loader.frame_rate       # Data frame rate

# Methods
positions = loader.get_positions(frame_idx=0)  # Single frame
all_positions = loader.get_all_positions()      # All frames
bounds = loader.get_data_bounds()               # Data bounds for visualization
```

### SkeletonModel

Defines skeleton structure and appearance.

```python
# Load from YAML
skeleton = SkeletonModel.from_yaml("config/marker_sets.yaml", "simple")

# Filter to available markers
skeleton = skeleton.filter_markers(available_markers)

# Properties
skeleton.markers      # List of marker names
skeleton.connections  # List of [marker1, marker2] connections

# Methods
color = skeleton.get_marker_color("HEAD")
color = skeleton.get_connection_color("HEAD", "NECK")
valid = skeleton.get_valid_connections(positions_dict)
```

### Visualizer3D

Creates 3D visualizations.

```python
visualizer = Visualizer3D(
    skeleton,
    figsize=(10, 8),
    bg_color="#FFFFFF",
    marker_size=50,
    line_width=2.0
)

visualizer.set_bounds(bounds)

# Render single frame
fig = visualizer.render_frame(positions, frame_number=0)

# Create animation frames
frames = visualizer.create_animation_frames(
    all_positions,
    frame_rate=100.0,
    show_labels=False
)
```

### Visualizer2D

Creates 2D projections.

```python
from src.visualizer_2d import Visualizer2D, MultiViewVisualizer

# Single view
visualizer = Visualizer2D(
    skeleton,
    view="sagittal",  # "sagittal", "frontal", "transverse"
    figsize=(10, 8)
)

# Multi-view
multi_vis = MultiViewVisualizer(
    skeleton,
    views=["sagittal", "frontal", "transverse"],
    figsize=(15, 5)
)
```

### VideoExporter

Exports frames to video.

```python
exporter = VideoExporter(output_fps=30)
exporter.export(frames, "output/video.mp4")
```

---

## Gait Correction Modules

### Loading Xsens Data

```python
from scripts.gait_correction import load_xsens_data, XsensDataLoader

# Load Excel data from Xsens
loader = load_xsens_data("data/NCC01-001.xlsx", frame_rate=60)

# Access data
positions = loader.positions      # Shape: (n_frames, n_segments, 3)
segment_names = loader.segment_names
```

### Drift Correction

```python
from scripts.gait_correction import (
    apply_smooth_pca_correction,
    apply_full_drift_correction,
    SmoothPCAParams,
)

# Step 1: Apply Smooth PCA heading correction
params = SmoothPCAParams(
    window_seconds=30.0,
    sample_interval_seconds=5.0,
    frame_rate=60,
)

corrected, angles, corrections = apply_smooth_pca_correction(
    positions,
    params=params,
    pelvis_index=0,
)

# Step 2: Apply Y-axis drift removal
corrected = apply_full_drift_correction(
    corrected,
    drift_window_seconds=30.0,
    frame_rate=60,
)
```

### Advanced Correction Methods

```python
from scripts.gait_correction.advanced_correction import apply_all_methods

# Apply all correction methods for comparison
results = apply_all_methods(
    data,
    frame_rate=60,
    pelvis_index=0,
    original_data=original_data,
)

# Each result contains: name, data, y_range_original, y_range_corrected
for result in results:
    print(f"{result.name}: {result.y_range_corrected:.2f}m")
```

### Turnaround Detection

```python
from scripts.gait_correction.turnaround import detect_turnarounds_adaptive

result = detect_turnarounds_adaptive(
    pelvis_x,  # X position array
    frame_rate=60,
)

# Access detected segments
for start, end in result.segments:
    print(f"Segment: frames {start} to {end}")
```

---

## Shared Utilities (NEW)

The `scripts.utils` module provides shared utilities for plotting and configuration.
See [Refactoring Guide](Refactoring-Guide) for detailed documentation.

### Plotting Utilities

```python
from scripts.utils.plotting import (
    plot_trajectory_comparison,
    plot_multi_method_comparison,
    plot_time_series_comparison,
    plot_segment_distribution,
    plot_overlay_trajectories,
    print_summary_table,
    PlotConfig,
    calc_range,
)
```

#### plot_trajectory_comparison()

Create side-by-side comparison of original vs corrected trajectory.

```python
from scripts.utils.plotting import plot_trajectory_comparison, PlotConfig

# Basic usage
plot_trajectory_comparison(
    original_data,      # Shape: (n_frames, n_segments, 3)
    corrected_data,     # Shape: (n_frames, n_segments, 3)
    output_path="comparison.png",
)

# With custom configuration
config = PlotConfig(
    figsize=(16, 8),
    dpi=300,
    colormap="plasma",
    frame_rate=60,
    unit_scale=1000,    # Display in mm
    unit_label="mm",
)

plot_trajectory_comparison(
    original_data,
    corrected_data,
    output_path="comparison.png",
    titles=["Before", "After"],
    suptitle="Drift Correction Results",
    config=config,
)
```

#### plot_multi_method_comparison()

Create grid comparison of multiple correction methods.

```python
from scripts.utils.plotting import plot_multi_method_comparison

results = [
    {"name": "Method A", "data": data_a},
    {"name": "Method B", "data": data_b},
    {"name": "Method C", "data": data_c},
]

plot_multi_method_comparison(
    results,
    original_data,
    output_path="methods.png",
    include_original=True,
    n_cols=2,
)
```

#### PlotConfig

Dataclass for consistent plot styling.

```python
from scripts.utils.plotting import PlotConfig

config = PlotConfig(
    figsize=(14, 6),        # Figure size
    dpi=150,                # Resolution
    colormap="viridis",     # Colormap for time coloring
    scatter_size=0.5,       # Point size
    scatter_alpha=0.5,      # Point transparency
    line_width=0.5,         # Line width
    grid_alpha=0.3,         # Grid transparency
    title_fontsize=11,      # Title font size
    label_fontsize=10,      # Axis label font size
    pelvis_index=0,         # Pelvis segment index
    frame_rate=60,          # Data frame rate
    unit_scale=1.0,         # Scale factor (1000 for mm)
    unit_label="m",         # Unit label
    show_colorbar=True,     # Show time colorbar
)
```

### Configuration Classes

```python
from scripts.utils.config import (
    setup_matplotlib,
    GaitCorrectionConfig,
    SegmentExtractionConfig,
    PlottingConfig,
    TurnaroundDetectionConfig,
    LSTMPreprocessingConfig,
    get_default_configs,
)
```

#### setup_matplotlib()

Configure matplotlib backend before importing pyplot.

```python
from scripts.utils.config import setup_matplotlib

# Must be called before importing matplotlib.pyplot
setup_matplotlib()          # Default: 'Agg' backend

# Or specify backend
setup_matplotlib(backend='TkAgg')

# Then import pyplot
import matplotlib.pyplot as plt
```

#### GaitCorrectionConfig

Parameters for drift correction algorithms.

```python
from scripts.utils.config import GaitCorrectionConfig

config = GaitCorrectionConfig(
    frame_rate=60,
    pelvis_index=0,
    drift_window_seconds=30.0,
    highpass_window_seconds=10.0,
    pca_window_seconds=30.0,
    reference_seconds=60.0,
)

# Computed properties
window_frames = config.drift_window_frames        # 1800
highpass_frames = config.highpass_window_frames   # 601 (always odd)
```

#### SegmentExtractionConfig

Parameters for segment extraction.

```python
from scripts.utils.config import SegmentExtractionConfig

config = SegmentExtractionConfig(
    frame_rate=60,
    velocity_threshold=0.4,
    heading_change_threshold=0.1,
    min_segment_meters=5.0,
)

# Computed properties
trim_frames = config.trim_start_frames  # 30
window_frames = config.window_frames    # 300
```

---

## Segment Extraction

### Pipeline API

```python
from scripts.gait_analysis.segment_extraction_pipeline import (
    SegmentExtractionConfig,
    extract_segments_from_data,
    get_segments_as_array,
)

# Configure extraction
config = SegmentExtractionConfig(
    frame_rate=60,
    use_preprocessing=True,
    drift_correction_strength='moderate',
    velocity_threshold=0.4,
    heading_change_threshold=0.1,
    min_segment_meters=5.0,
)

# Extract segments
segments, info = extract_segments_from_data(data, config=config)

# Convert to numpy array for ML
array = get_segments_as_array(segments, pad_to_length=300)
# Shape: (n_segments, n_frames, n_joints, 3)
```

---

## Example: Custom Analysis

```python
from src.data_loader import DataLoader
import numpy as np

# Load data
loader = DataLoader()
loader.load_csv("gait_trial.csv", frame_rate=100.0)

# Analyze hip angles across frames
all_positions = loader.get_all_positions()

for i, positions in enumerate(all_positions):
    hip_pos = positions.get("LHIP")
    knee_pos = positions.get("LKNE")

    if hip_pos is not None and knee_pos is not None:
        # Calculate hip-knee vector
        vector = knee_pos - hip_pos
        # Your analysis here...
```

## Example: Complete Correction Pipeline

```python
from scripts.gait_correction import load_xsens_data
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction, SmoothPCAParams
from scripts.gait_correction.drift_removal import apply_full_drift_correction
from scripts.utils.config import setup_matplotlib, GaitCorrectionConfig
from scripts.utils.plotting import plot_trajectory_comparison, PlotConfig

# Setup
setup_matplotlib()
config = GaitCorrectionConfig(frame_rate=60)

# Load data
loader = load_xsens_data("data/walking.xlsx", frame_rate=60)
original = loader.positions.copy()

# Apply corrections
pca_params = SmoothPCAParams(
    window_seconds=config.pca_window_seconds,
    sample_interval_seconds=config.pca_sample_interval_seconds,
    frame_rate=config.frame_rate,
)

corrected, _, _ = apply_smooth_pca_correction(original, params=pca_params)
corrected = apply_full_drift_correction(corrected, frame_rate=config.frame_rate)

# Visualize results
plot_config = PlotConfig(frame_rate=60, dpi=150)
plot_trajectory_comparison(
    original, corrected,
    output_path="correction_result.png",
    titles=["Original", "Corrected"],
    config=plot_config,
)
```

## Progress Callback

For long videos, use a progress callback:

```python
def show_progress(current, total):
    print(f"Processing frame {current}/{total}")

frames = visualizer.create_animation_frames(
    all_positions,
    frame_rate=100.0,
    progress_callback=show_progress
)
```

---

## Related Documentation

- [Getting Started](Getting-Started) - Installation and quick start
- [Refactoring Guide](Refactoring-Guide) - Shared utilities and configuration
- [Noise Removal](Noise-Removal) - Drift correction algorithms
- [Smooth PCA](Smooth-PCA) - PCA-based heading correction
