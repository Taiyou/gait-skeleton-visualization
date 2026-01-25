# Python API

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
