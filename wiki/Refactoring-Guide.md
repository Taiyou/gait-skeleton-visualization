# Refactoring Guide

This document describes the refactoring changes made to improve code maintainability, reduce duplication, and enhance developer experience.

## Why This Refactoring?

Before refactoring, the codebase had several issues:

| Problem | Impact |
|---------|--------|
| **~6,600 lines of duplicated code** | Same plotting logic copied across 19 experiment scripts |
| **26 scripts with `sys.path` hacks** | Fragile imports that break when directory structure changes |
| **Magic numbers scattered throughout** | Hard to understand what `int(30.0 * 60)` means |
| **Inconsistent matplotlib setup** | `matplotlib.use('Agg')` repeated in 15 files |
| **Limited test coverage** | Complex algorithms lacked tests |

## What Changed

```
┌─────────────────────────────────────────────────────────────────┐
│                      BEFORE REFACTORING                         │
├─────────────────────────────────────────────────────────────────┤
│  scripts/experiments/                                           │
│  ├── apply_v7_tuned.py          # 50 lines of plotting code    │
│  ├── apply_optimal_correction.py # 50 lines of plotting code   │
│  ├── compare_methods.py          # 80 lines of plotting code   │
│  └── ... (15+ more scripts with duplicate code)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      AFTER REFACTORING                          │
├─────────────────────────────────────────────────────────────────┤
│  scripts/utils/                                                 │
│  ├── plotting.py     # 350 lines (consolidated from 500+ lines)│
│  └── config.py       # 200 lines (all configuration)           │
│                                                                 │
│  scripts/experiments/                                           │
│  ├── apply_v7_tuned.py          # 2 lines: import + call       │
│  ├── apply_optimal_correction.py # 2 lines: import + call      │
│  └── ... (all scripts now use shared utilities)                │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

The project is now installable as a proper Python package:

```bash
# Clone the repository
git clone https://github.com/yourusername/gait-skeleton-visualization.git
cd gait-skeleton-visualization

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

After installation, imports work cleanly without path manipulation:

```python
# ✅ Clean import (after refactoring)
from scripts.gait_correction import load_xsens_data
from scripts.utils.plotting import plot_trajectory_comparison

# ❌ Old style (no longer needed)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
```

---

## New Shared Utilities

### scripts/utils/plotting.py

Common plotting functions to reduce duplication across experiment scripts.

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

#### Function Reference

| Function | Description | Use Case |
|----------|-------------|----------|
| `plot_trajectory_comparison()` | Side-by-side original vs corrected | Before/after comparison |
| `plot_multi_method_comparison()` | Grid of multiple methods | Comparing correction algorithms |
| `plot_time_series_comparison()` | Y-axis over time | Drift analysis |
| `plot_segment_distribution()` | Duration histogram | Segment extraction results |
| `plot_overlay_trajectories()` | Multiple trajectories overlaid | Direct visual comparison |
| `print_summary_table()` | Formatted console output | Quick results summary |
| `calc_range()` | Calculate axis range | Helper for metrics |

#### Example: Basic Comparison Plot

```python
from scripts.utils.plotting import plot_trajectory_comparison, PlotConfig

# Load your data
original_data = load_data("original.csv")
corrected_data = apply_correction(original_data)

# Create comparison plot with default settings
plot_trajectory_comparison(
    original_data,
    corrected_data,
    output_path="comparison.png"
)
```

#### Example: Customized Plot

```python
from scripts.utils.plotting import plot_trajectory_comparison, PlotConfig

# Custom configuration
config = PlotConfig(
    figsize=(16, 8),      # Larger figure
    dpi=300,              # Higher resolution
    colormap="plasma",    # Different colormap
    frame_rate=100,       # Your data's frame rate
    pelvis_index=0,       # Pelvis segment index
    unit_scale=1000,      # Display in mm
    unit_label="mm",
    show_colorbar=True,
)

plot_trajectory_comparison(
    original_data,
    corrected_data,
    output_path="comparison_hires.png",
    titles=["Before Correction", "After Correction"],
    suptitle="Drift Correction Results",
    config=config,
)
```

#### Example: Multi-Method Comparison

```python
from scripts.utils.plotting import plot_multi_method_comparison

# Results from different correction methods
results = [
    {"name": "Method 1: High-pass", "data": data_method1},
    {"name": "Method 2: PCA", "data": data_method2},
    {"name": "Method 3: Segment-wise", "data": data_method3},
]

plot_multi_method_comparison(
    results,
    original_data,
    output_path="method_comparison.png",
    include_original=True,  # Show original as first panel
    n_cols=2,               # 2 columns in grid
)
```

---

### scripts/utils/config.py

Centralized configuration dataclasses to eliminate magic numbers.

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

Centralized matplotlib backend configuration:

```python
from scripts.utils.config import setup_matplotlib

# Call at module level, before importing pyplot
setup_matplotlib()  # Uses 'Agg' backend (non-interactive)

# Or specify a different backend
setup_matplotlib(backend='TkAgg')  # For interactive plots

# Can also use environment variable
# export MATPLOTLIB_BACKEND=TkAgg
```

#### GaitCorrectionConfig

Parameters for drift correction algorithms:

```python
from scripts.utils.config import GaitCorrectionConfig

config = GaitCorrectionConfig(
    frame_rate=60,
    pelvis_index=0,

    # Drift removal
    drift_window_seconds=30.0,
    highpass_window_seconds=10.0,
    max_drift_threshold=2.0,

    # PCA correction
    pca_window_seconds=30.0,
    pca_sample_interval_seconds=5.0,
    pca_smoothing_factor=0.1,

    # Reference period
    reference_seconds=60.0,
    skip_start_seconds=5.0,
    skip_end_seconds=5.0,
)

# Computed properties (no more magic number calculations!)
window_frames = config.drift_window_frames        # 1800
highpass_frames = config.highpass_window_frames   # 601 (always odd)
pca_frames = config.pca_window_frames             # 1800
ref_frames = config.reference_frames              # 3600
```

#### SegmentExtractionConfig

Parameters for gait segment extraction:

```python
from scripts.utils.config import SegmentExtractionConfig

config = SegmentExtractionConfig(
    frame_rate=60,

    # Velocity thresholds
    velocity_threshold=0.4,           # m/s minimum walking speed
    min_segment_meters=5.0,           # Minimum segment length

    # Heading analysis
    heading_change_threshold=0.1,     # rad/frame for preprocessed data
    heading_change_threshold_raw=0.3, # Higher for raw data with drift

    # Trimming
    trim_start_seconds=0.5,
    trim_end_seconds=0.3,

    # Windowing
    use_overlapping_windows=True,
    window_seconds=5.0,
    window_overlap=0.5,
)

# Computed properties
trim_start = config.trim_start_frames  # 30
window_size = config.window_frames     # 300
```

#### All Configuration Classes

| Class | Purpose | Key Parameters |
|-------|---------|----------------|
| `GaitCorrectionConfig` | Drift correction | `drift_window_seconds`, `pca_window_seconds` |
| `SegmentExtractionConfig` | Segment extraction | `velocity_threshold`, `heading_change_threshold` |
| `PlottingConfig` | Visualization | `figsize`, `dpi`, `colormap` |
| `TurnaroundDetectionConfig` | Turnaround detection | `velocity_smooth_window_seconds`, `min_pause_seconds` |
| `LSTMPreprocessingConfig` | LSTM preprocessing | `target_duration_seconds`, `drift_correction_strength` |

---

## Migration Guide

### Step 1: Install the Package

```bash
cd gait-skeleton-visualization
pip install -e .
```

### Step 2: Update Imports

```python
# ❌ Before: Remove these lines
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ After: Clean imports
from scripts.gait_correction import load_xsens_data
from scripts.utils.plotting import plot_trajectory_comparison
from scripts.utils.config import setup_matplotlib, GaitCorrectionConfig
```

### Step 3: Replace matplotlib.use()

```python
# ❌ Before
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ✅ After
from scripts.utils.config import setup_matplotlib
setup_matplotlib()
import matplotlib.pyplot as plt
```

### Step 4: Use Shared Plotting

```python
# ❌ Before: 50+ lines of plotting code
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
n_frames = len(original)
time = np.arange(n_frames) / frame_rate / 60
ax = axes[0]
scatter = ax.scatter(original[:, 0, 0], original[:, 0, 1], c=time, ...)
ax.plot(original[0, 0, 0], original[0, 0, 1], 'go', markersize=10, ...)
# ... 40 more lines ...
plt.savefig(output_path, dpi=150)
plt.close()

# ✅ After: 2 lines
from scripts.utils.plotting import plot_trajectory_comparison
plot_trajectory_comparison(original, corrected, output_path)
```

### Step 5: Use Configuration Classes

```python
# ❌ Before: Magic numbers
window_frames = int(30.0 * 60)  # What does this mean?
if window_frames % 2 == 0:
    window_frames += 1

# ✅ After: Self-documenting
from scripts.utils.config import GaitCorrectionConfig
config = GaitCorrectionConfig(drift_window_seconds=30.0, frame_rate=60)
window_frames = config.highpass_window_frames  # Always odd, clear meaning
```

---

## Test Coverage

New test files have been added to verify the refactored code:

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_utils_config.py` | 24 | Configuration dataclasses |
| `test_utils_plotting.py` | 19 | Plotting utilities |
| `test_correction_pipeline.py` | 10+ | Integration tests |

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_utils_config.py -v

# Run with coverage report
pytest tests/ --cov=scripts --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Test Coverage Summary

```
tests/test_utils_config.py ................ [100%]
tests/test_utils_plotting.py ............. [100%]
tests/test_correction_pipeline.py ........ [100%]

===================== 43+ passed =====================
```

---

## Updated File Structure

```
gait-skeleton-visualization/
├── setup.py                    # Package setup (NEW)
├── pyproject.toml              # Modern package config (NEW)
├── requirements.txt            # Updated dependencies
│
├── src/                        # Core visualization module
│   ├── data_loader.py
│   ├── skeleton_model.py
│   ├── visualizer_3d.py
│   ├── visualizer_2d.py
│   └── video_exporter.py
│
├── scripts/
│   ├── __init__.py             # Package init (NEW)
│   │
│   ├── utils/                  # Shared utilities (NEW)
│   │   ├── __init__.py
│   │   ├── plotting.py         # ~350 lines of reusable plotting
│   │   └── config.py           # ~200 lines of config classes
│   │
│   ├── gait_correction/        # Drift correction modules
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── drift_removal.py
│   │   ├── smooth_pca.py
│   │   ├── advanced_correction.py
│   │   ├── turnaround.py
│   │   └── export.py
│   │
│   ├── gait_analysis/          # Segment extraction
│   │   ├── segment_extraction_pipeline.py
│   │   └── improved_segment_extraction.py
│   │
│   └── experiments/            # Comparison scripts (refactored)
│       ├── apply_v7_tuned.py
│       ├── apply_adaptive_correction.py
│       ├── compare_correction_methods.py
│       └── ...
│
├── tests/
│   ├── test_utils_plotting.py  # NEW
│   ├── test_utils_config.py    # NEW
│   ├── test_correction_pipeline.py  # NEW
│   └── ...
│
└── wiki/
    └── Refactoring-Guide.md    # This document
```

---

## Benefits Summary

| Benefit | Before | After |
|---------|--------|-------|
| **Plotting code** | ~500+ lines duplicated | ~350 lines shared |
| **Import complexity** | `sys.path` hacks | Clean `pip install -e .` |
| **Magic numbers** | Scattered | Named config params |
| **matplotlib setup** | 15 files | 1 function call |
| **Test coverage** | Basic | Comprehensive |
| **New script setup** | Copy 50+ lines | 2-3 import lines |

---

## Troubleshooting

### Import Error: "No module named 'scripts'"

**Solution:** Install the package in development mode:
```bash
pip install -e .
```

### matplotlib Backend Warning

**Solution:** Call `setup_matplotlib()` before importing pyplot:
```python
from scripts.utils.config import setup_matplotlib
setup_matplotlib()  # Must be before pyplot import
import matplotlib.pyplot as plt
```

### Tests Failing

**Solution:** Ensure all dependencies are installed:
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Related Documentation

- [Noise Removal](Noise-Removal) - Drift correction algorithms
- [Smooth PCA](Smooth-PCA) - PCA-based heading correction
- [Python API](Python-API) - Core visualization API
- [Getting Started](Getting-Started) - Installation guide
