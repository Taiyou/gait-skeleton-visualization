# Getting Started

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Taiyou/gait-skeleton-visualization.git
cd gait-skeleton-visualization
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

### 3. Install the Package

**Option A: Install as a package (Recommended)**

```bash
# Install in development mode
pip install -e .

# Or install with development dependencies (pytest, black, etc.)
pip install -e ".[dev]"
```

**Option B: Install dependencies only**

```bash
pip install -r requirements.txt
```

> **Note:** Installing as a package (Option A) is recommended because it enables clean imports without `sys.path` manipulation. See [Refactoring Guide](Refactoring-Guide) for details.

## Quick Start

### Generate Sample Data

```bash
python scripts/generate_realistic_gait.py
```

This creates test CSV files in the `data/` folder with realistic walking motion.

### Create Your First Video

```bash
# Create a sagittal (side) view video
python main.py -i data/realistic_gait.csv -o output/skeleton_sagittal.mp4 -v sagittal
```

### Try Different Views

```bash
# 3D view
python main.py -i data/realistic_gait.csv -o output/skeleton_3d.mp4 -v 3d

# Frontal (front) view
python main.py -i data/realistic_gait.csv -o output/skeleton_frontal.mp4 -v frontal

# Multi-view (all three planes)
python main.py -i data/realistic_gait.csv -o output/skeleton_multi.mp4 -v multi
```

## Using the Python API

After installing the package, you can use the Python API directly:

```python
from scripts.gait_correction import load_xsens_data
from scripts.utils.plotting import plot_trajectory_comparison
from scripts.utils.config import setup_matplotlib

# Setup matplotlib for non-interactive use
setup_matplotlib()

# Load and process your data
loader = load_xsens_data("data/walking.xlsx", frame_rate=60)
print(f"Loaded {loader.positions.shape[0]} frames")
```

See [Python API](Python-API) for complete documentation.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html
```

## Next Steps

- Learn about [CSV Data Format](CSV-Data-Format) for your own data
- Customize [Marker Set Configuration](Marker-Set-Configuration)
- Explore all [View Types](View-Types)
- Apply [Noise Removal](Noise-Removal) for drift correction
- Use [Shared Utilities](Refactoring-Guide) for custom scripts
- Check the [Python API](Python-API) for programmatic access
