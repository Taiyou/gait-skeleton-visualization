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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

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

## Next Steps

- Learn about [CSV Data Format](CSV-Data-Format) for your own data
- Customize [Marker Set Configuration](Marker-Set-Configuration)
- Explore all [View Types](View-Types)
