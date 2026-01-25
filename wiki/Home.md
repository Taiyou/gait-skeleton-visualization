# Gait Skeleton Visualization Wiki

Welcome to the Gait Skeleton Visualization wiki! This tool creates skeleton animation videos from gait analysis marker data.

## Quick Navigation

- [Getting Started](Getting-Started)
- [CSV Data Format](CSV-Data-Format)
- [Marker Set Configuration](Marker-Set-Configuration)
- [View Types](View-Types)
- [Python API](Python-API)
- [Troubleshooting](Troubleshooting)

## Features

- Load CSV marker data from motion capture systems
- Customizable skeleton models via YAML configuration
- Multiple visualization options:
  - 3D view with rotation
  - Sagittal plane (side view)
  - Frontal plane (front view)
  - Transverse plane (top view)
  - Multi-view display
- Export to MP4 video format

## Coordinate System

The tool uses a standard biomechanics coordinate system:

| Axis | Direction | Description |
|------|-----------|-------------|
| X | Anterior-Posterior | Forward (+) / Backward (-) |
| Y | Medio-Lateral | Left (+) / Right (-) |
| Z | Vertical | Up (+) / Down (-) |
