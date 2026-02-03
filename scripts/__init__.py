"""
Gait Analysis Scripts Package

This package contains modules for:
- gait_correction: Drift correction and noise removal
- gait_analysis: Segment extraction and feature analysis
- utils: Shared plotting and configuration utilities
- experiments: Experimental comparison scripts

After installing the package with `pip install -e .`, imports work without
sys.path manipulation:

    from scripts.gait_correction import load_xsens_data
    from scripts.gait_analysis import extract_segments_improved
    from scripts.utils import plot_trajectory_comparison
"""

__version__ = "0.1.0"
