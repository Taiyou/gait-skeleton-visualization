#!/usr/bin/env python3
"""
Setup script for gait-skeleton-visualization package.

This allows the package to be installed in development mode:
    pip install -e .

After installation, imports work uniformly without sys.path manipulation:
    from src.data_loader import load_csv
    from scripts.gait_correction.loader import load_xsens_data
    from scripts.utils.plotting import plot_trajectory_comparison
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="gait-skeleton-visualization",
    version="0.1.0",
    author="Gait Analysis Team",
    description="Skeleton animation visualization from gait analysis marker data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gait-skeleton-visualization",
    packages=find_packages(include=["src", "src.*", "scripts", "scripts.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.5.0",
        "pyyaml>=6.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "openpyxl>=3.0.0",  # For Excel file reading
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gait-viz=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="gait analysis, skeleton visualization, motion capture, biomechanics",
)
