"""
Data Loader Module
Handles loading and parsing of marker data from various file formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class DataLoader:
    """
    Loads marker data from CSV files.
    
    Expected CSV format:
    - First column: Frame number or Time
    - Subsequent columns: MarkerName_X, MarkerName_Y, MarkerName_Z
    
    Example:
        Frame, HEAD_X, HEAD_Y, HEAD_Z, LSHO_X, LSHO_Y, LSHO_Z, ...
        0, 100.5, 200.3, 1500.2, 85.2, 180.5, 1450.3, ...
        1, 100.6, 200.4, 1500.1, 85.3, 180.6, 1450.2, ...
    """
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.markers: List[str] = []
        self.frame_rate: float = 100.0  # Default frame rate (Hz)
        self.num_frames: int = 0
        
    def load_csv(
        self,
        filepath: Union[str, Path],
        frame_rate: float = 100.0,
        time_column: Optional[str] = None,
        header_row: int = 0,
        skip_rows: int = 0,
        coordinate_suffix: Tuple[str, str, str] = ("_X", "_Y", "_Z"),
    ) -> "DataLoader":
        """
        Load marker data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            frame_rate: Recording frame rate in Hz (default: 100)
            time_column: Name of the time/frame column (auto-detected if None)
            header_row: Row number containing column headers (default: 0)
            skip_rows: Number of rows to skip at the beginning (default: 0)
            coordinate_suffix: Tuple of suffixes for X, Y, Z coordinates
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load CSV
        self.data = pd.read_csv(
            filepath,
            header=header_row,
            skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None,
        )
        
        self.frame_rate = frame_rate
        self.num_frames = len(self.data)
        
        # Extract marker names from columns
        self._extract_marker_names(coordinate_suffix)
        
        print(f"Loaded {self.num_frames} frames with {len(self.markers)} markers")
        print(f"Markers: {self.markers}")
        
        return self
    
    def _extract_marker_names(
        self, 
        coordinate_suffix: Tuple[str, str, str] = ("_X", "_Y", "_Z")
    ) -> None:
        """Extract unique marker names from column headers."""
        if self.data is None:
            return
            
        x_suffix, y_suffix, z_suffix = coordinate_suffix
        markers = set()
        
        for col in self.data.columns:
            col_str = str(col)
            for suffix in [x_suffix, y_suffix, z_suffix]:
                if col_str.endswith(suffix):
                    marker_name = col_str[:-len(suffix)]
                    markers.add(marker_name)
                    break
        
        self.markers = sorted(list(markers))
    
    def get_marker_positions(
        self,
        frame: int,
        coordinate_suffix: Tuple[str, str, str] = ("_X", "_Y", "_Z"),
    ) -> Dict[str, np.ndarray]:
        """
        Get positions of all markers for a specific frame.
        
        Args:
            frame: Frame number (0-indexed)
            coordinate_suffix: Tuple of suffixes for X, Y, Z coordinates
            
        Returns:
            Dictionary mapping marker names to their 3D positions
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        if frame < 0 or frame >= self.num_frames:
            raise ValueError(f"Frame {frame} out of range [0, {self.num_frames - 1}]")
        
        x_suffix, y_suffix, z_suffix = coordinate_suffix
        positions = {}
        
        for marker in self.markers:
            x_col = f"{marker}{x_suffix}"
            y_col = f"{marker}{y_suffix}"
            z_col = f"{marker}{z_suffix}"
            
            if all(col in self.data.columns for col in [x_col, y_col, z_col]):
                x = self.data.iloc[frame][x_col]
                y = self.data.iloc[frame][y_col]
                z = self.data.iloc[frame][z_col]
                
                # Handle NaN values
                if pd.isna(x) or pd.isna(y) or pd.isna(z):
                    positions[marker] = None
                else:
                    positions[marker] = np.array([x, y, z], dtype=np.float64)
        
        return positions
    
    def get_all_positions(
        self,
        coordinate_suffix: Tuple[str, str, str] = ("_X", "_Y", "_Z"),
    ) -> List[Dict[str, np.ndarray]]:
        """
        Get positions of all markers for all frames.
        
        Returns:
            List of dictionaries, one per frame
        """
        return [
            self.get_marker_positions(frame, coordinate_suffix)
            for frame in range(self.num_frames)
        ]
    
    def get_marker_trajectory(
        self,
        marker_name: str,
        coordinate_suffix: Tuple[str, str, str] = ("_X", "_Y", "_Z"),
    ) -> np.ndarray:
        """
        Get the trajectory of a single marker across all frames.
        
        Args:
            marker_name: Name of the marker
            coordinate_suffix: Tuple of suffixes for X, Y, Z coordinates
            
        Returns:
            Array of shape (num_frames, 3) containing XYZ positions
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        x_suffix, y_suffix, z_suffix = coordinate_suffix
        x_col = f"{marker_name}{x_suffix}"
        y_col = f"{marker_name}{y_suffix}"
        z_col = f"{marker_name}{z_suffix}"
        
        if not all(col in self.data.columns for col in [x_col, y_col, z_col]):
            raise ValueError(f"Marker '{marker_name}' not found in data")
        
        trajectory = np.column_stack([
            self.data[x_col].values,
            self.data[y_col].values,
            self.data[z_col].values,
        ])
        
        return trajectory
    
    def get_data_bounds(
        self,
        coordinate_suffix: Tuple[str, str, str] = ("_X", "_Y", "_Z"),
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get the min/max bounds of the marker data.
        
        Returns:
            Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        x_suffix, y_suffix, z_suffix = coordinate_suffix
        
        x_cols = [c for c in self.data.columns if str(c).endswith(x_suffix)]
        y_cols = [c for c in self.data.columns if str(c).endswith(y_suffix)]
        z_cols = [c for c in self.data.columns if str(c).endswith(z_suffix)]
        
        x_min = self.data[x_cols].min().min()
        x_max = self.data[x_cols].max().max()
        y_min = self.data[y_cols].min().min()
        y_max = self.data[y_cols].max().max()
        z_min = self.data[z_cols].min().min()
        z_max = self.data[z_cols].max().max()
        
        return {
            'x': (x_min, x_max),
            'y': (y_min, y_max),
            'z': (z_min, z_max),
        }
    
    def resample(self, target_frame_rate: float) -> "DataLoader":
        """
        Resample data to a different frame rate.
        
        Args:
            target_frame_rate: Target frame rate in Hz
            
        Returns:
            New DataLoader with resampled data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        ratio = target_frame_rate / self.frame_rate
        new_num_frames = int(self.num_frames * ratio)
        
        # Create new index
        old_index = np.arange(self.num_frames)
        new_index = np.linspace(0, self.num_frames - 1, new_num_frames)
        
        # Interpolate each column
        new_data = {}
        for col in self.data.columns:
            if self.data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                new_data[col] = np.interp(new_index, old_index, self.data[col].values)
            else:
                # For non-numeric columns, use nearest neighbor
                indices = np.round(new_index).astype(int)
                new_data[col] = self.data[col].values[indices]
        
        new_loader = DataLoader()
        new_loader.data = pd.DataFrame(new_data)
        new_loader.markers = self.markers.copy()
        new_loader.frame_rate = target_frame_rate
        new_loader.num_frames = new_num_frames
        
        return new_loader
