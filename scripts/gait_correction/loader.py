"""
Xsens Data Loader
Handles loading motion capture data from Xsens Excel files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class XsensDataLoader:
    """Container for loaded Xsens data."""
    positions: np.ndarray  # Shape: (n_frames, n_segments, 3)
    segment_names: List[str]
    frame_rate: int
    n_frames: int

    def get_segment_index(self, name: str) -> int:
        """Get index of a segment by name."""
        return self.segment_names.index(name)

    def get_pelvis_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get pelvis X and Y coordinates."""
        pelvis_idx = self.get_segment_index("Pelvis")
        return (
            self.positions[:, pelvis_idx, 0],
            self.positions[:, pelvis_idx, 1],
        )


def load_xsens_data(
    filepath: Union[str, Path],
    sheet_name: str = "Segment Position",
    frame_rate: int = 60,
    skip_header_rows: int = 0,
) -> XsensDataLoader:
    """
    Load Xsens motion capture data from Excel file.

    Args:
        filepath: Path to the Excel file
        sheet_name: Name of the sheet containing position data
        frame_rate: Frame rate of the recording (Hz)
        skip_header_rows: Number of header rows to skip

    Returns:
        XsensDataLoader containing the loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading data from {filepath.name}...")

    # Read Excel file
    df = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        skiprows=skip_header_rows,
    )

    # Extract segment names from columns
    # Columns are: Frame, Segment1 x, Segment1 y, Segment1 z, Segment2 x, ...
    columns = list(df.columns)

    # Skip 'Frame' column if present
    start_col = 1 if columns[0].lower() == 'frame' else 0

    # Extract unique segment names
    segment_names = []
    for col in columns[start_col:]:
        col_str = str(col)
        # Remove coordinate suffix (x, y, z)
        if col_str.endswith(' x'):
            segment_name = col_str[:-2]
            if segment_name not in segment_names:
                segment_names.append(segment_name)

    n_frames = len(df)
    n_segments = len(segment_names)

    print(f"Found {n_frames} frames, {n_segments} segments")
    print(f"Segments: {segment_names}")

    # Build position array
    positions = np.zeros((n_frames, n_segments, 3))

    for i, segment in enumerate(segment_names):
        x_col = f"{segment} x"
        y_col = f"{segment} y"
        z_col = f"{segment} z"

        if all(col in df.columns for col in [x_col, y_col, z_col]):
            positions[:, i, 0] = df[x_col].values
            positions[:, i, 1] = df[y_col].values
            positions[:, i, 2] = df[z_col].values
        else:
            print(f"Warning: Missing columns for segment {segment}")

    return XsensDataLoader(
        positions=positions,
        segment_names=segment_names,
        frame_rate=frame_rate,
        n_frames=n_frames,
    )
