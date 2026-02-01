"""
Data Export Module
Exports corrected data to various formats.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union


def export_to_csv(
    data: np.ndarray,
    output_path: Union[str, Path],
    segment_names: List[str],
    frame_rate: int = 60,
    subsample: int = 1,
    include_frame_column: bool = True,
) -> Path:
    """
    Export position data to CSV file.

    Args:
        data: Position data, shape (n_frames, n_segments, 3)
        output_path: Output file path
        segment_names: List of segment names
        frame_rate: Frame rate in Hz
        subsample: Subsample factor (e.g., 2 = every 2nd frame)
        include_frame_column: Whether to include frame number column

    Returns:
        Path to the created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames, n_segments, _ = data.shape

    # Subsample if requested
    if subsample > 1:
        data = data[::subsample]
        n_frames = len(data)

    # Build DataFrame
    df_data = {}

    if include_frame_column:
        df_data['Frame'] = np.arange(n_frames)

    for i, segment in enumerate(segment_names):
        df_data[f'{segment}_X'] = data[:, i, 0]
        df_data[f'{segment}_Y'] = data[:, i, 1]
        df_data[f'{segment}_Z'] = data[:, i, 2]

    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)

    print(f"Exported to {output_path}")
    print(f"  Frames: {n_frames}")
    print(f"  Segments: {n_segments}")
    if subsample > 1:
        print(f"  Subsampled by factor {subsample}")

    return output_path


def export_to_excel(
    data: np.ndarray,
    output_path: Union[str, Path],
    segment_names: List[str],
    frame_rate: int = 60,
    subsample: int = 1,
) -> Path:
    """
    Export position data to Excel file.

    Args:
        data: Position data, shape (n_frames, n_segments, 3)
        output_path: Output file path
        segment_names: List of segment names
        frame_rate: Frame rate in Hz
        subsample: Subsample factor

    Returns:
        Path to the created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames, n_segments, _ = data.shape

    # Subsample if requested
    if subsample > 1:
        data = data[::subsample]
        n_frames = len(data)

    # Build DataFrame with Xsens-style column names
    df_data = {'Frame': np.arange(n_frames)}

    for i, segment in enumerate(segment_names):
        df_data[f'{segment} x'] = data[:, i, 0]
        df_data[f'{segment} y'] = data[:, i, 1]
        df_data[f'{segment} z'] = data[:, i, 2]

    df = pd.DataFrame(df_data)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Segment Position', index=False)

    print(f"Exported to {output_path}")
    print(f"  Frames: {n_frames}")
    print(f"  Segments: {n_segments}")

    return output_path
