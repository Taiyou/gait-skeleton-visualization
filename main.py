#!/usr/bin/env python3
"""
Gait Skeleton Visualization - Main Script

This script provides the main entry point for creating skeleton animations
from gait analysis marker data.

Usage:
    python main.py --input data/sample_gait.csv --output output/gait_video.mp4
    python main.py --input data/sample_gait.csv --output output/gait_video.mp4 --view 3d
    python main.py --input data/sample_gait.csv --output output/gait_video.mp4 --view sagittal
    python main.py --input data/sample_gait.csv --output output/gait_video.mp4 --view multi
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.skeleton_model import SkeletonModel
from src.visualizer_3d import Visualizer3D
from src.visualizer_2d import Visualizer2D, MultiViewVisualizer
from src.visualizer_combined import CombinedVisualizer
from src.video_exporter import VideoExporter


def progress_bar(current: int, total: int, bar_length: int = 40) -> None:
    """Display a progress bar in the console."""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    sys.stdout.write(f'\r  Progress: |{bar}| {percent*100:.1f}% ({current}/{total})')
    sys.stdout.flush()
    if current == total:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Create skeleton animation from gait marker data"
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path for the output MP4 file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/marker_sets.yaml",
        help="Path to marker set configuration file"
    )
    parser.add_argument(
        "--marker-set", "-m",
        type=str,
        default="simple",
        help="Name of the marker set to use (default: simple)"
    )
    parser.add_argument(
        "--view", "-v",
        type=str,
        choices=["3d", "sagittal", "frontal", "transverse", "multi"],
        default="3d",
        help="View type: 3d, sagittal, frontal, transverse, or multi (default: 3d)"
    )
    parser.add_argument(
        "--frame-rate", "-f",
        type=float,
        default=100.0,
        help="Input data frame rate in Hz (default: 100)"
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=30,
        help="Output video frame rate (default: 30)"
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show marker labels in visualization"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame for processing (default: 0)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame for processing (default: all frames)"
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=50,
        help="Size of marker points (default: 50)"
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Width of bone lines (default: 2.0)"
    )
    parser.add_argument(
        "--auto-skeleton",
        action="store_true",
        help="Automatically create skeleton from data markers (no connections)"
    )
    parser.add_argument(
        "--coordinate-suffix",
        type=str,
        choices=["uppercase", "lowercase"],
        default="uppercase",
        help="Coordinate column suffix case: uppercase (_X,_Y,_Z) or lowercase (_x,_y,_z)"
    )
    parser.add_argument(
        "--exclude-markers",
        nargs="+",
        default=[],
        help="List of marker names to exclude (e.g. mean)"
    )
    parser.add_argument(
        "--clean-3d",
        action="store_true",
        help="Remove all background, axes, and labels from 3D view"
    )
    parser.add_argument(
        "--phase-dynamics",
        action="store_true",
        help="Show 3D skeleton + joint angle time-series side by side"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Use every Nth frame (e.g. 2 = every other frame). "
             "Combine with --output-fps for real-time playback."
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom factor for 3D view (>1 = larger body, default: 1.0)"
    )

    args = parser.parse_args()
    
    print("=" * 60)
    print("Gait Skeleton Visualization")
    print("=" * 60)

    # Determine coordinate suffix
    if args.coordinate_suffix == "lowercase":
        coord_suffix = ("_x", "_y", "_z")
    else:
        coord_suffix = ("_X", "_Y", "_Z")

    # Load data
    print(f"\n1. Loading data from: {args.input}")
    loader = DataLoader()
    loader.load_csv(
        args.input,
        frame_rate=args.frame_rate,
        coordinate_suffix=coord_suffix,
    )
    
    # Load or create skeleton model
    print(f"\n2. Loading skeleton model...")
    config_path = Path(args.config)
    
    if args.auto_skeleton:
        # Create skeleton from data markers (no connections)
        print("   Using auto-skeleton mode (markers only, no connections)")
        skeleton = SkeletonModel.from_markers(loader.markers)
    elif config_path.exists():
        try:
            skeleton = SkeletonModel.from_yaml(str(config_path), args.marker_set)
            print(f"   Loaded marker set: {skeleton.name}")
        except ValueError as e:
            print(f"   Warning: {e}")
            print("   Creating skeleton from data markers...")
            skeleton = SkeletonModel.from_markers(loader.markers)
    else:
        print(f"   Config file not found: {config_path}")
        print("   Creating skeleton from data markers...")
        skeleton = SkeletonModel.from_markers(loader.markers)
    
    # Exclude specified markers
    available_markers = [m for m in loader.markers if m not in args.exclude_markers]
    if args.exclude_markers:
        print(f"   Excluding markers: {args.exclude_markers}")

    # Filter skeleton to only include markers in the data
    skeleton = skeleton.filter_markers(available_markers)
    print(f"   Active markers: {len(skeleton.markers)}")
    print(f"   Active connections: {len(skeleton.connections)}")

    # Get data bounds
    bounds = loader.get_data_bounds(coordinate_suffix=coord_suffix)
    print(f"\n3. Data bounds:")
    print(f"   X: {bounds['x'][0]:.1f} to {bounds['x'][1]:.1f}")
    print(f"   Y: {bounds['y'][0]:.1f} to {bounds['y'][1]:.1f}")
    print(f"   Z: {bounds['z'][0]:.1f} to {bounds['z'][1]:.1f}")
    
    # Get positions for selected frame range
    start = args.start_frame
    end = args.end_frame if args.end_frame is not None else loader.num_frames
    end = min(end, loader.num_frames)
    
    print(f"\n4. Processing frames {start} to {end} ({end - start} frames)")
    exclude_set = set(args.exclude_markers)
    all_positions = []
    for frame in range(start, end, args.subsample):
        pos = loader.get_marker_positions(frame, coordinate_suffix=coord_suffix)
        if exclude_set:
            pos = {k: v for k, v in pos.items() if k not in exclude_set}
        all_positions.append(pos)

    # Effective frame rate after subsampling
    effective_frame_rate = args.frame_rate / args.subsample
    if args.subsample > 1:
        print(f"   Subsampled every {args.subsample} frames -> {len(all_positions)} frames"
              f" (effective {effective_frame_rate:.1f} Hz)")
    
    # Create visualizer based on view type and mode
    if args.phase_dynamics:
        mode_label = "phase-dynamics (3D + joint angles)"
    elif args.clean_3d:
        mode_label = "clean 3D"
    else:
        mode_label = args.view
    print(f"\n5. Creating {mode_label} visualization...")

    if args.phase_dynamics:
        visualizer = CombinedVisualizer(
            skeleton,
            marker_size=args.marker_size,
            line_width=args.line_width,
            clean_mode=args.clean_3d,
            zoom=args.zoom,
        )
    elif args.view == "3d":
        visualizer = Visualizer3D(
            skeleton,
            marker_size=args.marker_size,
            line_width=args.line_width,
            clean_mode=args.clean_3d,
            zoom=args.zoom,
        )
    elif args.view == "multi":
        visualizer = MultiViewVisualizer(
            skeleton,
            views=["sagittal", "frontal", "transverse"],
            marker_size=args.marker_size,
            line_width=args.line_width,
        )
    else:
        visualizer = Visualizer2D(
            skeleton,
            view=args.view,
            marker_size=args.marker_size,
            line_width=args.line_width,
        )
    
    visualizer.set_bounds(bounds)
    
    # Generate frames
    print("\n6. Generating animation frames...")
    frames = visualizer.create_animation_frames(
        all_positions,
        frame_rate=effective_frame_rate,
        show_labels=args.show_labels,
        progress_callback=progress_bar,
    )
    
    # Export video
    print(f"\n7. Exporting video to: {args.output}")
    exporter = VideoExporter(output_fps=args.output_fps)
    exporter.export(frames, args.output, progress_callback=progress_bar)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
