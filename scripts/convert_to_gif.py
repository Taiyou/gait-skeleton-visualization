#!/usr/bin/env python3
"""
Convert MP4 video to GIF for README embedding.
"""

import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path


def convert_mp4_to_gif(
    input_path: str,
    output_path: str,
    fps: int = 10,
    scale: float = 0.5,
    max_frames: int = 100,
    skip_frames: int = 3
):
    """
    Convert MP4 to GIF.

    Args:
        input_path: Path to input MP4 file
        output_path: Path to output GIF file
        fps: Output GIF frame rate
        scale: Scale factor for resizing (0.5 = half size)
        max_frames: Maximum number of frames to include
        skip_frames: Skip every N frames to reduce size
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce file size
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        if scale != 1.0:
            new_width = int(frame_rgb.shape[1] * scale)
            new_height = int(frame_rgb.shape[0] * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)

        frame_count += 1

        if len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise ValueError("No frames extracted from video")

    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )

    print(f"Converted {len(frames)} frames to GIF: {output_path}")

    # Get file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"GIF size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 to GIF")
    parser.add_argument("-i", "--input", required=True, help="Input MP4 file")
    parser.add_argument("-o", "--output", required=True, help="Output GIF file")
    parser.add_argument("--fps", type=int, default=10, help="Output FPS")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames")
    parser.add_argument("--skip", type=int, default=3, help="Skip every N frames")

    args = parser.parse_args()

    convert_mp4_to_gif(
        args.input,
        args.output,
        fps=args.fps,
        scale=args.scale,
        max_frames=args.max_frames,
        skip_frames=args.skip
    )


if __name__ == "__main__":
    main()
