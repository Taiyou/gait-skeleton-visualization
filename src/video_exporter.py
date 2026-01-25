"""
Video Exporter Module
Exports animation frames to video files.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Callable

# Try to import OpenCV, fall back to matplotlib if not available
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available. Using matplotlib for video export.")


class VideoExporter:
    """
    Exports animation frames to MP4 video files.
    """
    
    def __init__(
        self,
        output_fps: int = 30,
        codec: str = "mp4v",
        quality: int = 95,
    ):
        """
        Initialize the video exporter.
        
        Args:
            output_fps: Output video frame rate
            codec: Video codec (mp4v, avc1, XVID, etc.)
            quality: Video quality (0-100, higher is better)
        """
        self.output_fps = output_fps
        self.codec = codec
        self.quality = quality
        
    def export(
        self,
        frames: List[np.ndarray],
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Export frames to a video file.
        
        Args:
            frames: List of frame images as numpy arrays (RGB format)
            output_path: Path for the output video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the created video file
        """
        if not frames:
            raise ValueError("No frames to export")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        num_frames = len(frames)
        
        video_written = False
        
        if HAS_OPENCV:
            # Try multiple codecs for compatibility
            codecs_to_try = [self.codec, 'avc1', 'XVID', 'MJPG']
            
            for codec in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        self.output_fps,
                        (width, height)
                    )
                    
                    if not writer.isOpened():
                        writer.release()
                        continue
                    
                    for i, frame in enumerate(frames):
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame_bgr)
                        
                        if progress_callback:
                            progress_callback(i + 1, num_frames)
                    
                    writer.release()
                    video_written = True
                    print(f"Used codec: {codec}")
                    break
                    
                except Exception as e:
                    print(f"Codec {codec} failed: {e}")
                    continue
        
        if not video_written:
            # Use matplotlib for video export as fallback
            print("OpenCV video export failed, using matplotlib fallback...")
            self._export_with_matplotlib(frames, output_path, progress_callback)
        
        print(f"Video saved to: {output_path}")
        print(f"Resolution: {width}x{height}, FPS: {self.output_fps}, Frames: {num_frames}")
        
        return str(output_path)
    
    def _export_with_matplotlib(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Export using matplotlib animation (fallback when OpenCV unavailable)."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        
        fig, ax = plt.subplots(figsize=(frames[0].shape[1]/100, frames[0].shape[0]/100), dpi=100)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        im = ax.imshow(frames[0])
        
        def update(frame_idx):
            im.set_array(frames[frame_idx])
            if progress_callback:
                progress_callback(frame_idx + 1, len(frames))
            return [im]
        
        anim = FuncAnimation(
            fig, update, frames=len(frames),
            interval=1000/self.output_fps, blit=True
        )
        
        # Try FFmpeg first, fall back to Pillow for GIF
        output_str = str(output_path)
        try:
            writer = FFMpegWriter(fps=self.output_fps, bitrate=5000)
            anim.save(output_str, writer=writer)
        except Exception as e:
            print(f"FFmpeg not available ({e}), saving as GIF instead...")
            gif_path = output_path.with_suffix('.gif')
            writer = PillowWriter(fps=self.output_fps)
            anim.save(str(gif_path), writer=writer)
            print(f"Saved as GIF: {gif_path}")
        
        plt.close(fig)
    
    def export_with_audio(
        self,
        frames: List[np.ndarray],
        output_path: str,
        audio_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Export frames to video with audio track.
        
        Note: Requires ffmpeg to be installed.
        
        Args:
            frames: List of frame images
            output_path: Path for the output video
            audio_path: Path to the audio file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the created video file
        """
        import subprocess
        import tempfile
        
        # First export video without audio
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        self.export(frames, temp_video_path, progress_callback)
        
        # Combine with audio using ffmpeg
        output_path = Path(output_path)
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
        finally:
            Path(temp_video_path).unlink()
        
        return str(output_path)
    
    @staticmethod
    def resize_frames(
        frames: List[np.ndarray],
        target_size: Tuple[int, int],
        maintain_aspect: bool = True,
    ) -> List[np.ndarray]:
        """
        Resize all frames to a target size.
        
        Args:
            frames: List of frame images
            target_size: Target (width, height)
            maintain_aspect: Whether to maintain aspect ratio (adds padding)
            
        Returns:
            List of resized frames
        """
        if not HAS_OPENCV:
            print("Warning: resize_frames requires OpenCV. Returning original frames.")
            return frames
            
        resized_frames = []
        target_width, target_height = target_size
        
        for frame in frames:
            h, w = frame.shape[:2]
            
            if maintain_aspect:
                # Calculate scale to fit within target
                scale = min(target_width / w, target_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Add padding to center
                pad_left = (target_width - new_w) // 2
                pad_top = (target_height - new_h) // 2
                
                padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
                resized_frames.append(padded)
            else:
                resized = cv2.resize(
                    frame, (target_width, target_height),
                    interpolation=cv2.INTER_AREA
                )
                resized_frames.append(resized)
        
        return resized_frames
    
    @staticmethod
    def add_text_overlay(
        frames: List[np.ndarray],
        text: str,
        position: Tuple[int, int] = (20, 40),
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> List[np.ndarray]:
        """
        Add a text overlay to all frames.
        
        Args:
            frames: List of frame images
            text: Text to overlay
            position: (x, y) position for the text
            font_scale: Font scale
            color: Text color (RGB)
            thickness: Text thickness
            
        Returns:
            List of frames with text overlay
        """
        if not HAS_OPENCV:
            print("Warning: add_text_overlay requires OpenCV. Returning original frames.")
            return frames
            
        annotated = []
        for frame in frames:
            frame_copy = frame.copy()
            # Convert RGB color to BGR for OpenCV
            bgr_color = color[::-1]
            cv2.putText(
                frame_copy, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                bgr_color, thickness, cv2.LINE_AA
            )
            annotated.append(frame_copy)
        
        return annotated
    
    @staticmethod
    def concatenate_videos(
        video_paths: List[str],
        output_path: str,
        direction: str = "horizontal",
    ) -> str:
        """
        Concatenate multiple videos side by side or vertically.
        
        Note: Requires ffmpeg to be installed.
        
        Args:
            video_paths: List of paths to input videos
            output_path: Path for the output video
            direction: 'horizontal' or 'vertical'
            
        Returns:
            Path to the concatenated video
        """
        import subprocess
        
        n = len(video_paths)
        if n < 2:
            raise ValueError("Need at least 2 videos to concatenate")
        
        inputs = []
        for path in video_paths:
            inputs.extend(['-i', path])
        
        if direction == "horizontal":
            filter_complex = f"hstack=inputs={n}"
        else:
            filter_complex = f"vstack=inputs={n}"
        
        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-filter_complex', filter_complex,
            '-c:v', 'libx264',
            '-crf', '18',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
