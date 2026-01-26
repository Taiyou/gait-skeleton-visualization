"""
3D Visualizer Module
Creates 3D visualizations of skeleton motion.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Callable
from .skeleton_model import SkeletonModel


class Visualizer3D:
    """
    Creates 3D visualizations of skeletal motion data.
    """
    
    def __init__(
        self,
        skeleton: SkeletonModel,
        figsize: Tuple[int, int] = (10, 10),
        bg_color: str = "#FFFFFF",
        marker_size: int = 50,
        line_width: float = 2.0,
        view_elevation: float = 20,
        view_azimuth: float = -60,
        clean_mode: bool = False,
        zoom: float = 1.0,
    ):
        """
        Initialize the 3D visualizer.

        Args:
            skeleton: SkeletonModel defining the structure
            figsize: Figure size in inches
            bg_color: Background color
            marker_size: Size of marker points
            line_width: Width of bone lines
            view_elevation: Camera elevation angle
            view_azimuth: Camera azimuth angle
            clean_mode: If True, remove all axes, panes, grid, and labels
            zoom: Zoom factor (>1 = closer/larger body, <1 = farther/smaller)
        """
        self.skeleton = skeleton
        self.figsize = figsize
        self.bg_color = bg_color
        self.marker_size = marker_size
        self.line_width = line_width
        self.view_elevation = view_elevation
        self.view_azimuth = view_azimuth
        self.clean_mode = clean_mode
        self.zoom = zoom
        
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Axes3D] = None
        self.bounds: Optional[Dict[str, Tuple[float, float]]] = None
        
    def set_bounds(self, bounds: Dict[str, Tuple[float, float]]) -> "Visualizer3D":
        """
        Set the axis bounds for consistent scaling.
        
        Args:
            bounds: Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        self.bounds = bounds
        return self
    
    def _setup_figure(self) -> None:
        """Create and configure the matplotlib figure."""
        self.fig = plt.figure(figsize=self.figsize, facecolor=self.bg_color)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.bg_color)

        if self.clean_mode:
            self._apply_clean_mode()
        else:
            # Style the axes
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('gray')
            self.ax.yaxis.pane.set_edgecolor('gray')
            self.ax.zaxis.pane.set_edgecolor('gray')

            # Set labels
            self.ax.set_xlabel('X (mm)', color='black', fontsize=10)
            self.ax.set_ylabel('Y (mm)', color='black', fontsize=10)
            self.ax.set_zlabel('Z (mm)', color='black', fontsize=10)

            # Set tick colors
            self.ax.tick_params(colors='white', labelsize=8)

        # Set view angle
        self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)

    def _apply_clean_mode(self) -> None:
        """Remove all axes, panes, grid, and labels for clean rendering."""
        self.ax.set_axis_off()
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('none')
        self.ax.yaxis.pane.set_edgecolor('none')
        self.ax.zaxis.pane.set_edgecolor('none')
        self.ax.grid(False)
        
    def _apply_bounds(self) -> None:
        """Apply axis bounds with equal aspect ratio."""
        if self.bounds is None or self.ax is None:
            return
        
        # Calculate ranges
        x_range = self.bounds['x'][1] - self.bounds['x'][0]
        y_range = self.bounds['y'][1] - self.bounds['y'][0]
        z_range = self.bounds['z'][1] - self.bounds['z'][0]
        
        # Find max range for equal aspect ratio
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (self.bounds['x'][0] + self.bounds['x'][1]) / 2
        y_center = (self.bounds['y'][0] + self.bounds['y'][1]) / 2
        z_center = (self.bounds['z'][0] + self.bounds['z'][1]) / 2
        
        # Set equal aspect ratio bounds (zoom shrinks the view cube)
        margin = max_range * 0.05
        half_range = (max_range / 2 + margin) / self.zoom
        
        self.ax.set_xlim(x_center - half_range, x_center + half_range)
        self.ax.set_ylim(y_center - half_range, y_center + half_range)
        self.ax.set_zlim(z_center - half_range, z_center + half_range)
        
    def render_frame(
        self,
        marker_positions: Dict[str, np.ndarray],
        frame_number: Optional[int] = None,
        time_seconds: Optional[float] = None,
        show_labels: bool = False,
        ax: Optional[Axes3D] = None,
    ) -> plt.Figure:
        """
        Render a single frame.
        
        Args:
            marker_positions: Dictionary of marker names to 3D positions
            frame_number: Optional frame number to display
            time_seconds: Optional time in seconds to display
            show_labels: Whether to show marker labels
            ax: Optional existing axis to draw on
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            self._setup_figure()
            ax = self.ax
        else:
            ax.cla()
            self.ax = ax
        
        # Plot markers
        for marker, position in marker_positions.items():
            if position is not None:
                color = self.skeleton.get_marker_color(marker)
                ax.scatter(
                    position[0], position[1], position[2],
                    c=color, s=self.marker_size, marker='o',
                    edgecolors='white', linewidths=0.5
                )
                
                if show_labels:
                    ax.text(
                        position[0], position[1], position[2],
                        f"  {marker}", color='black', fontsize=8
                    )
        
        # Plot bones (connections)
        valid_connections = self.skeleton.get_valid_connections(marker_positions)
        for marker1, marker2 in valid_connections:
            pos1 = marker_positions[marker1]
            pos2 = marker_positions[marker2]
            color = self.skeleton.get_connection_color(marker1, marker2)
            
            ax.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                [pos1[2], pos2[2]],
                c=color, linewidth=self.line_width
            )
        
        # Apply bounds
        self._apply_bounds()

        # Re-apply clean mode after bounds (set_xlim etc. can re-enable axes)
        if self.clean_mode:
            self._apply_clean_mode()

        # Add title with frame info
        title_parts = []
        if frame_number is not None:
            title_parts.append(f"Frame: {frame_number}")
        if time_seconds is not None:
            title_parts.append(f"Time: {time_seconds:.3f}s")
        if title_parts:
            ax.set_title(" | ".join(title_parts), color='black', fontsize=12)

        return self.fig
    
    def create_animation_frames(
        self,
        all_positions: List[Dict[str, np.ndarray]],
        frame_rate: float = 100.0,
        show_labels: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """
        Create all animation frames as numpy arrays.
        
        Args:
            all_positions: List of marker positions for each frame
            frame_rate: Frame rate for time display
            show_labels: Whether to show marker labels
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of frame images as numpy arrays (RGB)
        """
        frames = []
        num_frames = len(all_positions)
        
        # Setup figure once
        self._setup_figure()
        
        for i, positions in enumerate(all_positions):
            self.ax.cla()

            # Re-apply axis settings after clear
            if self.clean_mode:
                self._apply_clean_mode()
            else:
                self.ax.set_xlabel('X (mm)', color='black', fontsize=10)
                self.ax.set_ylabel('Y (mm)', color='black', fontsize=10)
                self.ax.set_zlabel('Z (mm)', color='black', fontsize=10)
                self.ax.tick_params(colors='white', labelsize=8)
            self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
            
            # Render the frame
            time_seconds = i / frame_rate
            self.render_frame(
                positions,
                frame_number=i,
                time_seconds=time_seconds,
                show_labels=show_labels,
                ax=self.ax,
            )
            
            # Convert to image array
            self.fig.canvas.draw()
            # Use buffer_rgba for compatibility with all backends
            buf = self.fig.canvas.buffer_rgba()
            image = np.asarray(buf, dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB and make a copy (important!)
            image = image[:, :, :3].copy()
            frames.append(image)
            
            if progress_callback:
                progress_callback(i + 1, num_frames)
        
        plt.close(self.fig)
        return frames
    
    def show_frame(
        self,
        marker_positions: Dict[str, np.ndarray],
        frame_number: Optional[int] = None,
        show_labels: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """Render a single frame and optionally save to file."""
        fig = self.render_frame(marker_positions, frame_number, show_labels=show_labels)
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
        return fig
    
    def close(self) -> None:
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
