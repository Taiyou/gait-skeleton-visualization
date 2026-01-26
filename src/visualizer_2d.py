"""
2D Visualizer Module
Creates 2D projections of skeleton motion.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable, Literal
from .skeleton_model import SkeletonModel


ViewType = Literal["sagittal", "frontal", "transverse", "custom"]


class Visualizer2D:
    """
    Creates 2D visualizations of skeletal motion data.
    
    Supports three standard anatomical views:
    - Sagittal (side view): X-Z plane
    - Frontal (front view): Y-Z plane  
    - Transverse (top view): X-Y plane
    """
    
    # Axis mappings for different views
    VIEW_AXES = {
        "sagittal": (0, 2),     # X, Z - side view
        "frontal": (1, 2),      # Y, Z - front view
        "transverse": (0, 1),   # X, Y - top view
    }
    
    VIEW_LABELS = {
        "sagittal": ("X (Anterior-Posterior)", "Z (Vertical)"),
        "frontal": ("Y (Medio-Lateral)", "Z (Vertical)"),
        "transverse": ("X (Anterior-Posterior)", "Y (Medio-Lateral)"),
    }
    
    def __init__(
        self,
        skeleton: SkeletonModel,
        view: ViewType = "sagittal",
        figsize: Tuple[int, int] = (10, 8),
        bg_color: str = "#FFFFFF",
        marker_size: int = 80,
        line_width: float = 2.5,
        custom_axes: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the 2D visualizer.
        
        Args:
            skeleton: SkeletonModel defining the structure
            view: View type ('sagittal', 'frontal', 'transverse', 'custom')
            figsize: Figure size in inches
            bg_color: Background color
            marker_size: Size of marker points
            line_width: Width of bone lines
            custom_axes: Custom axis indices for 'custom' view (e.g., (0, 2) for X-Z)
        """
        self.skeleton = skeleton
        self.view = view
        self.figsize = figsize
        self.bg_color = bg_color
        self.marker_size = marker_size
        self.line_width = line_width
        
        # Set axis indices
        if view == "custom":
            if custom_axes is None:
                raise ValueError("custom_axes required when view='custom'")
            self.axis_indices = custom_axes
            self.axis_labels = ("Axis 1", "Axis 2")
        else:
            self.axis_indices = self.VIEW_AXES[view]
            self.axis_labels = self.VIEW_LABELS[view]
        
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.bounds: Optional[Dict[str, Tuple[float, float]]] = None
        
    def set_bounds(self, bounds: Dict[str, Tuple[float, float]]) -> "Visualizer2D":
        """
        Set the axis bounds for consistent scaling.
        
        Args:
            bounds: Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        self.bounds = bounds
        return self
    
    def _setup_figure(self) -> None:
        """Create and configure the matplotlib figure."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize, facecolor=self.bg_color)
        self.ax.set_facecolor(self.bg_color)
        
        # Style the axes
        self.ax.spines['bottom'].set_color('gray')
        self.ax.spines['top'].set_color('gray')
        self.ax.spines['left'].set_color('gray')
        self.ax.spines['right'].set_color('gray')
        
        # Set labels
        self.ax.set_xlabel(self.axis_labels[0], color='black', fontsize=10)
        self.ax.set_ylabel(self.axis_labels[1], color='black', fontsize=10)
        
        # Set tick colors
        self.ax.tick_params(colors='white', labelsize=8)
        
        # Equal aspect ratio
        self.ax.set_aspect('equal', adjustable='box')
        
    def _apply_bounds(self, current_positions: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Apply axis bounds with equal aspect ratio."""
        if self.ax is None:
            return

        # Get the bounds for the two axes we're using
        bounds_keys = ['x', 'y', 'z']
        axis1_key = bounds_keys[self.axis_indices[0]]
        axis2_key = bounds_keys[self.axis_indices[1]]

        if self.bounds is not None:
            axis1_min, axis1_max = self.bounds[axis1_key]
            axis2_min, axis2_max = self.bounds[axis2_key]
        elif current_positions is not None:
            # Calculate bounds from current frame
            coords1 = [p[self.axis_indices[0]] for p in current_positions.values() if p is not None]
            coords2 = [p[self.axis_indices[1]] for p in current_positions.values() if p is not None]
            if coords1 and coords2:
                axis1_min, axis1_max = min(coords1), max(coords1)
                axis2_min, axis2_max = min(coords2), max(coords2)
            else:
                return
        else:
            return

        axis1_range = axis1_max - axis1_min
        axis2_range = axis2_max - axis2_min

        # Find max range for equal aspect ratio
        max_range = max(axis1_range, axis2_range)

        # Ensure minimum range for visibility (adaptive to data scale)
        data_scale = max(axis1_range, axis2_range)
        min_range = data_scale * 1.0 if data_scale > 0 else 800
        max_range = max(max_range, min_range)

        # Calculate centers
        axis1_center = (axis1_min + axis1_max) / 2
        axis2_center = (axis2_min + axis2_max) / 2

        # Set equal aspect ratio bounds with smaller margin
        margin = max_range * 0.15
        half_range = max_range / 2 + margin

        self.ax.set_xlim(axis1_center - half_range, axis1_center + half_range)
        self.ax.set_ylim(axis2_center - half_range, axis2_center + half_range)
    
    def _get_2d_position(self, position_3d: np.ndarray) -> Tuple[float, float]:
        """Project a 3D position to 2D based on the view."""
        return (
            position_3d[self.axis_indices[0]],
            position_3d[self.axis_indices[1]]
        )
    
    def render_frame(
        self,
        marker_positions: Dict[str, np.ndarray],
        frame_number: Optional[int] = None,
        time_seconds: Optional[float] = None,
        show_labels: bool = False,
        ax: Optional[plt.Axes] = None,
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
                x, y = self._get_2d_position(position)
                color = self.skeleton.get_marker_color(marker)
                ax.scatter(
                    x, y,
                    c=color, s=self.marker_size, marker='o',
                    edgecolors='white', linewidths=0.5, zorder=3
                )
                
                if show_labels:
                    ax.text(
                        x, y, f"  {marker}",
                        color='black', fontsize=8, zorder=4
                    )
        
        # Plot bones (connections)
        valid_connections = self.skeleton.get_valid_connections(marker_positions)
        for marker1, marker2 in valid_connections:
            pos1 = marker_positions[marker1]
            pos2 = marker_positions[marker2]
            x1, y1 = self._get_2d_position(pos1)
            x2, y2 = self._get_2d_position(pos2)
            color = self.skeleton.get_connection_color(marker1, marker2)
            
            ax.plot(
                [x1, x2], [y1, y2],
                c=color, linewidth=self.line_width, zorder=2
            )
        
        # Apply bounds
        self._apply_bounds(marker_positions)
        
        # Add title with frame info
        view_name = self.view.capitalize() if self.view != "custom" else "Custom View"
        title_parts = [view_name]
        if frame_number is not None:
            title_parts.append(f"Frame: {frame_number}")
        if time_seconds is not None:
            title_parts.append(f"Time: {time_seconds:.3f}s")
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
            self.ax.set_facecolor(self.bg_color)
            self.ax.set_xlabel(self.axis_labels[0], color='black', fontsize=10)
            self.ax.set_ylabel(self.axis_labels[1], color='black', fontsize=10)
            self.ax.tick_params(colors='white', labelsize=8)
            self.ax.set_aspect('equal', adjustable='box')
            
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


class MultiViewVisualizer:
    """
    Creates multi-panel visualizations with multiple views simultaneously.
    """
    
    def __init__(
        self,
        skeleton: SkeletonModel,
        views: List[ViewType] = ["sagittal", "frontal", "transverse"],
        figsize: Tuple[int, int] = (15, 5),
        bg_color: str = "#FFFFFF",
        marker_size: int = 60,
        line_width: float = 2.0,
    ):
        """
        Initialize the multi-view visualizer.
        
        Args:
            skeleton: SkeletonModel defining the structure
            views: List of views to display
            figsize: Figure size in inches
            bg_color: Background color
            marker_size: Size of marker points
            line_width: Width of bone lines
        """
        self.skeleton = skeleton
        self.views = views
        self.figsize = figsize
        self.bg_color = bg_color
        self.marker_size = marker_size
        self.line_width = line_width
        
        self.fig: Optional[plt.Figure] = None
        self.axes: List[plt.Axes] = []
        self.bounds: Optional[Dict[str, Tuple[float, float]]] = None
        
        # Create individual visualizers
        self.visualizers = [
            Visualizer2D(
                skeleton, view=view,
                marker_size=marker_size,
                line_width=line_width
            )
            for view in views
        ]
    
    def set_bounds(self, bounds: Dict[str, Tuple[float, float]]) -> "MultiViewVisualizer":
        """Set bounds for all visualizers."""
        self.bounds = bounds
        for vis in self.visualizers:
            vis.set_bounds(bounds)
        return self
    
    def _setup_figure(self) -> None:
        """Create and configure the matplotlib figure."""
        n_views = len(self.views)
        self.fig, self.axes = plt.subplots(
            1, n_views, figsize=self.figsize, facecolor=self.bg_color
        )
        if n_views == 1:
            self.axes = [self.axes]
    
    def render_frame(
        self,
        marker_positions: Dict[str, np.ndarray],
        frame_number: Optional[int] = None,
        time_seconds: Optional[float] = None,
        show_labels: bool = False,
    ) -> plt.Figure:
        """Render a single frame in all views."""
        self._setup_figure()
        
        for i, (vis, ax) in enumerate(zip(self.visualizers, self.axes)):
            ax.set_facecolor(self.bg_color)
            vis.render_frame(
                marker_positions,
                frame_number=frame_number,
                time_seconds=time_seconds,
                show_labels=show_labels,
                ax=ax,
            )
        
        self.fig.tight_layout()
        return self.fig
    
    def create_animation_frames(
        self,
        all_positions: List[Dict[str, np.ndarray]],
        frame_rate: float = 100.0,
        show_labels: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """Create all animation frames with multiple views."""
        frames = []
        num_frames = len(all_positions)
        
        self._setup_figure()
        
        for i, positions in enumerate(all_positions):
            for ax in self.axes:
                ax.cla()
                ax.set_facecolor(self.bg_color)
            
            time_seconds = i / frame_rate
            for vis, ax in zip(self.visualizers, self.axes):
                vis.render_frame(
                    positions,
                    frame_number=i,
                    time_seconds=time_seconds,
                    show_labels=show_labels,
                    ax=ax,
                )
            
            self.fig.tight_layout()
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
    
    def close(self) -> None:
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = []
