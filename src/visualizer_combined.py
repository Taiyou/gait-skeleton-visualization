"""
Combined Visualizer Module
Creates side-by-side 3D skeleton + joint angle time-series visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Callable
from .skeleton_model import SkeletonModel
from .phase_dynamics import compute_joint_angles, DEFAULT_JOINT_DEFINITIONS


# Display configuration for joint angle plots
# Maps joint labels to (display_name, side)
JOINT_DISPLAY = {
    "knee_right": ("Knee", "right"),
    "knee_left": ("Knee", "left"),
    "hip_right": ("Hip", "right"),
    "hip_left": ("Hip", "left"),
    "elbow_right": ("Elbow", "right"),
    "elbow_left": ("Elbow", "left"),
}

# Group joints by body region for plotting
JOINT_GROUPS = [
    ("Knee Angle", ["knee_right", "knee_left"]),
    ("Hip Angle", ["hip_right", "hip_left"]),
    ("Elbow Angle", ["elbow_right", "elbow_left"]),
]

COLOR_RIGHT = "#E74C3C"
COLOR_LEFT = "#2980B9"


class CombinedVisualizer:
    """
    Creates side-by-side 3D skeleton + joint angle time-series.
    Left panel: 3D skeleton animation.
    Right panels: Joint angle time-series with current-time indicator.
    """

    def __init__(
        self,
        skeleton: SkeletonModel,
        figsize: Tuple[int, int] = (16, 8),
        bg_color: str = "#FFFFFF",
        marker_size: int = 50,
        line_width: float = 2.0,
        view_elevation: float = 20,
        view_azimuth: float = -60,
        clean_mode: bool = False,
        zoom: float = 1.0,
        joint_definitions: Optional[List[Tuple[str, str, str, str]]] = None,
    ):
        self.skeleton = skeleton
        self.figsize = figsize
        self.bg_color = bg_color
        self.marker_size = marker_size
        self.line_width = line_width
        self.view_elevation = view_elevation
        self.view_azimuth = view_azimuth
        self.clean_mode = clean_mode
        self.zoom = zoom
        self.joint_definitions = joint_definitions or DEFAULT_JOINT_DEFINITIONS

        self.fig: Optional[plt.Figure] = None
        self.bounds: Optional[Dict[str, Tuple[float, float]]] = None

    def set_bounds(self, bounds: Dict[str, Tuple[float, float]]) -> "CombinedVisualizer":
        self.bounds = bounds
        return self

    def _apply_3d_bounds(self, ax) -> None:
        if self.bounds is None:
            return
        x_range = self.bounds['x'][1] - self.bounds['x'][0]
        y_range = self.bounds['y'][1] - self.bounds['y'][0]
        z_range = self.bounds['z'][1] - self.bounds['z'][0]
        max_range = max(x_range, y_range, z_range)
        x_center = (self.bounds['x'][0] + self.bounds['x'][1]) / 2
        y_center = (self.bounds['y'][0] + self.bounds['y'][1]) / 2
        z_center = (self.bounds['z'][0] + self.bounds['z'][1]) / 2
        margin = max_range * 0.05
        half_range = (max_range / 2 + margin) / self.zoom
        ax.set_xlim(x_center - half_range, x_center + half_range)
        ax.set_ylim(y_center - half_range, y_center + half_range)
        ax.set_zlim(z_center - half_range, z_center + half_range)

    def _render_skeleton(self, ax, positions: Dict[str, np.ndarray]) -> None:
        """Render the 3D skeleton on the given axes."""
        for marker, position in positions.items():
            if position is not None:
                color = self.skeleton.get_marker_color(marker)
                ax.scatter(
                    position[0], position[1], position[2],
                    c=color, s=self.marker_size, marker='o',
                    edgecolors='white', linewidths=0.5,
                )

        valid_connections = self.skeleton.get_valid_connections(positions)
        for m1, m2 in valid_connections:
            p1 = positions[m1]
            p2 = positions[m2]
            color = self.skeleton.get_connection_color(m1, m2)
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                c=color, linewidth=self.line_width,
            )

        self._apply_3d_bounds(ax)

    def create_animation_frames(
        self,
        all_positions: List[Dict[str, np.ndarray]],
        frame_rate: float = 100.0,
        show_labels: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """Create all animation frames with skeleton + angle time-series."""
        num_frames = len(all_positions)
        time_array = np.arange(num_frames) / frame_rate

        # Pre-compute all joint angles
        angles = compute_joint_angles(all_positions, self.joint_definitions)

        # Pre-compute y-axis ranges for each joint group
        group_ylims = []
        for _, joint_keys in JOINT_GROUPS:
            all_vals = np.concatenate([angles[k] for k in joint_keys if k in angles])
            valid = all_vals[~np.isnan(all_vals)]
            if len(valid) > 0:
                ymin, ymax = valid.min(), valid.max()
                margin = (ymax - ymin) * 0.1 if ymax > ymin else 5.0
                group_ylims.append((ymin - margin, ymax + margin))
            else:
                group_ylims.append((0, 180))

        frames = []
        n_groups = len(JOINT_GROUPS)

        self.fig = plt.figure(figsize=self.figsize, facecolor=self.bg_color)

        for i in range(num_frames):
            self.fig.clf()
            gs = GridSpec(n_groups, 2, width_ratios=[1.2, 1], figure=self.fig)

            # Left panel: 3D skeleton
            ax3d = self.fig.add_subplot(gs[:, 0], projection='3d')
            ax3d.set_facecolor(self.bg_color)
            ax3d.view_init(elev=self.view_elevation, azim=self.view_azimuth)

            if self.clean_mode:
                ax3d.set_axis_off()
                ax3d.xaxis.pane.fill = False
                ax3d.yaxis.pane.fill = False
                ax3d.zaxis.pane.fill = False
                ax3d.xaxis.pane.set_edgecolor('none')
                ax3d.yaxis.pane.set_edgecolor('none')
                ax3d.zaxis.pane.set_edgecolor('none')
                ax3d.grid(False)
            else:
                ax3d.xaxis.pane.fill = False
                ax3d.yaxis.pane.fill = False
                ax3d.zaxis.pane.fill = False
                ax3d.xaxis.pane.set_edgecolor('gray')
                ax3d.yaxis.pane.set_edgecolor('gray')
                ax3d.zaxis.pane.set_edgecolor('gray')
                ax3d.tick_params(colors='white', labelsize=8)

            self._render_skeleton(ax3d, all_positions[i])

            # Re-apply clean mode after bounds (set_xlim etc. can re-enable axes)
            if self.clean_mode:
                ax3d.set_axis_off()
                ax3d.xaxis.pane.fill = False
                ax3d.yaxis.pane.fill = False
                ax3d.zaxis.pane.fill = False
                ax3d.xaxis.pane.set_edgecolor('none')
                ax3d.yaxis.pane.set_edgecolor('none')
                ax3d.zaxis.pane.set_edgecolor('none')
                ax3d.grid(False)

            # Right panels: joint angle time-series
            current_time = time_array[i]
            for g_idx, (group_name, joint_keys) in enumerate(JOINT_GROUPS):
                ax_ts = self.fig.add_subplot(gs[g_idx, 1])
                ax_ts.set_facecolor(self.bg_color)

                for jkey in joint_keys:
                    if jkey not in angles:
                        continue
                    display = JOINT_DISPLAY.get(jkey)
                    if display is None:
                        continue
                    _, side = display
                    color = COLOR_RIGHT if side == "right" else COLOR_LEFT
                    label = f"Right" if side == "right" else "Left"
                    ax_ts.plot(time_array, angles[jkey], color=color, linewidth=1.2, label=label)

                # Current time indicator
                ax_ts.axvline(x=current_time, color='gray', linestyle='--', linewidth=0.8)

                ax_ts.set_ylabel("Angle (deg)", fontsize=8)
                ax_ts.set_title(group_name, fontsize=9, fontweight='bold')
                ax_ts.set_xlim(time_array[0], time_array[-1])
                ax_ts.set_ylim(group_ylims[g_idx])
                ax_ts.tick_params(labelsize=7)

                if g_idx == n_groups - 1:
                    ax_ts.set_xlabel("Time (s)", fontsize=8)
                else:
                    ax_ts.set_xticklabels([])

                if g_idx == 0:
                    ax_ts.legend(fontsize=7, loc='upper right')

            self.fig.tight_layout()
            self.fig.canvas.draw()
            buf = self.fig.canvas.buffer_rgba()
            image = np.asarray(buf, dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3].copy()
            frames.append(image)

            if progress_callback:
                progress_callback(i + 1, num_frames)

        plt.close(self.fig)
        return frames

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
