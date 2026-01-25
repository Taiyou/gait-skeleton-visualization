#!/usr/bin/env python3
"""
Generate realistic gait data for visualization.

This script creates synthetic walking motion data based on
biomechanical gait patterns with proper phase relationships
and joint kinematics.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class GaitCycleGenerator:
    """
    Generates realistic gait cycle data based on biomechanics.

    A normal gait cycle consists of:
    - Stance phase (~60% of cycle): foot is on the ground
    - Swing phase (~40% of cycle): foot is in the air

    Key events:
    - Initial contact (0%): heel strike
    - Loading response (0-10%): weight acceptance
    - Mid-stance (10-30%): single leg support
    - Terminal stance (30-50%): heel off
    - Pre-swing (50-60%): toe off
    - Initial swing (60-73%): acceleration
    - Mid-swing (73-87%): limb advancement
    - Terminal swing (87-100%): deceleration
    """

    def __init__(
        self,
        height_mm: float = 1700,
        step_length_mm: float = 600,
        cadence_steps_per_min: float = 110,
    ):
        """
        Initialize the gait generator.

        Args:
            height_mm: Body height in mm
            step_length_mm: Step length in mm
            cadence_steps_per_min: Cadence in steps/minute
        """
        self.height = height_mm
        self.step_length = step_length_mm
        self.cadence = cadence_steps_per_min

        # Body segment proportions (as fraction of height)
        # Based on anthropometric data
        self.head_height = height_mm * 0.93  # Top of head
        self.shoulder_height = height_mm * 0.82
        self.elbow_height = height_mm * 0.63
        self.hip_height = height_mm * 0.53
        self.knee_height = height_mm * 0.29
        self.ankle_height = height_mm * 0.05

        # Widths
        self.shoulder_width = height_mm * 0.26
        self.hip_width = height_mm * 0.18
        self.elbow_offset = height_mm * 0.05  # Lateral offset from shoulder

        # Calculate gait timing
        self.step_time = 60.0 / cadence_steps_per_min  # seconds per step
        self.gait_cycle_time = self.step_time * 2  # full cycle = 2 steps

    def _smooth_step(self, x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
        """Smooth step function (Hermite interpolation)."""
        t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
        return t * t * (3 - 2 * t)

    def _hip_flexion_angle(self, phase: np.ndarray) -> np.ndarray:
        """
        Generate hip flexion/extension angle pattern.

        Based on typical gait kinematics:
        - Maximum extension (~-20°) at terminal stance (~50%)
        - Maximum flexion (~+35°) at terminal swing (~85%)

        Positive = flexion (leg forward), Negative = extension (leg backward)
        """
        # Two harmonics capture the hip angle pattern well
        # Amplified for better visualization
        # Negated to make forward walking (leg swings forward with positive X)
        angle = -(
            30 * np.sin(2 * np.pi * phase - np.pi/3) +  # Main component (amplified)
            8 * np.sin(4 * np.pi * phase + np.pi/4)      # Secondary
        )
        return angle

    def _knee_flexion_angle(self, phase: np.ndarray) -> np.ndarray:
        """
        Generate knee flexion angle pattern.

        Knee has characteristic double-peak pattern:
        - First peak (~20°) at loading response (~10%)
        - Valley (~5°) at mid-stance (~30%)
        - Second peak (~70°) at initial swing (~70%)
        - Extension at terminal swing (~90%)
        """
        # Loading response peak (amplified)
        loading_peak = 20 * np.exp(-((phase - 0.08) ** 2) / 0.003)

        # Swing phase flexion (major peak - amplified)
        swing_peak = 70 * np.exp(-((phase - 0.70) ** 2) / 0.018)

        # Base extension pattern
        base = 8 * (1 - np.cos(2 * np.pi * phase))

        return loading_peak + swing_peak + base

    def _ankle_angle(self, phase: np.ndarray) -> np.ndarray:
        """
        Generate ankle dorsiflexion/plantarflexion pattern.

        Pattern:
        - Plantarflexion at initial contact
        - Dorsiflexion through stance (foot flat, heel off)
        - Rapid plantarflexion at push-off
        - Dorsiflexion in swing for foot clearance
        """
        # Initial contact plantarflexion
        ic_plantar = -10 * np.exp(-((phase - 0.0) ** 2) / 0.005)

        # Stance dorsiflexion
        stance_dorsi = 10 * self._smooth_step(phase, 0.1, 0.35)
        stance_dorsi *= (1 - self._smooth_step(phase, 0.45, 0.60))

        # Push-off plantarflexion
        pushoff = -20 * np.exp(-((phase - 0.55) ** 2) / 0.008)

        # Swing dorsiflexion
        swing_dorsi = 5 * self._smooth_step(phase, 0.60, 0.75)
        swing_dorsi *= (1 - self._smooth_step(phase, 0.90, 1.0))

        return ic_plantar + stance_dorsi + pushoff + swing_dorsi

    def _vertical_oscillation(self, phase: np.ndarray) -> np.ndarray:
        """
        Body center of mass vertical oscillation.

        CoM is highest at mid-stance (single support)
        CoM is lowest at double support (heel strike)
        """
        # Two peaks per gait cycle (one for each step) - amplified
        return 45 * np.sin(4 * np.pi * phase)

    def _lateral_sway(self, phase: np.ndarray) -> np.ndarray:
        """
        Lateral sway of the body.

        Body shifts laterally over the stance leg.
        """
        return 25 * np.sin(2 * np.pi * phase)

    def _pelvic_rotation(self, phase: np.ndarray) -> np.ndarray:
        """Pelvic transverse rotation in degrees."""
        return 8 * np.sin(2 * np.pi * phase)

    def _pelvic_tilt(self, phase: np.ndarray) -> np.ndarray:
        """Pelvic sagittal tilt (list) in degrees."""
        return 5 * np.sin(2 * np.pi * phase)

    def _shoulder_rotation(self, phase: np.ndarray) -> np.ndarray:
        """
        Shoulder counter-rotation to pelvis.
        Out of phase with pelvis for balance.
        """
        return -12 * np.sin(2 * np.pi * phase + np.pi)

    def _arm_swing_angle(self, phase: np.ndarray) -> np.ndarray:
        """
        Arm swing angle pattern.
        Arms swing opposite to legs (contralateral).
        """
        # Amplified for better visualization
        # Negated to match forward walking direction
        return -35 * np.sin(2 * np.pi * phase)

    def _foot_clearance(self, phase: np.ndarray) -> np.ndarray:
        """
        Foot vertical clearance during swing.
        Foot lifts during swing phase for clearance.
        """
        # Swing phase is 60-100% of gait cycle
        swing_mask = (phase > 0.55) & (phase < 1.0)
        clearance = np.zeros_like(phase)

        # Peak clearance at mid-swing (~75%) - amplified for visibility
        swing_phase_normalized = np.clip((phase - 0.55) / 0.45, 0, 1)
        clearance = 120 * np.sin(np.pi * swing_phase_normalized)
        clearance = np.where(swing_mask, clearance, 0)

        return clearance

    def generate(
        self,
        num_frames: int = 300,
        frame_rate: float = 100.0,
        num_cycles: int = 3,
    ) -> pd.DataFrame:
        """
        Generate realistic gait data.

        Args:
            num_frames: Number of frames to generate
            frame_rate: Frame rate in Hz
            num_cycles: Number of gait cycles

        Returns:
            DataFrame with marker positions
        """
        t = np.arange(num_frames) / frame_rate
        duration = num_frames / frame_rate

        # Phase (0-1 for each gait cycle, repeated)
        gait_freq = num_cycles / duration
        phase = (gait_freq * t) % 1.0

        # Right leg is 50% out of phase
        phase_L = phase
        phase_R = (phase + 0.5) % 1.0

        # Get joint angles
        hip_flex_L = self._hip_flexion_angle(phase_L)
        hip_flex_R = self._hip_flexion_angle(phase_R)

        knee_flex_L = self._knee_flexion_angle(phase_L)
        knee_flex_R = self._knee_flexion_angle(phase_R)

        ankle_L = self._ankle_angle(phase_L)
        ankle_R = self._ankle_angle(phase_R)

        # Body oscillations
        vert_osc = self._vertical_oscillation(phase)
        lat_sway = self._lateral_sway(phase)

        # Pelvis motion
        pelv_rot = self._pelvic_rotation(phase)
        pelv_tilt = self._pelvic_tilt(phase)

        # Shoulder motion (counter-rotation)
        sho_rot = self._shoulder_rotation(phase)

        # Arm swing
        arm_swing = self._arm_swing_angle(phase)

        # Foot clearance
        foot_clear_L = self._foot_clearance(phase_L)
        foot_clear_R = self._foot_clearance(phase_R)

        # Convert angles to radians for position calculation
        hip_flex_L_rad = np.radians(hip_flex_L)
        hip_flex_R_rad = np.radians(hip_flex_R)
        knee_flex_L_rad = np.radians(knee_flex_L)
        knee_flex_R_rad = np.radians(knee_flex_R)
        arm_swing_rad = np.radians(arm_swing)
        pelv_rot_rad = np.radians(pelv_rot)
        sho_rot_rad = np.radians(sho_rot)

        # Segment lengths
        thigh_len = self.hip_height - self.knee_height
        shank_len = self.knee_height - self.ankle_height
        upper_arm_len = self.shoulder_height - self.elbow_height

        markers = {}

        # ============== HEAD ==============
        markers['HEAD_X'] = 20 * np.sin(4 * np.pi * phase)  # Slight bob
        markers['HEAD_Y'] = lat_sway * 0.8  # Follows body sway
        markers['HEAD_Z'] = self.head_height + vert_osc * 0.9

        # ============== SHOULDERS ==============
        sho_x_offset = 30 * np.sin(sho_rot_rad)

        markers['LSHO_X'] = sho_x_offset
        markers['LSHO_Y'] = self.shoulder_width / 2
        markers['LSHO_Z'] = self.shoulder_height + vert_osc * 0.85

        markers['RSHO_X'] = -sho_x_offset
        markers['RSHO_Y'] = -self.shoulder_width / 2
        markers['RSHO_Z'] = self.shoulder_height + vert_osc * 0.85

        # ============== ELBOWS (arm swing) ==============
        # Left arm swings with right leg, right arm with left leg
        markers['LELB_X'] = upper_arm_len * np.sin(-arm_swing_rad) + sho_x_offset
        markers['LELB_Y'] = self.shoulder_width / 2 + self.elbow_offset
        markers['LELB_Z'] = (self.shoulder_height - upper_arm_len * np.cos(arm_swing_rad) +
                           vert_osc * 0.7)

        markers['RELB_X'] = upper_arm_len * np.sin(arm_swing_rad) - sho_x_offset
        markers['RELB_Y'] = -self.shoulder_width / 2 - self.elbow_offset
        markers['RELB_Z'] = (self.shoulder_height - upper_arm_len * np.cos(-arm_swing_rad) +
                           vert_osc * 0.7)

        # ============== HIPS ==============
        hip_x_offset = 15 * np.sin(pelv_rot_rad)
        hip_z_tilt = 10 * np.sin(2 * np.pi * phase)  # Pelvic drop

        markers['LHIP_X'] = hip_x_offset
        markers['LHIP_Y'] = self.hip_width / 2
        markers['LHIP_Z'] = self.hip_height + vert_osc + hip_z_tilt

        markers['RHIP_X'] = -hip_x_offset
        markers['RHIP_Y'] = -self.hip_width / 2
        markers['RHIP_Z'] = self.hip_height + vert_osc - hip_z_tilt

        # ============== KNEES ==============
        # Knee position based on hip flexion and knee flexion
        # Simplified 2-link model in sagittal plane

        # Left knee
        knee_x_L = (hip_x_offset +
                   thigh_len * np.sin(hip_flex_L_rad))
        knee_z_L = (self.hip_height + vert_osc + hip_z_tilt -
                   thigh_len * np.cos(hip_flex_L_rad))

        markers['LKNE_X'] = knee_x_L
        markers['LKNE_Y'] = self.hip_width / 2 - 10
        markers['LKNE_Z'] = knee_z_L

        # Right knee
        knee_x_R = (-hip_x_offset +
                   thigh_len * np.sin(hip_flex_R_rad))
        knee_z_R = (self.hip_height + vert_osc - hip_z_tilt -
                   thigh_len * np.cos(hip_flex_R_rad))

        markers['RKNE_X'] = knee_x_R
        markers['RKNE_Y'] = -self.hip_width / 2 + 10
        markers['RKNE_Z'] = knee_z_R

        # ============== ANKLES ==============
        # Ankle position based on knee position and knee flexion
        # Knee flexion angle is relative to thigh

        # Total angle of shank from vertical
        shank_angle_L = hip_flex_L_rad - knee_flex_L_rad
        shank_angle_R = hip_flex_R_rad - knee_flex_R_rad

        # Left ankle
        ankle_x_L = knee_x_L + shank_len * np.sin(shank_angle_L)
        ankle_z_L = knee_z_L - shank_len * np.cos(shank_angle_L)
        # Add foot clearance during swing
        ankle_z_L = ankle_z_L + foot_clear_L
        # Ensure ankle doesn't go below ground
        ankle_z_L = np.maximum(ankle_z_L, self.ankle_height)

        markers['LANK_X'] = ankle_x_L
        markers['LANK_Y'] = self.hip_width / 2 - 20
        markers['LANK_Z'] = ankle_z_L

        # Right ankle
        ankle_x_R = knee_x_R + shank_len * np.sin(shank_angle_R)
        ankle_z_R = knee_z_R - shank_len * np.cos(shank_angle_R)
        # Add foot clearance during swing
        ankle_z_R = ankle_z_R + foot_clear_R
        # Ensure ankle doesn't go below ground
        ankle_z_R = np.maximum(ankle_z_R, self.ankle_height)

        markers['RANK_X'] = ankle_x_R
        markers['RANK_Y'] = -self.hip_width / 2 + 20
        markers['RANK_Z'] = ankle_z_R

        # Create DataFrame
        df = pd.DataFrame(markers)
        df.insert(0, 'Frame', np.arange(num_frames))

        return df


def generate_all_samples():
    """Generate multiple sample data files."""
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Realistic Gait Data")
    print("=" * 60)

    # Normal walking speed
    print("\n1. Normal walking gait...")
    generator = GaitCycleGenerator(
        height_mm=1700,
        step_length_mm=650,
        cadence_steps_per_min=110,
    )
    df = generator.generate(num_frames=300, frame_rate=100.0, num_cycles=3)
    path = output_dir / "realistic_gait.csv"
    df.to_csv(path, index=False)
    print(f"   Saved: {path}")
    print(f"   Frames: {len(df)}, Duration: 3.0s, Cycles: 3")

    # Slow walking
    print("\n2. Slow walking gait...")
    generator_slow = GaitCycleGenerator(
        height_mm=1700,
        step_length_mm=500,
        cadence_steps_per_min=80,
    )
    df_slow = generator_slow.generate(num_frames=400, frame_rate=100.0, num_cycles=2)
    path_slow = output_dir / "slow_gait.csv"
    df_slow.to_csv(path_slow, index=False)
    print(f"   Saved: {path_slow}")
    print(f"   Frames: {len(df_slow)}, Duration: 4.0s, Cycles: 2")

    # Fast walking
    print("\n3. Fast walking gait...")
    generator_fast = GaitCycleGenerator(
        height_mm=1700,
        step_length_mm=750,
        cadence_steps_per_min=130,
    )
    df_fast = generator_fast.generate(num_frames=250, frame_rate=100.0, num_cycles=4)
    path_fast = output_dir / "fast_gait.csv"
    df_fast.to_csv(path_fast, index=False)
    print(f"   Saved: {path_fast}")
    print(f"   Frames: {len(df_fast)}, Duration: 2.5s, Cycles: 4")

    print("\n" + "=" * 60)
    print("Sample commands to create videos:")
    print("=" * 60)
    print(f"\n# Sagittal view (side view - best for gait)")
    print(f"python main.py -i {path} -o output/realistic_gait_sagittal.mp4 -v sagittal")
    print(f"\n# 3D view")
    print(f"python main.py -i {path} -o output/realistic_gait_3d.mp4 -v 3d")
    print(f"\n# Multi-view")
    print(f"python main.py -i {path} -o output/realistic_gait_multi.mp4 -v multi")


if __name__ == "__main__":
    generate_all_samples()
