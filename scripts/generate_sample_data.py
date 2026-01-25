#!/usr/bin/env python3
"""
Generate sample gait data for testing the visualization system.

This script creates synthetic walking motion data that simulates
realistic marker trajectories during gait.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_treadmill_walking(
    num_frames: int = 300,
    frame_rate: float = 100.0,
    gait_cycles: int = 3,
) -> pd.DataFrame:
    """
    Generate treadmill walking motion (stationary position, visible leg movement).
    
    This creates a walking motion where the body stays in place but the legs
    move as if walking on a treadmill - making the motion clearly visible.
    
    Args:
        num_frames: Number of frames to generate
        frame_rate: Frame rate in Hz
        gait_cycles: Number of complete gait cycles
        
    Returns:
        DataFrame with marker positions
    """
    t = np.arange(num_frames) / frame_rate
    duration = num_frames / frame_rate
    
    # Gait frequency (cycles per second)
    gait_freq = gait_cycles / duration
    
    # Phase for left and right legs (180 degrees apart)
    phase_left = 2 * np.pi * gait_freq * t
    phase_right = phase_left + np.pi
    
    markers = {}
    
    # ============== BODY DIMENSIONS (mm) ==============
    head_height = 1700
    shoulder_height = 1450
    shoulder_width = 400
    elbow_height = 1100
    hip_height = 950
    hip_width = 280
    knee_height = 500
    ankle_height = 80
    
    # ============== MOTION AMPLITUDES ==============
    # Vertical body oscillation (double frequency - up at each step)
    body_vertical_amp = 40
    
    # Forward/backward swing amplitudes
    leg_swing_amp = 300      # How far legs swing forward/back
    arm_swing_amp = 200      # How far arms swing
    knee_forward_amp = 250   # Knee forward swing
    ankle_forward_amp = 350  # Ankle forward swing (larger than knee)
    
    # Vertical lift during swing phase
    knee_lift_amp = 120      # Knee lifts during swing
    ankle_lift_amp = 100     # Ankle lifts during swing
    
    # Lateral sway
    lateral_sway = 30
    
    # ============== HEAD ==============
    markers['HEAD_X'] = 30 * np.sin(2 * phase_left)  # Slight forward/back
    markers['HEAD_Y'] = lateral_sway * np.sin(phase_left)  # Lateral sway
    markers['HEAD_Z'] = head_height + body_vertical_amp * np.sin(2 * phase_left)
    
    # ============== SHOULDERS ==============
    # Shoulders rotate opposite to hips (counter-rotation)
    shoulder_rotation = 30 * np.sin(phase_left)
    
    markers['LSHO_X'] = shoulder_rotation
    markers['LSHO_Y'] = shoulder_width / 2
    markers['LSHO_Z'] = shoulder_height + body_vertical_amp * np.sin(2 * phase_left)
    
    markers['RSHO_X'] = -shoulder_rotation
    markers['RSHO_Y'] = -shoulder_width / 2
    markers['RSHO_Z'] = shoulder_height + body_vertical_amp * np.sin(2 * phase_left)
    
    # ============== ELBOWS (arm swing) ==============
    # Arms swing opposite to legs (left arm forward when right leg forward)
    markers['LELB_X'] = arm_swing_amp * np.sin(phase_right)  # Opposite to left leg
    markers['LELB_Y'] = shoulder_width / 2 + 80
    markers['LELB_Z'] = elbow_height + 50 * np.sin(2 * phase_left) - 30 * np.abs(np.sin(phase_right))
    
    markers['RELB_X'] = arm_swing_amp * np.sin(phase_left)  # Opposite to right leg
    markers['RELB_Y'] = -shoulder_width / 2 - 80
    markers['RELB_Z'] = elbow_height + 50 * np.sin(2 * phase_left) - 30 * np.abs(np.sin(phase_left))
    
    # ============== HIPS ==============
    # Hips rotate slightly and tilt
    hip_rotation = 20 * np.sin(phase_left)
    
    markers['LHIP_X'] = hip_rotation
    markers['LHIP_Y'] = hip_width / 2
    markers['LHIP_Z'] = hip_height + 30 * np.sin(2 * phase_left) + 15 * np.sin(phase_left)
    
    markers['RHIP_X'] = -hip_rotation
    markers['RHIP_Y'] = -hip_width / 2
    markers['RHIP_Z'] = hip_height + 30 * np.sin(2 * phase_left) - 15 * np.sin(phase_left)
    
    # ============== KNEES ==============
    # Knees swing forward/back and lift during swing phase
    def knee_motion(phase):
        # Forward/backward swing
        x = knee_forward_amp * np.sin(phase)
        
        # Vertical: lifts during swing (when moving forward)
        # swing phase is when sin(phase) > 0
        swing_factor = np.maximum(0, np.sin(phase))
        z = knee_height + knee_lift_amp * swing_factor
        
        return x, z
    
    lkne_x, lkne_z = knee_motion(phase_left)
    rkne_x, rkne_z = knee_motion(phase_right)
    
    markers['LKNE_X'] = lkne_x
    markers['LKNE_Y'] = hip_width / 2 - 20
    markers['LKNE_Z'] = lkne_z
    
    markers['RKNE_X'] = rkne_x
    markers['RKNE_Y'] = -hip_width / 2 + 20
    markers['RKNE_Z'] = rkne_z
    
    # ============== ANKLES ==============
    # Ankles have largest swing and lift during swing phase
    def ankle_motion(phase):
        # Forward/backward swing (larger amplitude)
        x = ankle_forward_amp * np.sin(phase)
        
        # Vertical: lifts during swing phase for foot clearance
        swing_factor = np.maximum(0, np.sin(phase))
        # Additional lift at mid-swing
        mid_swing_boost = np.maximum(0, np.sin(phase * 2)) * 0.3
        z = ankle_height + ankle_lift_amp * (swing_factor + mid_swing_boost)
        
        return x, z
    
    lank_x, lank_z = ankle_motion(phase_left)
    rank_x, rank_z = ankle_motion(phase_right)
    
    markers['LANK_X'] = lank_x
    markers['LANK_Y'] = hip_width / 2 - 30
    markers['LANK_Z'] = lank_z
    
    markers['RANK_X'] = rank_x
    markers['RANK_Y'] = -hip_width / 2 + 30
    markers['RANK_Z'] = rank_z
    
    # Create DataFrame
    df = pd.DataFrame(markers)
    df.insert(0, 'Frame', np.arange(num_frames))
    
    return df


def generate_exaggerated_walk(
    num_frames: int = 200,
    frame_rate: float = 100.0,
) -> pd.DataFrame:
    """
    Generate exaggerated walking motion for clear visualization.
    All movements are amplified for better visibility.
    """
    t = np.arange(num_frames) / frame_rate
    duration = num_frames / frame_rate
    
    # 2 complete gait cycles
    gait_freq = 2.0 / duration
    phase_L = 2 * np.pi * gait_freq * t
    phase_R = phase_L + np.pi
    
    # Amplitudes (exaggerated)
    leg_swing = 400    # Large leg swing
    arm_swing = 250    # Large arm swing
    knee_lift = 180    # High knee lift
    ankle_lift = 150   # High ankle lift
    body_bob = 60      # Noticeable body bob
    
    # Body dimensions
    head_z = 1700
    shoulder_z = 1450
    shoulder_y = 200
    hip_z = 950
    hip_y = 140
    knee_z = 500
    ankle_z = 80
    
    data = {
        'Frame': np.arange(num_frames),
        
        # HEAD - bobs up and down
        'HEAD_X': 40 * np.sin(2 * phase_L),
        'HEAD_Y': 40 * np.sin(phase_L),  # Lateral sway
        'HEAD_Z': head_z + body_bob * np.sin(2 * phase_L),
        
        # SHOULDERS - counter-rotate
        'LSHO_X': 50 * np.sin(phase_L),
        'LSHO_Y': shoulder_y * np.ones(num_frames),
        'LSHO_Z': shoulder_z + body_bob * np.sin(2 * phase_L),
        
        'RSHO_X': -50 * np.sin(phase_L),
        'RSHO_Y': -shoulder_y * np.ones(num_frames),
        'RSHO_Z': shoulder_z + body_bob * np.sin(2 * phase_L),
        
        # ELBOWS - large arm swing (opposite to legs)
        'LELB_X': arm_swing * np.sin(phase_R),  # Left arm swings with right leg
        'LELB_Y': (shoulder_y + 80) * np.ones(num_frames),
        'LELB_Z': 1100 + 40 * np.sin(2 * phase_L) - 50 * np.abs(np.sin(phase_R)),
        
        'RELB_X': arm_swing * np.sin(phase_L),  # Right arm swings with left leg
        'RELB_Y': -(shoulder_y + 80) * np.ones(num_frames),
        'RELB_Z': 1100 + 40 * np.sin(2 * phase_L) - 50 * np.abs(np.sin(phase_L)),
        
        # HIPS - rotate
        'LHIP_X': 30 * np.sin(phase_L),
        'LHIP_Y': hip_y * np.ones(num_frames),
        'LHIP_Z': hip_z + 30 * np.sin(2 * phase_L) + 20 * np.sin(phase_L),
        
        'RHIP_X': -30 * np.sin(phase_L),
        'RHIP_Y': -hip_y * np.ones(num_frames),
        'RHIP_Z': hip_z + 30 * np.sin(2 * phase_L) - 20 * np.sin(phase_L),
        
        # KNEES - swing and lift
        'LKNE_X': leg_swing * 0.7 * np.sin(phase_L),
        'LKNE_Y': (hip_y - 20) * np.ones(num_frames),
        'LKNE_Z': knee_z + knee_lift * np.maximum(0, np.sin(phase_L)),
        
        'RKNE_X': leg_swing * 0.7 * np.sin(phase_R),
        'RKNE_Y': -(hip_y - 20) * np.ones(num_frames),
        'RKNE_Z': knee_z + knee_lift * np.maximum(0, np.sin(phase_R)),
        
        # ANKLES - large swing and lift
        'LANK_X': leg_swing * np.sin(phase_L),
        'LANK_Y': (hip_y - 30) * np.ones(num_frames),
        'LANK_Z': ankle_z + ankle_lift * np.maximum(0, np.sin(phase_L)),
        
        'RANK_X': leg_swing * np.sin(phase_R),
        'RANK_Y': -(hip_y - 30) * np.ones(num_frames),
        'RANK_Z': ankle_z + ankle_lift * np.maximum(0, np.sin(phase_R)),
    }
    
    return pd.DataFrame(data)


def generate_lower_body_only(
    num_frames: int = 200,
    frame_rate: float = 100.0,
) -> pd.DataFrame:
    """
    Generate lower body only walking data.
    This reduces the Z-axis range so leg movement is more visible.
    """
    t = np.arange(num_frames) / frame_rate
    duration = num_frames / frame_rate
    
    # 3 complete gait cycles for better observation
    gait_freq = 3.0 / duration
    phase_L = 2 * np.pi * gait_freq * t
    phase_R = phase_L + np.pi
    
    # Large amplitudes for clear visibility
    leg_swing = 500      # Very large leg swing
    knee_lift = 250      # Very high knee lift
    ankle_lift = 200     # Very high ankle lift
    
    # Lower body dimensions only (reduces Z range)
    hip_z = 900
    hip_y = 150
    knee_z = 480
    ankle_z = 70
    
    data = {
        'Frame': np.arange(num_frames),
        
        # HIPS - pelvis movement
        'LHIP_X': 50 * np.sin(phase_L),
        'LHIP_Y': hip_y * np.ones(num_frames),
        'LHIP_Z': hip_z + 50 * np.sin(2 * phase_L) + 30 * np.sin(phase_L),
        
        'RHIP_X': -50 * np.sin(phase_L),
        'RHIP_Y': -hip_y * np.ones(num_frames),
        'RHIP_Z': hip_z + 50 * np.sin(2 * phase_L) - 30 * np.sin(phase_L),
        
        # KNEES - large swing and high lift
        'LKNE_X': leg_swing * 0.6 * np.sin(phase_L),
        'LKNE_Y': (hip_y - 20) * np.ones(num_frames),
        'LKNE_Z': knee_z + knee_lift * np.maximum(0, np.sin(phase_L)),
        
        'RKNE_X': leg_swing * 0.6 * np.sin(phase_R),
        'RKNE_Y': -(hip_y - 20) * np.ones(num_frames),
        'RKNE_Z': knee_z + knee_lift * np.maximum(0, np.sin(phase_R)),
        
        # ANKLES - very large swing and lift
        'LANK_X': leg_swing * np.sin(phase_L),
        'LANK_Y': (hip_y - 30) * np.ones(num_frames),
        'LANK_Z': ankle_z + ankle_lift * np.maximum(0, np.sin(phase_L)),
        
        'RANK_X': leg_swing * np.sin(phase_R),
        'RANK_Y': -(hip_y - 30) * np.ones(num_frames),
        'RANK_Z': ankle_z + ankle_lift * np.maximum(0, np.sin(phase_R)),
    }
    
    return pd.DataFrame(data)


def main():
    """Generate sample data files."""
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating sample gait data")
    print("=" * 60)
    
    # Generate treadmill walking motion (stationary body, visible leg movement)
    print("\n1. Generating treadmill walking data...")
    print("   (Body stays in place, legs move clearly)")
    treadmill_df = generate_treadmill_walking(num_frames=300, frame_rate=100.0, gait_cycles=3)
    treadmill_path = output_dir / "sample_treadmill.csv"
    treadmill_df.to_csv(treadmill_path, index=False)
    print(f"   Saved to: {treadmill_path}")
    print(f"   Frames: {len(treadmill_df)}, Duration: 3.0 sec, Gait cycles: 3")
    
    # Generate exaggerated walking for clear visualization
    print("\n2. Generating exaggerated walking data...")
    print("   (Large, clearly visible movements)")
    exaggerated_df = generate_exaggerated_walk(num_frames=200, frame_rate=100.0)
    exaggerated_path = output_dir / "sample_exaggerated.csv"
    exaggerated_df.to_csv(exaggerated_path, index=False)
    print(f"   Saved to: {exaggerated_path}")
    print(f"   Frames: {len(exaggerated_df)}, Duration: 2.0 sec, Gait cycles: 2")
    
    # Generate lower body only data (better visibility)
    print("\n3. Generating lower body only data...")
    print("   (Smaller Z range = more visible leg movement)")
    lower_df = generate_lower_body_only(num_frames=200, frame_rate=100.0)
    lower_path = output_dir / "sample_lower_body.csv"
    lower_df.to_csv(lower_path, index=False)
    print(f"   Saved to: {lower_path}")
    print(f"   Frames: {len(lower_df)}, Duration: 2.0 sec, Gait cycles: 3")
    
    # Show sample of the data
    print("\n4. Sample data preview (lower body walking):")
    print("   左足首と右足首のX座標（前後の動き）:")
    for i in [0, 17, 33, 50, 67, 83, 100]:
        t = i / 100.0
        lx = lower_df.iloc[i]['LANK_X']
        rx = lower_df.iloc[i]['RANK_X']
        print(f"   {t:.2f}s: 左足X={lx:+7.0f}mm, 右足X={rx:+7.0f}mm")
    
    print("\n" + "=" * 60)
    print("Done! Run these commands to create videos:")
    print("=" * 60)
    print(f"\n  # Lower body only (RECOMMENDED - most visible)")
    print(f"  python main.py -i {lower_path} -o output/lower_body_sagittal.mp4 -v sagittal --auto-skeleton")
    print(f"\n  # Full body sagittal view")
    print(f"  python main.py -i {treadmill_path} -o output/walking_sagittal.mp4 -v sagittal")
    print(f"\n  # Full body 3D view")
    print(f"  python main.py -i {treadmill_path} -o output/walking_3d.mp4 -v 3d")


if __name__ == "__main__":
    main()
