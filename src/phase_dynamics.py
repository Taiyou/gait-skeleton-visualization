"""
Phase Dynamics Module
Computes joint angles from marker positions for gait analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# Default joint definitions for type03 marker set
# Each entry: (joint_label, proximal_marker, joint_marker, distal_marker)
DEFAULT_JOINT_DEFINITIONS: List[Tuple[str, str, str, str]] = [
    ("knee_right", "hipR", "knR", "anR"),
    ("knee_left", "hipL", "knL", "anL"),
    ("hip_right", "ribR", "hipR", "knR"),
    ("hip_left", "ribL", "hipL", "knL"),
    ("elbow_right", "shR", "elbR", "wrR"),
    ("elbow_left", "shL", "elbL", "wrL"),
]


def compute_angle(
    p1: np.ndarray,
    p_joint: np.ndarray,
    p2: np.ndarray,
) -> float:
    """
    Compute the angle at p_joint formed by segments p1-p_joint and p2-p_joint.

    Args:
        p1: Position of the proximal marker (3D)
        p_joint: Position of the joint marker (3D)
        p2: Position of the distal marker (3D)

    Returns:
        Angle in degrees (0-180), or NaN if any input is invalid.
    """
    v1 = p1 - p_joint
    v2 = p2 - p_joint

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        return float("nan")

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_joint_angles(
    all_positions: List[Dict[str, Optional[np.ndarray]]],
    joint_definitions: Optional[List[Tuple[str, str, str, str]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute joint angles for all frames.

    Args:
        all_positions: List of marker position dicts, one per frame.
        joint_definitions: List of (label, proximal, joint, distal) tuples.
            Defaults to DEFAULT_JOINT_DEFINITIONS.

    Returns:
        Dictionary mapping joint label to array of angles (degrees) per frame.
    """
    if joint_definitions is None:
        joint_definitions = DEFAULT_JOINT_DEFINITIONS

    num_frames = len(all_positions)
    angles: Dict[str, np.ndarray] = {
        jdef[0]: np.full(num_frames, np.nan) for jdef in joint_definitions
    }

    for i, positions in enumerate(all_positions):
        for label, proximal, joint, distal in joint_definitions:
            p1 = positions.get(proximal)
            pj = positions.get(joint)
            p2 = positions.get(distal)

            if p1 is not None and pj is not None and p2 is not None:
                angles[label][i] = compute_angle(p1, pj, p2)

    return angles
