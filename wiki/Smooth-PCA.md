# Smooth PCA Correction

This page provides a detailed explanation of the Smooth PCA algorithm used for correcting IMU heading drift in gait trajectory data.

## Overview

**Smooth PCA** is a trajectory correction method that uses Principal Component Analysis (PCA) to detect and correct gradual heading drift in motion capture data. It is particularly effective for long-duration recordings where accumulated drift causes the trajectory to curve or spiral.

**Location:** `scripts/gait_correction/smooth_pca.py`

---

## The Problem: IMU Heading Drift

IMU-based motion capture systems (like Xsens MVN) suffer from heading drift due to gyroscope integration error. Over time, the measured heading angle gradually deviates from the true heading.

### Visual Example

A straight walking path recorded over 30 minutes might appear as a curved or spiral trajectory:

```
Expected:    ──────────────────────────►

Actual:      ──────────╮
                       │
                       ╰──────────►
```

---

## How Smooth PCA Works

![PCA Concept](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/pca_concept.png)

### Core Idea

The main walking direction can be estimated using PCA:
- **PC1 (First Principal Component)**: Direction of maximum variance = walking direction
- By tracking how PC1 changes over time, we can detect and correct heading drift

### The Algorithm

![Smooth PCA Pipeline](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/smooth_pca_pipeline.png)

#### Step 1: Calculate PCA Angles at Regular Intervals

```python
for center_frame in range(0, n_frames, sample_interval):
    # Extract window of data
    window_data = trajectory[center - window//2 : center + window//2]

    # Fit PCA to find principal axis
    pca = PCA(n_components=2)
    pca.fit(window_data)

    # Calculate angle of first principal component
    angle = arctan2(pca.components_[0, 1], pca.components_[0, 0])
```

**Parameters:**
- `window_seconds`: Size of sliding window (default: 30s)
- `sample_interval_seconds`: Interval between samples (default: 5s)

#### Step 2: Unwrap 180-Degree Flips

PCA has a fundamental ambiguity: the principal component can point in either direction (±180°).

![180-Degree Flip](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/pca_180_flip.png)

```python
def unwrap_pca_angles(angles):
    unwrapped = angles.copy()
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i-1]

        if diff > π/2:      # Jump > 90°
            unwrapped[i] -= π   # Subtract 180°
        elif diff < -π/2:   # Jump < -90°
            unwrapped[i] += π   # Add 180°

    return unwrapped
```

#### Step 3: Smooth and Interpolate

Apply smoothing and interpolate to all frames:

```python
# 1. Apply moving average to sampled angles
smoothed = uniform_filter1d(angles, size=smooth_window)

# 2. Cubic interpolation to all frames
interpolator = interp1d(sample_frames, smoothed, kind='cubic')
all_angles = interpolator(all_frame_indices)

# 3. Additional smoothing (2-second window)
final_angles = uniform_filter1d(all_angles, size=frame_rate * 2)
```

#### Step 4: Calculate and Apply Correction

```python
# Reference angle = first frame
ref_angle = all_angles[0]

# Correction = negative of drift
correction_angles = -(all_angles - ref_angle)

# Apply rotation to each frame
for each frame:
    x_corrected = x * cos(correction) - y * sin(correction)
    y_corrected = x * sin(correction) + y * cos(correction)
```

---

## Parameters

```python
@dataclass
class SmoothPCAParams:
    window_seconds: float = 30.0          # PCA window size
    sample_interval_seconds: float = 5.0   # Sampling interval
    smoothing_factor: float = 0.1          # Smoothing strength
    min_window_samples: int = 100          # Minimum samples per window
    frame_rate: int = 60                   # Frame rate (Hz)
```

### Parameter Effects

![Parameter Effects](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/pca_parameters.png)

| Parameter | Too Small | Recommended | Too Large |
|-----------|-----------|-------------|-----------|
| `window_seconds` | Follows noise | 30s | Over-smoothed |
| `sample_interval_seconds` | Slow computation | 5s | Misses changes |
| `smoothing_factor` | Jerky corrections | 0.1 | Laggy response |

### Recommended Settings by Data Type

| Data Type | Duration | window_seconds | sample_interval |
|-----------|----------|----------------|-----------------|
| Short walk | < 5 min | 15-20 | 3 |
| Medium walk | 5-30 min | 30 (default) | 5 (default) |
| Long walk | > 30 min | 45-60 | 10 |
| Back-and-forth | Any | 20-30 | 5 |

---

## Usage Examples

### Basic Usage

```python
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction

# Apply with default parameters
corrected, orig_x, orig_y = apply_smooth_pca_correction(
    positions,
    skip_start_seconds=5,
    skip_end_seconds=5
)
```

### Custom Parameters

```python
from scripts.gait_correction.smooth_pca import (
    apply_smooth_pca_correction,
    SmoothPCAParams
)

# For long recordings with slow drift
params = SmoothPCAParams(
    window_seconds=45.0,
    sample_interval_seconds=10.0,
    smoothing_factor=0.15
)

corrected, orig_x, orig_y = apply_smooth_pca_correction(
    positions,
    params=params,
    skip_start_seconds=10,
    skip_end_seconds=10
)
```

### Manual Step-by-Step

```python
from scripts.gait_correction.smooth_pca import (
    calculate_pca_angles,
    unwrap_pca_angles,
    smooth_and_interpolate_angles,
    apply_rotation_to_segments,
    SmoothPCAParams
)

params = SmoothPCAParams()

# Step 1: Calculate angles
sample_frames, sample_angles = calculate_pca_angles(x, y, params)

# Step 2: Unwrap flips
unwrapped = unwrap_pca_angles(sample_angles)

# Step 3: Smooth and interpolate
all_angles = smooth_and_interpolate_angles(
    sample_frames, unwrapped, n_frames, params
)

# Step 4: Calculate correction
correction = -(all_angles - all_angles[0])

# Step 5: Apply to all segments
corrected = apply_rotation_to_segments(
    positions, correction, pelvis_x, pelvis_y, skip_start, skip_end
)
```

---

## Understanding the Math

### PCA for Direction Finding

Given a set of 2D points, PCA finds the directions that explain the most variance:

```
Covariance Matrix:
    C = [ var(x)      cov(x,y) ]
        [ cov(x,y)    var(y)   ]

Eigenvectors of C = Principal Components
PC1 = eigenvector with largest eigenvalue = main walking direction
```

### Rotation Matrix

To correct heading by angle θ:

```
[ x' ]   [ cos(θ)  -sin(θ) ] [ x ]
[ y' ] = [ sin(θ)   cos(θ) ] [ y ]
```

### Angle Calculation

```python
angle = arctan2(PC1[1], PC1[0])
```

Where `PC1 = [PC1_x, PC1_y]` is the first principal component.

---

## Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Smooth PCA** | Works with position data only, handles gradual drift | Requires sufficient data per window |
| **IMU Heading Correction** | Uses raw sensor data | Needs orientation data |
| **Turnaround-based** | Precise at turn points | Only works with back-and-forth walking |
| **Kalman Filter** | Optimal estimation | Complex implementation |

---

## Troubleshooting

### Problem: Correction is too aggressive

**Symptoms:** Trajectory oscillates or overcorrects

**Solutions:**
- Increase `window_seconds` (e.g., 45 or 60)
- Increase `smoothing_factor` (e.g., 0.15 or 0.2)

### Problem: Correction is not enough

**Symptoms:** Trajectory still curves significantly

**Solutions:**
- Decrease `window_seconds` (e.g., 20 or 25)
- Check if data has very fast drift (may need different approach)

### Problem: Jerky movements in output

**Symptoms:** Corrected skeleton moves unnaturally

**Solutions:**
- Increase `smoothing_factor`
- Ensure frame rate is set correctly

### Problem: "Not enough samples" error

**Symptoms:** Algorithm fails on short recordings

**Solutions:**
- Decrease `window_seconds`
- Decrease `min_window_samples`
- Ensure `skip_start/end_seconds` doesn't remove too much data

---

## API Reference

### apply_smooth_pca_correction()

```python
def apply_smooth_pca_correction(
    positions: pd.DataFrame,
    skip_start_seconds: float = 0,
    skip_end_seconds: float = 0,
    params: Optional[SmoothPCAParams] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Apply complete Smooth PCA correction pipeline.

    Parameters
    ----------
    positions : pd.DataFrame
        Original position data from Xsens
        Required columns: 'Pelvis x', 'Pelvis y', etc.
    skip_start_seconds : float
        Seconds to skip at the start (unstable data)
    skip_end_seconds : float
        Seconds to skip at the end (unstable data)
    params : SmoothPCAParams, optional
        PCA parameters (uses defaults if None)
    verbose : bool
        Print progress information

    Returns
    -------
    corrected_data : pd.DataFrame
        Corrected position data with columns like 'PELVIS_X', 'PELVIS_Y', etc.
    original_x : np.ndarray
        Original pelvis X coordinates (for visualization)
    original_y : np.ndarray
        Original pelvis Y coordinates (for visualization)
    """
```

### SmoothPCAParams

```python
@dataclass
class SmoothPCAParams:
    window_seconds: float = 30.0
    sample_interval_seconds: float = 5.0
    smoothing_factor: float = 0.1
    min_window_samples: int = 100
    frame_rate: int = 60
```

---

## Related Pages

- [Noise Removal](Noise-Removal) - Overview of all noise removal methods
- [Python API](Python-API) - General API documentation
- [Troubleshooting](Troubleshooting) - Common issues and solutions
