# Noise Removal and Drift Correction

This page explains the noise removal and drift correction techniques implemented in the Gait Skeleton Visualization tool.

## Overview

Motion capture data from IMU-based systems (like Xsens MVN) often contains various types of noise and drift artifacts. This tool provides multiple correction methods to clean the data before visualization.

## Types of Noise and Drift

![Drift Types](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/drift_types.png)

| Type | Description | Cause |
|------|-------------|-------|
| **IMU Heading Drift** | Gradual rotation of the trajectory over time | Gyroscope integration error |
| **Y-axis Drift** | Lateral displacement accumulating over time | Accelerometer bias |
| **180-degree Flips** | Sudden direction changes in PCA analysis | Ambiguity in principal component direction |
| **High-frequency Noise** | Jitter in position data | Sensor noise, electromagnetic interference |

---

## Recommended Workflow

![Workflow Diagram](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/workflow_diagram.png)

For best results, apply corrections in this order:

1. **Load raw data** - Import from Xsens Excel or CSV
2. **Skip unstable frames** - Remove first/last 5-10 seconds
3. **Smooth PCA correction** - Fix heading drift
4. **Y-drift removal** - Remove lateral drift
5. **Horizontal alignment** - Align with X-axis
6. **Export corrected data** - Save for visualization

---

## Correction Methods

### 1. Smooth PCA Correction

> **See also:** [Smooth PCA Detailed Documentation](Smooth-PCA) for in-depth explanation with examples.

**Location:** `scripts/gait_correction/smooth_pca.py`

The Smooth PCA method corrects heading drift by:
1. Calculating PCA angles at regular intervals using sliding windows
2. Unwrapping 180-degree flips
3. Smoothly interpolating angles across all frames
4. Applying continuous rotation correction

![Smooth PCA Pipeline](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/smooth_pca_pipeline.png)

#### Parameters

```python
@dataclass
class SmoothPCAParams:
    window_seconds: float = 30.0          # Window size for PCA calculation
    sample_interval_seconds: float = 5.0   # Interval between PCA samples
    smoothing_factor: float = 0.1          # Fraction of samples for smoothing
    min_window_samples: int = 100          # Minimum samples per window
    frame_rate: int = 60                   # Frame rate in Hz
```

#### Usage

```python
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction, SmoothPCAParams

# Apply with default parameters
corrected_data, orig_x, orig_y = apply_smooth_pca_correction(
    positions,
    skip_start_seconds=5,
    skip_end_seconds=5
)

# Apply with custom parameters
params = SmoothPCAParams(
    window_seconds=20.0,
    sample_interval_seconds=3.0,
    smoothing_factor=0.15
)
corrected_data, orig_x, orig_y = apply_smooth_pca_correction(
    positions,
    params=params
)
```

#### Algorithm Details

**Step 1: Calculate PCA Angles**
```
For each sample interval:
    1. Extract window of (x, y) data around center frame
    2. Fit PCA to find principal axis
    3. Calculate angle: arctan2(component[0,1], component[0,0])
```

**Step 2: Unwrap Angles**
```
For each consecutive angle pair:
    If difference > 90 degrees:
        Subtract 180 degrees from current angle
    If difference < -90 degrees:
        Add 180 degrees to current angle
```

**Step 3: Smooth and Interpolate**
```
1. Apply uniform_filter1d to sampled angles
2. Use cubic interpolation to get angles for all frames
3. Apply additional smoothing with 2-second window
```

**Step 4: Apply Rotation**
```
For each frame and segment:
    1. Shift coordinates relative to pelvis center
    2. Apply rotation matrix using correction angle
    3. Reconstruct absolute positions
```

---

### 2. Y-axis Drift Removal

**Location:** `scripts/gait_correction/drift_removal.py`

Removes lateral drift using high-pass filtering.

![Y-drift Removal](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/y_drift_removal.png)

#### Usage

```python
from scripts.gait_correction.drift_removal import remove_y_drift

corrected = remove_y_drift(
    data,
    window_seconds=30.0,
    frame_rate=60
)
```

#### Algorithm

```
1. Extract pelvis Y trajectory
2. Apply low-pass filter (uniform_filter1d) to estimate drift
3. Subtract drift from all Y coordinates
```

The low-pass filter window (default 30 seconds) determines the cutoff frequency. Larger windows remove slower drift components.

---

### 3. Horizontal Alignment

**Location:** `scripts/gait_correction/drift_removal.py`

Aligns the trajectory with the X-axis using PCA.

#### Usage

```python
from scripts.gait_correction.drift_removal import align_horizontally

aligned_data = align_horizontally(data)
```

#### Algorithm

```
1. Fit PCA to pelvis (x, y) trajectory
2. Calculate rotation angle to align principal axis with X-axis
3. Apply rotation to all segments
```

---

### 4. Full Drift Correction Pipeline

**Location:** `scripts/gait_correction/drift_removal.py`

Combines Y-drift removal and horizontal alignment.

#### Usage

```python
from scripts.gait_correction.drift_removal import apply_full_drift_correction

corrected = apply_full_drift_correction(
    data,
    drift_window_seconds=30.0,
    frame_rate=60
)
```

---

### 5. Improved Drift Correction (with IMU Heading)

**Location:** `scripts/improved_drift_correction.py`

Advanced correction using IMU orientation data and turnaround detection.

#### Features

- **Turnaround Detection**: Automatically detects when walking direction reverses
- **IMU Heading Correction**: Uses gyroscope yaw data to correct heading drift
- **Segment-wise Correction**: Applies different corrections between turnarounds

![Turnaround Detection](https://raw.githubusercontent.com/Taiyou/gait-skeleton-visualization/main/assets/wiki/turnaround_detection.png)

#### Usage

```python
from scripts.improved_drift_correction import (
    detect_turnarounds,
    correct_imu_heading_drift,
    apply_heading_correction,
    apply_translation_drift_correction
)

# Detect turnarounds
turnarounds = detect_turnarounds(pelvis_x, frame_rate=60)

# Correct IMU heading drift
corrected_heading = correct_imu_heading_drift(pelvis_yaw, turnarounds)

# Apply corrections
x_corrected, y_corrected, angles = apply_heading_correction(
    positions, corrected_heading
)

# Apply translation drift correction
x_final, y_final = apply_translation_drift_correction(
    x_corrected, y_corrected, turnarounds
)
```

#### Turnaround Detection Algorithm

```
1. Calculate X velocity from position data
2. Smooth velocity with 0.5-second window
3. Apply threshold (0.05 m/s) to avoid noise triggers
4. Detect zero-crossings with minimum 3-second interval
```

---

## Filtering Techniques

### uniform_filter1d (scipy.ndimage)

Used for low-pass filtering and smoothing. Acts as a moving average filter.

```python
from scipy.ndimage import uniform_filter1d

# Apply 30-second moving average at 60 Hz
window = 30 * 60  # 1800 samples
smoothed = uniform_filter1d(data, size=window, mode='nearest')
```

### Cubic Interpolation

Used for smooth interpolation between sampled points.

```python
from scipy.interpolate import interp1d

interpolator = interp1d(
    sample_frames,
    sample_values,
    kind='cubic',
    bounds_error=False,
    fill_value=(first_value, last_value)
)
all_values = interpolator(np.arange(n_frames))
```

### PCA-based Analysis

Used to find the principal axis of motion.

```python
from sklearn.decomposition import PCA

data = np.column_stack([x, y])
pca = PCA(n_components=2)
pca.fit(data)

# Principal axis angle
angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
```

---

## Complete Example

```python
from scripts.gait_correction.loader import load_xsens_data
from scripts.gait_correction.smooth_pca import apply_smooth_pca_correction
from scripts.gait_correction.drift_removal import apply_full_drift_correction
from scripts.gait_correction.export import export_to_csv

# Load data
positions = load_xsens_data("data/trial.xlsx")

# Apply Smooth PCA correction
corrected, _, _ = apply_smooth_pca_correction(
    positions,
    skip_start_seconds=5,
    skip_end_seconds=5
)

# Apply drift correction
final = apply_full_drift_correction(corrected)

# Export
export_to_csv(final, "output/corrected.csv", subsample=60)
```

---

## Visualization

The correction process can be visualized using the scripts in `scripts/`:

```bash
# Visualize Smooth PCA correction
python scripts/create_smooth_pca_animation.py

# Visualize improved drift correction
python scripts/improved_drift_correction.py
```

These scripts generate diagnostic plots showing:
- Original vs corrected trajectory
- PCA angles over time
- Turnaround detection
- Correction angles applied
- Statistical summary

---

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Trajectory still curved | Window size too small | Increase `window_seconds` |
| Jerky corrections | Smoothing factor too low | Increase `smoothing_factor` |
| Over-correction | Reference angle incorrect | Check skip start/end settings |
| Turnarounds not detected | Threshold too high | Lower velocity threshold |
| Residual Y drift | Drift window too small | Increase `drift_window_seconds` |

---

## References

- Xsens MVN User Manual - Segment Position Data
- Scipy Documentation - uniform_filter1d
- Scikit-learn Documentation - PCA
