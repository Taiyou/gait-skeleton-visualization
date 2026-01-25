# View Types

## Available Views

The tool supports five different visualization modes:

| View | Flag | Description |
|------|------|-------------|
| 3D | `-v 3d` | Full 3D perspective view |
| Sagittal | `-v sagittal` | Side view (X-Z plane) |
| Frontal | `-v frontal` | Front view (Y-Z plane) |
| Transverse | `-v transverse` | Top view (X-Y plane) |
| Multi | `-v multi` | All three 2D planes side-by-side |

## 3D View

Best for overall motion visualization.

```bash
python main.py -i data.csv -o output.mp4 -v 3d
```

**Axes displayed:**
- X: Anterior-Posterior (forward/backward)
- Y: Medio-Lateral (left/right)
- Z: Vertical (up/down)

## Sagittal View (Side View)

Ideal for analyzing:
- Forward progression
- Hip, knee, ankle flexion/extension
- Trunk lean

```bash
python main.py -i data.csv -o output.mp4 -v sagittal
```

**Plane:** X-Z (looking from the side)

## Frontal View (Front View)

Ideal for analyzing:
- Lateral trunk sway
- Hip abduction/adduction
- Foot placement width

```bash
python main.py -i data.csv -o output.mp4 -v frontal
```

**Plane:** Y-Z (looking from the front)

## Transverse View (Top View)

Ideal for analyzing:
- Pelvic rotation
- Step width
- Foot progression angle

```bash
python main.py -i data.csv -o output.mp4 -v transverse
```

**Plane:** X-Y (looking from above)

## Multi-View

Shows all three 2D planes simultaneously in a single video.

```bash
python main.py -i data.csv -o output.mp4 -v multi
```

This creates a wider video with three panels:
1. Sagittal (left)
2. Frontal (center)
3. Transverse (right)

## View Customization

### Marker Size

```bash
python main.py -i data.csv -o output.mp4 --marker-size 80
```

### Line Width

```bash
python main.py -i data.csv -o output.mp4 --line-width 3.0
```

### Show Labels

```bash
python main.py -i data.csv -o output.mp4 --show-labels
```
