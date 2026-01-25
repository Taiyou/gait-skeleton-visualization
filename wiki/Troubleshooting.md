# Troubleshooting

## Common Issues

### "No module named 'cv2'" Error

OpenCV is required for video export.

```bash
pip install opencv-python
```

### "No module named 'yaml'" Error

PyYAML is required for configuration files.

```bash
pip install pyyaml
```

### "ffmpeg not found" Error

FFmpeg may be required for certain video codecs.

**Mac:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Markers Not Displaying

1. **Check CSV column names** - Must follow `MarkerName_X`, `MarkerName_Y`, `MarkerName_Z` format
2. **Verify marker names match** - YAML config marker names must match CSV prefixes
3. **Try auto-skeleton** - Use `--auto-skeleton` to auto-detect markers

```bash
python main.py -i data.csv -o output.mp4 --auto-skeleton
```

### Video Appears Static

This issue occurs when frames are not being captured correctly.

**Solution:** Ensure you have the latest version with the `.copy()` fix in the visualizers.

### Video Shows Backward Walking

The gait direction depends on how angles are calculated. If walking appears backward:

1. Check your data's X-axis direction
2. Verify coordinate system matches the expected:
   - X: positive = forward
   - Y: positive = left
   - Z: positive = up

### Skeleton Not Visible (White on White)

Change the skeleton color in `config/marker_sets.yaml`:

```yaml
colors:
  default: "#000000"  # Black skeleton on white background
```

### Memory Issues with Large Files

For very large datasets:

1. Use frame range options:
```bash
python main.py -i data.csv -o output.mp4 --start-frame 0 --end-frame 1000
```

2. Reduce output FPS:
```bash
python main.py -i data.csv -o output.mp4 --output-fps 15
```

### Video Codec Issues

If video export fails, the tool tries multiple codecs:
1. `mp4v` (default)
2. `avc1`
3. `XVID`

If all fail, check your OpenCV installation:
```bash
python -c "import cv2; print(cv2.getBuildInformation())"
```

## Getting Help

If you encounter issues not listed here:

1. Check the [GitHub Issues](https://github.com/Taiyou/gait-skeleton-visualization/issues)
2. Create a new issue with:
   - Your Python version
   - Your operating system
   - The error message
   - Sample of your CSV data format
