# CSV Data Format

## Basic Structure

The input CSV file should follow this format:

```csv
Frame,HEAD_X,HEAD_Y,HEAD_Z,LSHO_X,LSHO_Y,LSHO_Z,...
0,100.5,0.0,1700.2,80.2,200.0,1500.3,...
1,100.6,0.1,1700.1,80.3,200.1,1500.2,...
```

## Column Requirements

### Frame Column
- First column: Frame number or timestamp
- Can be named `Frame`, `Time`, or similar

### Marker Columns
- Format: `MarkerName_X`, `MarkerName_Y`, `MarkerName_Z`
- Each marker requires three columns (X, Y, Z coordinates)
- Units: millimeters (mm) recommended

## Example

For a marker named `HEAD`:
- `HEAD_X` - X coordinate (anterior-posterior)
- `HEAD_Y` - Y coordinate (medio-lateral)
- `HEAD_Z` - Z coordinate (vertical)

## Common Marker Naming Conventions

| Marker | Description |
|--------|-------------|
| HEAD | Head/top of head |
| LSHO / RSHO | Left/Right shoulder |
| LELB / RELB | Left/Right elbow |
| LWRA / RWRA | Left/Right wrist |
| LASI / RASI | Left/Right ASIS (pelvis front) |
| LPSI / RPSI | Left/Right PSIS (pelvis back) |
| LKNE / RKNE | Left/Right knee |
| LANK / RANK | Left/Right ankle |
| LHEE / RHEE | Left/Right heel |
| LTOE / RTOE | Left/Right toe |

## Using Custom Data

1. Ensure your CSV follows the `MarkerName_X/Y/Z` format
2. Either:
   - Use `--auto-skeleton` flag to auto-detect markers
   - Create a custom marker set in `config/marker_sets.yaml`

```bash
# Auto-detect markers (no connections)
python main.py -i your_data.csv -o output.mp4 --auto-skeleton

# Use custom marker set
python main.py -i your_data.csv -o output.mp4 -m your_marker_set
```
