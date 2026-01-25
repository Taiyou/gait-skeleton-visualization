# Marker Set Configuration

## Configuration File

Marker sets are defined in `config/marker_sets.yaml`. Each set specifies:
- Which markers to display
- How markers are connected (bones)
- Color scheme for visualization

## YAML Structure

```yaml
my_custom_set:
  name: "My Custom Markers"
  description: "Description of the marker set"

  markers:
    - MARKER1
    - MARKER2
    - MARKER3

  connections:
    - [MARKER1, MARKER2]
    - [MARKER2, MARKER3]

  colors:
    default: "#000000"
```

## Available Marker Sets

### `simple`
Minimal 11-marker set for basic visualization:
- HEAD, LSHO, RSHO, LELB, RELB
- LHIP, RHIP, LKNE, RKNE, LANK, RANK

### `default`
Full-body marker set based on standard gait analysis protocols:
- Head markers (HEAD, LFHD, RFHD, LBHD, RBHD)
- Trunk markers (C7, T10, CLAV, STRN)
- Pelvis markers (LASI, RASI, LPSI, RPSI)
- Upper extremity markers
- Lower extremity markers

### `lower_body`
Lower body only for gait analysis:
- Pelvis (LASI, RASI, LPSI, RPSI)
- Legs (LKNE, RKNE, LANK, RANK, LHEE, RHEE, LTOE, RTOE)

## Creating Custom Marker Sets

### Step 1: Define Markers

List all marker names that match your CSV column prefixes:

```yaml
my_set:
  markers:
    - HEAD
    - CHEST
    - LSHO
    - RSHO
```

### Step 2: Define Connections

Specify pairs of markers to connect with lines:

```yaml
  connections:
    - [HEAD, CHEST]
    - [CHEST, LSHO]
    - [CHEST, RSHO]
```

### Step 3: Set Colors (Optional)

```yaml
  colors:
    head: "#FF6B6B"
    trunk: "#4ECDC4"
    default: "#000000"

  body_parts:
    head: [HEAD]
    trunk: [CHEST, LSHO, RSHO]
```

### Step 4: Use Your Marker Set

```bash
python main.py -i data.csv -o output.mp4 -m my_set
```

## Tips

- Marker names in YAML must exactly match CSV column prefixes
- The tool automatically filters out markers not found in the data
- Use `--show-labels` to display marker names in the video
