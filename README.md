# Gait Skeleton Visualization

歩行分析のマーカーデータから骨格アニメーション動画を作成するPythonツールです。

## 機能

- **CSV形式のマーカーデータ読み込み**
- **カスタマイズ可能な骨格モデル** - YAMLファイルでマーカーセットを定義
- **3D可視化** - 自由な視点からの表示
- **2D可視化** - 矢状面（sagittal）、前額面（frontal）、水平面（transverse）
- **複数ビュー同時表示** - 3つの面を並べて表示
- **MP4動画出力**

## インストール

```bash
# リポジトリをクローン（または移動）
cd gait-skeleton-visualization

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
.\venv\Scripts\activate  # Windows

# 依存関係をインストール
pip install -r requirements.txt
```

## クイックスタート

### 1. サンプルデータの生成

```bash
python scripts/generate_sample_data.py
```

これにより、`data/` フォルダにテスト用のCSVファイルが作成されます。

### 2. 動画の作成

```bash
# 3D表示
python main.py -i data/sample_simple.csv -o output/skeleton_3d.mp4 -v 3d

# 矢状面（横から見た図）
python main.py -i data/sample_simple.csv -o output/skeleton_sagittal.mp4 -v sagittal

# 前額面（正面から見た図）
python main.py -i data/sample_simple.csv -o output/skeleton_frontal.mp4 -v frontal

# 3つのビューを同時に表示
python main.py -i data/sample_simple.csv -o output/skeleton_multi.mp4 -v multi

# マーカーラベルを表示
python main.py -i data/sample_simple.csv -o output/skeleton_labels.mp4 -v 3d --show-labels
```

## コマンドラインオプション

| オプション | 短縮形 | 説明 | デフォルト |
|-----------|--------|------|-----------|
| `--input` | `-i` | 入力CSVファイルパス | (必須) |
| `--output` | `-o` | 出力MP4ファイルパス | (必須) |
| `--config` | `-c` | マーカーセット設定ファイル | `config/marker_sets.yaml` |
| `--marker-set` | `-m` | 使用するマーカーセット名 | `simple` |
| `--view` | `-v` | 表示タイプ (3d/sagittal/frontal/transverse/multi) | `3d` |
| `--frame-rate` | `-f` | 入力データのフレームレート (Hz) | `100` |
| `--output-fps` | | 出力動画のFPS | `30` |
| `--show-labels` | | マーカーラベルを表示 | `False` |
| `--start-frame` | | 開始フレーム | `0` |
| `--end-frame` | | 終了フレーム | 全フレーム |
| `--marker-size` | | マーカーポイントのサイズ | `50` |
| `--line-width` | | 骨格線の太さ | `2.0` |
| `--auto-skeleton` | | データから自動で骨格作成（接続なし） | `False` |

## CSV形式

入力CSVは以下の形式を想定しています：

```csv
Frame,HEAD_X,HEAD_Y,HEAD_Z,LSHO_X,LSHO_Y,LSHO_Z,...
0,100.5,0.0,1700.2,80.2,200.0,1500.3,...
1,100.6,0.1,1700.1,80.3,200.1,1500.2,...
```

- 最初の列：フレーム番号または時刻
- 以降の列：`マーカー名_X`, `マーカー名_Y`, `マーカー名_Z` の形式

## マーカーセットのカスタマイズ

`config/marker_sets.yaml` を編集してカスタムマーカーセットを定義できます：

```yaml
my_custom_set:
  name: "My Custom Markers"
  description: "カスタムマーカーセット"
  
  markers:
    - MARKER1
    - MARKER2
    - MARKER3
  
  connections:
    - [MARKER1, MARKER2]
    - [MARKER2, MARKER3]
  
  colors:
    default: "#FFFFFF"
```

使用する場合：

```bash
python main.py -i data/my_data.csv -o output/video.mp4 -m my_custom_set
```

## Pythonからの使用

```python
from src.data_loader import DataLoader
from src.skeleton_model import SkeletonModel
from src.visualizer_3d import Visualizer3D
from src.video_exporter import VideoExporter

# データ読み込み
loader = DataLoader()
loader.load_csv("data/sample.csv", frame_rate=100.0)

# 骨格モデル読み込み
skeleton = SkeletonModel.from_yaml("config/marker_sets.yaml", "simple")
skeleton = skeleton.filter_markers(loader.markers)

# 可視化
visualizer = Visualizer3D(skeleton)
visualizer.set_bounds(loader.get_data_bounds())

# 全フレームの位置データ取得
all_positions = loader.get_all_positions()

# アニメーションフレーム生成
frames = visualizer.create_animation_frames(all_positions, frame_rate=100.0)

# 動画出力
exporter = VideoExporter(output_fps=30)
exporter.export(frames, "output/video.mp4")
```

## プロジェクト構造

```
gait-skeleton-visualization/
├── main.py                    # メインスクリプト
├── requirements.txt           # 依存関係
├── README.md                  # このファイル
├── config/
│   └── marker_sets.yaml       # マーカーセット定義
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSVデータ読み込み
│   ├── skeleton_model.py      # 骨格モデル定義
│   ├── visualizer_3d.py       # 3D可視化
│   ├── visualizer_2d.py       # 2D可視化
│   └── video_exporter.py      # 動画出力
├── scripts/
│   └── generate_sample_data.py # サンプルデータ生成
├── data/                      # データファイル
└── output/                    # 出力動画
```

## 座標系

- **X軸**: 前後方向（Anterior-Posterior）
- **Y軸**: 左右方向（Medio-Lateral）
- **Z軸**: 上下方向（Vertical）

## トラブルシューティング

### 「No module named 'cv2'」エラー

```bash
pip install opencv-python
```

### 「ffmpeg not found」エラー（音声付き動画作成時）

FFmpegをインストールしてください：

```bash
# Mac
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロード
```

### マーカーが表示されない

1. CSVの列名が `マーカー名_X`, `マーカー名_Y`, `マーカー名_Z` の形式か確認
2. `--auto-skeleton` オプションを使用してデータから自動で骨格を作成

## ライセンス

MIT License
