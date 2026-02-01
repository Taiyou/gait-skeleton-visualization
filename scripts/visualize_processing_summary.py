#!/usr/bin/env python3
"""
Visualization script for datatype2 preprocessing results.
Creates summary plots showing the effectiveness of the preprocessing pipeline.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_corrected_data(csv_path: Path) -> pd.DataFrame:
    """Load corrected CSV data."""
    return pd.read_csv(csv_path)


def calculate_y_range(df: pd.DataFrame) -> float:
    """Calculate Y range from Pelvis position."""
    if "Pelvis_Y" in df.columns:
        return df["Pelvis_Y"].max() - df["Pelvis_Y"].min()
    return 0.0


def load_original_data(xlsx_path: Path) -> np.ndarray:
    """Load original data and return Pelvis positions."""
    try:
        from scripts.gait_correction.loader import load_xsens_data
        loader = load_xsens_data(xlsx_path)
        # Get Pelvis position (first segment, index 0)
        pelvis_positions = loader.positions[:, 0, :]  # (n_frames, 3) for X, Y, Z
        return pelvis_positions
    except Exception as e:
        print(f"Error loading {xlsx_path}: {e}")
        return None


def create_summary_visualization():
    """Create comprehensive summary visualization."""
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data" / "type2"

    # Collect data from all processed folders
    folders = [
        ("datatype2", "datatype2_processed"),
        ("datatype2 2", "datatype2 2_processed"),
        ("datatype2 3", "datatype2 3_processed"),
        ("datatype2 4", "datatype2 4_processed"),
    ]

    all_data = []

    for orig_folder, proc_folder in folders:
        orig_dir = base_dir / orig_folder
        proc_dir = base_dir / proc_folder

        if not proc_dir.exists():
            print(f"Skipping {proc_folder} - not found")
            continue

        # Find all corrected CSV files
        for csv_file in sorted(proc_dir.glob("*_corrected.csv")):
            file_id = csv_file.stem.replace("_corrected", "")
            orig_file = orig_dir / f"{file_id}.xlsx"

            # Load corrected data
            corrected_df = load_corrected_data(csv_file)
            corrected_y_range = calculate_y_range(corrected_df)

            # Load original data
            original_positions = load_original_data(orig_file)
            if original_positions is not None:
                original_y_range = original_positions[:, 1].max() - original_positions[:, 1].min()
            else:
                original_y_range = None

            # Calculate reduction percentage
            if original_y_range and original_y_range > 0:
                reduction_pct = (1 - corrected_y_range / original_y_range) * 100
            else:
                reduction_pct = None

            all_data.append({
                "folder": orig_folder,
                "file_id": file_id,
                "original_y_range": original_y_range,
                "corrected_y_range": corrected_y_range,
                "reduction_pct": reduction_pct,
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Create visualization
    fig = plt.figure(figsize=(20, 16))

    # 1. Bar chart comparing original vs corrected Y ranges
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax1.bar(x - width/2, df["original_y_range"], width, label="Original", color="salmon", alpha=0.8)
    bars2 = ax1.bar(x + width/2, df["corrected_y_range"], width, label="Corrected", color="steelblue", alpha=0.8)

    ax1.set_xlabel("File", fontsize=12)
    ax1.set_ylabel("Y Range (m)", fontsize=12)
    ax1.set_title("Original vs Corrected Y Range for All Files", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["file_id"], rotation=45, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. Reduction percentage bar chart
    ax2 = fig.add_subplot(2, 2, 2)
    colors = ["green" if p > 0 else "red" for p in df["reduction_pct"]]
    bars = ax2.bar(x, df["reduction_pct"], color=colors, alpha=0.7)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axhline(y=50, color="green", linestyle="--", linewidth=1, alpha=0.5, label="50% threshold")

    ax2.set_xlabel("File", fontsize=12)
    ax2.set_ylabel("Y Range Reduction (%)", fontsize=12)
    ax2.set_title("Drift Reduction Effectiveness by File", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["file_id"], rotation=45, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # 3. Box plot by folder
    ax3 = fig.add_subplot(2, 2, 3)
    folder_names = df["folder"].unique()
    folder_data = [df[df["folder"] == f]["reduction_pct"].values for f in folder_names]

    bp = ax3.boxplot(folder_data, labels=folder_names, patch_artist=True)
    colors_box = ["#FF9999", "#99FF99", "#9999FF", "#FFFF99"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.axhline(y=50, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Folder", fontsize=12)
    ax3.set_ylabel("Y Range Reduction (%)", fontsize=12)
    ax3.set_title("Reduction Distribution by Folder", fontsize=14, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 4. Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    # Calculate statistics
    total_files = len(df)
    successful_reductions = len(df[df["reduction_pct"] > 0])
    high_reductions = len(df[df["reduction_pct"] >= 50])
    avg_reduction = df["reduction_pct"].mean()
    median_reduction = df["reduction_pct"].median()
    max_reduction = df["reduction_pct"].max()
    min_reduction = df["reduction_pct"].min()
    avg_original = df["original_y_range"].mean()
    avg_corrected = df["corrected_y_range"].mean()

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║              PREPROCESSING SUMMARY STATISTICS                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Total Files Processed:              {total_files:>20}      ║
    ║  Files with Positive Reduction:      {successful_reductions:>20}      ║
    ║  Files with >50% Reduction:          {high_reductions:>20}      ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Average Y Range Reduction:          {avg_reduction:>17.1f}%     ║
    ║  Median Y Range Reduction:           {median_reduction:>17.1f}%     ║
    ║  Maximum Reduction:                  {max_reduction:>17.1f}%     ║
    ║  Minimum Reduction:                  {min_reduction:>17.1f}%     ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Average Original Y Range:           {avg_original:>16.2f} m     ║
    ║  Average Corrected Y Range:          {avg_corrected:>16.2f} m     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="center", horizontalalignment="center",
             fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = base_dir / "preprocessing_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary visualization to: {output_path}")

    # Also create a detailed table
    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE")
    print("=" * 80)
    print(f"{'Folder':<15} {'File ID':<15} {'Original (m)':<15} {'Corrected (m)':<15} {'Reduction (%)':<15}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['folder']:<15} {row['file_id']:<15} {row['original_y_range']:<15.2f} {row['corrected_y_range']:<15.2f} {row['reduction_pct']:<15.1f}")

    plt.show()

    return df, output_path


def create_grid_comparison():
    """Create a grid showing sample comparison images."""
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data" / "type2"

    # Collect all comparison images
    comparison_images = []
    folders = ["datatype2_processed", "datatype2 2_processed", "datatype2 3_processed", "datatype2 4_processed"]

    for folder in folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            images = sorted(folder_path.glob("*_comparison.png"))
            comparison_images.extend(images[:2])  # Take first 2 from each folder

    if len(comparison_images) < 8:
        # If not enough images, just use what we have
        comparison_images = comparison_images[:8]

    # Create grid
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for i, img_path in enumerate(comparison_images[:8]):
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(img_path.stem.replace("_comparison", ""), fontsize=10)
        axes[i].axis("off")

    # Hide unused axes
    for i in range(len(comparison_images), 8):
        axes[i].axis("off")

    plt.suptitle("Sample Comparison Results (Original vs Corrected)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = base_dir / "comparison_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison grid to: {output_path}")

    plt.show()

    return output_path


if __name__ == "__main__":
    print("Creating preprocessing summary visualization...")
    df, summary_path = create_summary_visualization()

    print("\nCreating comparison grid...")
    grid_path = create_grid_comparison()

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Summary: {summary_path}")
    print(f"Grid: {grid_path}")
