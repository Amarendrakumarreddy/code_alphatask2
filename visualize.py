"""
Stage 3: Data Exploration and Visualization Script
--------------------------------------------------
Explores and visualizes the handwritten digit & character dataset:
1. Displays what X (pixel intensity features) and y (target class labels) represent.
2. Generates and saves:
   - outputs/sample_characters.png: 6x6 grid showing an example of every single class (0-9, A-Z).
   - outputs/class_distribution.png: Frequency distribution of all 36 classes.
   - outputs/before_after_preprocessing.png: Raw [0, 255] vs Normalized [0, 1] comparisons.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for clean automated saving
import matplotlib.pyplot as plt

# Ensure scripts folder is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import get_project_root, get_class_mapping, load_data

def explain_dataset(X, y):
    """Prints a clear explanation of features (X) and labels (y)."""
    print("\n" + "=" * 70)
    print("DATASET REPRESENTATION EXPLANATION:")
    print("=" * 70)
    print("• X (Feature Matrix):")
    print(f"  - Shape: {X.shape}")
    print("  - Representation: Each row represents one 28x28 grayscale handwritten image.")
    print("  - Total pixel columns: 784 (28 rows * 28 columns = 784 pixels).")
    print(f"  - Values: Pixel brightness ranging from {X.min():.0f} (background/black) to {X.max():.0f} (stroke/white).")
    print("\n• y (Target Labels):")
    print(f"  - Shape: {y.shape}")
    print(f"  - Representation: Integer class identifier from 0 to 35.")
    print("  - 0 to 9   -> Digits '0' through '9' (10 classes)")
    print("  - 10 to 35 -> Uppercase English alphabets 'A' through 'Z' (26 classes)")
    print("=" * 70 + "\n")

def plot_sample_grid(X, y, mapping, output_dir):
    """Plots a 6x6 grid showcasing a sample from all 36 classes."""
    print("[Visualization] Generating 6x6 grid of sample characters across all 36 classes...")
    fig, axes = plt.subplots(6, 6, figsize=(10, 10))
    fig.suptitle("Sample Handwritten Characters (36 Classes: 0-9, A-Z)", fontsize=16, fontweight='bold', y=0.95)
    
    unique_classes = sorted(list(mapping.keys()))
    for idx, c in enumerate(unique_classes):
        row_idx = idx // 6
        col_idx = idx % 6
        ax = axes[row_idx, col_idx]
        
        # Find first sample matching class c
        matches = np.where(y == c)[0]
        if len(matches) > 0:
            img = X[matches[0]].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"'{mapping[c]}' (ID: {c})", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, "sample_characters.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved: {output_path}")

def plot_class_distribution(csv_path, mapping, output_dir):
    """Plots the distribution of classes across the dataset."""
    print("[Visualization] Computing and plotting class distribution...")
    # Fast scan of just the class column
    class_series = pd.read_csv(csv_path, usecols=['class'])['class']
    counts = class_series.value_counts().sort_index()
    
    char_labels = [mapping.get(c, str(c)) for c in counts.index]
    
    plt.figure(figsize=(14, 6))
    colors = ['#2b5c8f' if c < 10 else '#3b925f' for c in counts.index]
    bars = plt.bar(char_labels, counts.values, color=colors, edgecolor='black', alpha=0.85)
    
    plt.title("Handwritten Character Class Distribution (Digits 0-9 vs Alphabets A-Z)", fontsize=14, fontweight='bold')
    plt.xlabel("Character Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2b5c8f', edgecolor='black', label='Digits (0-9)'),
        Patch(facecolor='#3b925f', edgecolor='black', label='Alphabets (A-Z)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    output_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved: {output_path}")

def plot_preprocessing_comparison(X, y, mapping, output_dir):
    """Compares raw image with normalized preprocessed tensor."""
    print("[Visualization] Generating before/after preprocessing comparison...")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Image Representation: Before Preprocessing vs After Normalization", fontsize=14, fontweight='bold')
    
    sample_indices = [0, 1, 2, 3]
    for i, idx in enumerate(sample_indices):
        raw_img = X[idx].reshape(28, 28)
        norm_img = raw_img / 255.0
        label_char = mapping[y[idx]]
        
        # Raw image
        ax_raw = axes[0, i]
        im0 = ax_raw.imshow(raw_img, cmap='gray', vmin=0, vmax=255)
        ax_raw.set_title(f"Raw '{label_char}'\nMin={raw_img.min():.0f}, Max={raw_img.max():.0f}", fontsize=10)
        ax_raw.axis('off')
        
        # Normalized image
        ax_norm = axes[1, i]
        im1 = ax_norm.imshow(norm_img, cmap='viridis', vmin=0.0, vmax=1.0)
        ax_norm.set_title(f"Normalized (0-1)\nMin={norm_img.min():.2f}, Max={norm_img.max():.2f}", fontsize=10)
        ax_norm.axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, "before_after_preprocessing.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved: {output_path}")

def run_exploration():
    root_dir = get_project_root()
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(root_dir, "dataset", "digit_char_dataset.csv")
    
    # Load representative balanced sample for visualization
    X, y, mapping = load_data(samples_per_class=100)
    
    explain_dataset(X, y)
    plot_sample_grid(X, y, mapping, output_dir)
    plot_class_distribution(csv_path, mapping, output_dir)
    plot_preprocessing_comparison(X, y, mapping, output_dir)
    
    print("\n[Visualization] All exploratory plots generated and saved in outputs/ directory.")

if __name__ == "__main__":
    run_exploration()
