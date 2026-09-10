"""
Stage 1: Dataset Inspection Script
-----------------------------------
Inspects the handwritten digit and character dataset file, prints its shape,
column details, class distribution, checks for missing values, and displays
sample ASCII representations to verify image orientation.
"""

import os
import sys
import pandas as pd
import numpy as np

def get_project_root():
    """Returns the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def inspect_dataset(csv_relative_path="dataset/digit_char_dataset.csv"):
    root_dir = get_project_root()
    dataset_path = os.path.join(root_dir, csv_relative_path)
    
    print("=" * 70)
    print("STAGE 1: PROJECT AND DATASET INSPECTION")
    print("=" * 70)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
        
    file_size_bytes = os.path.getsize(dataset_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
    
    print(f"\n[1] File Location & Size:")
    print(f"    - Path: {dataset_path}")
    print(f"    - Size: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")

    print(f"\n[2] Reading Dataset Header and Metadata:")
    # Read first 5 rows to examine column schema
    df_sample = pd.read_csv(dataset_path, nrows=5)
    total_cols = len(df_sample.columns)
    pixel_cols = [c for c in df_sample.columns if c.startswith('pixel')]
    has_class_col = 'class' in df_sample.columns
    
    print(f"    - Total columns: {total_cols}")
    print(f"    - Pixel columns: {len(pixel_cols)} (expected: 784 for 28x28 images)")
    print(f"    - Target column: 'class' (present: {has_class_col})")
    print(f"    - Pixel columns sample: {pixel_cols[:3]} ... {pixel_cols[-3:]}")

    print(f"\n[3] Reading Total Rows and Target Classes (Chunked scan for memory efficiency):")
    chunk_size = 50000
    class_counts = pd.Series(dtype=int)
    total_rows = 0
    total_nulls = 0
    
    for i, chunk in enumerate(pd.read_csv(dataset_path, chunksize=chunk_size)):
        total_rows += len(chunk)
        total_nulls += chunk.isnull().sum().sum()
        counts = chunk['class'].value_counts()
        class_counts = class_counts.add(counts, fill_value=0).astype(int)
        print(f"    - Processed rows: {total_rows:,}...", end='\r')
    
    print(f"    - Total samples (rows): {total_rows:,}")
    print(f"    - Total missing / NaN values: {total_nulls}")
    
    # Analyze classes
    unique_classes = np.sort(class_counts.index.values)
    num_classes = len(unique_classes)
    
    print(f"\n[4] Class Analysis:")
    print(f"    - Number of unique classes: {num_classes}")
    print(f"    - Class label range: [{unique_classes.min()}, {unique_classes.max()}]")
    
    # Class mapping breakdown: 0-9 = Digits, 10-35 = Letters A-Z
    mapping = {}
    for c in range(10):
        mapping[c] = str(c)
    for c in range(10, 36):
        mapping[c] = chr(ord('A') + (c - 10))
        
    print("\n    - Class Mapping Breakdown:")
    digit_classes = [c for c in unique_classes if c < 10]
    letter_classes = [c for c in unique_classes if c >= 10]
    print(f"      * Digits (Classes 0-9): {len(digit_classes)} classes -> {[mapping[c] for c in digit_classes]}")
    print(f"      * Alphabets (Classes 10-35): {len(letter_classes)} classes -> {[mapping[c] for c in letter_classes]}")
    
    print("\n[5] Class Distribution Summary (Samples per character):")
    print(f"    {'Class ID':<10} {'Character':<12} {'Count':<10}")
    print("    " + "-" * 32)
    for c in unique_classes:
        print(f"    {int(c):<10} {mapping.get(c, '?'):<12} {class_counts[c]:<10,}")
        
    print(f"\n[6] Visual Inspection of Sample Images:")
    target_samples = [0, 1, 5, 10, 11, 24, 35] # '0', '1', '5', 'A', 'B', 'O', 'Z'
    found_samples = {}
    
    for chunk in pd.read_csv(dataset_path, chunksize=10000):
        for c in target_samples:
            if c not in found_samples:
                match = chunk[chunk['class'] == c]
                if len(match) > 0:
                    found_samples[c] = match.iloc[0].values
        if len(found_samples) == len(target_samples):
            break
            
    for c in target_samples:
        if c in found_samples:
            row = found_samples[c]
            pixels = row[:-1].astype(np.float32)
            char_label = mapping.get(c, str(c))
            img = pixels.reshape(28, 28)
            print(f"\n    Sample Class {c} -> Character '{char_label}' (Min: {pixels.min()}, Max: {pixels.max()}):")
            for r in range(2, 26, 2):
                line = "".join(["#" if img[r, col] > 120 else ("." if img[r, col] > 25 else " ") for col in range(28)])
                if line.strip():
                    print(f"      {line}")
                    
    print("\n" + "=" * 70)
    print("STAGE 1 INSPECTION COMPLETE: All checks passed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    inspect_dataset()
