"""
Stage 2: Data Loader Module
---------------------------
Provides robust, memory-conscious functions to load the handwritten
digits and characters dataset (442,450 samples, 36 classes).
Supports loading full or balanced stratified subsets suitable for CPU training.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

def get_project_root():
    """Returns the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_class_mapping():
    """
    Returns the mapping from class ID (0 to 35) to character string:
    - 0 to 9: '0' to '9'
    - 10 to 35: 'A' to 'Z'
    """
    mapping = {i: str(i) for i in range(10)}
    for i in range(10, 36):
        mapping[i] = chr(ord('A') + (i - 10))
    return mapping

def load_data(
    csv_relative_path="dataset/digit_char_dataset.csv",
    samples_per_class=1200,
    random_state=42
):
    """
    Loads handwritten digit/character data from CSV.
    
    Parameters:
    -----------
    csv_relative_path : str
        Relative path to the CSV file from project root.
    samples_per_class : int or None
        Number of samples to draw per class.
        If None, loads all 442,450 samples (requires ~3 GB RAM).
        If set (e.g. 1200), creates a perfectly balanced dataset of 1200 * 36 = 43,200 samples,
        ideal for standard 8 GB RAM CPU systems.
    random_state : int
        Seed for reproducibility when sampling.
        
    Returns:
    --------
    X : np.ndarray
        Array of shape (N, 784), pixel values in [0, 255].
    y : np.ndarray
        Array of shape (N,), integer class labels in [0, 35].
    class_mapping : dict
        Mapping from class index to character string.
    """
    root_dir = get_project_root()
    dataset_path = os.path.join(root_dir, csv_relative_path)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {dataset_path}")
        
    print(f"[DataLoader] Reading from: {dataset_path}")
    mapping = get_class_mapping()
    
    if samples_per_class is None:
        print("[DataLoader] Loading FULL dataset (442,450 rows)...")
        # Load full CSV
        df = pd.read_csv(dataset_path)
        y = df['class'].to_numpy(dtype=np.int64)
        X = df.drop(columns=['class']).to_numpy(dtype=np.float32)
    else:
        print(f"[DataLoader] Loading balanced stratified subset ({samples_per_class} per class across 36 classes)...")
        # Stream CSV in chunks and collect samples per class
        collected = {c: [] for c in range(36)}
        remaining_needed = {c: samples_per_class for c in range(36)}
        
        chunk_size = 30000
        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
            for c in range(36):
                needed = remaining_needed[c]
                if needed > 0:
                    class_rows = chunk[chunk['class'] == c]
                    if len(class_rows) > 0:
                        take = min(needed, len(class_rows))
                        collected[c].append(class_rows.iloc[:take])
                        remaining_needed[c] -= take
            
            # Check if all classes are satisfied
            if all(rem <= 0 for rem in remaining_needed.values()):
                break
                
        # Combine collected samples
        all_dfs = []
        for c in range(36):
            if collected[c]:
                class_df = pd.concat(collected[c], ignore_index=True)
                all_dfs.append(class_df)
                
        df_balanced = pd.concat(all_dfs, ignore_index=True)
        # Shuffle with seed
        df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
        y = df_balanced['class'].to_numpy(dtype=np.int64)
        X = df_balanced.drop(columns=['class']).to_numpy(dtype=np.float32)
        
    print(f"[DataLoader] Successfully loaded data:")
    print(f"    - X shape (features): {X.shape} (N samples, 784 pixels each)")
    print(f"    - y shape (labels):   {y.shape} (N samples)")
    print(f"    - Number of classes:  {len(np.unique(y))}")
    print(f"    - Pixel range: min={X.min():.1f}, max={X.max():.1f}")
    
    return X, y, mapping

if __name__ == "__main__":
    print("Testing data_loader.py...")
    X, y, mapping = load_data(samples_per_class=500)
    print("X sample slice:", X[0, :10])
    print("y sample:", y[:10], "->", [mapping[val] for val in y[:10]])
    print("Data loader test passed successfully!")
