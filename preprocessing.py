"""
Stage 4: Data Preprocessing Module
----------------------------------
Preprocesses handwritten character images for CNN training:
1. Normalizes pixel values from [0, 255] to [0.0, 1.0].
2. Reshapes 1D flat (784,) vectors to 4D CNN tensor format: (N, 28, 28, 1).
3. Performs stratified Train-Validation-Test splitting (e.g., 70% train, 15% val, 15% test).
4. Saves class label mapping to models/class_mapping.json.
5. Saves preprocessed arrays to dataset/preprocessed_data.npz for fast instant loading.
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure scripts folder is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import get_project_root, get_class_mapping, load_data

def preprocess_images(X):
    """
    Normalizes pixel intensities to [0.0, 1.0] and reshapes to (N, 28, 28, 1).
    """
    print("[Preprocessing] Normalizing pixels to [0.0, 1.0] and reshaping to (N, 28, 28, 1)...")
    X_norm = (X / 255.0).astype(np.float32)
    X_reshaped = X_norm.reshape(-1, 28, 28, 1)
    return X_reshaped

def split_dataset(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits data into stratified train, validation, and test sets.
    """
    print(f"[Preprocessing] Splitting dataset: {1 - test_size - val_size:.0%} train, {val_size:.0%} val, {test_size:.0%} test...")
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Next split validation set from train_val
    adjusted_val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_ratio, random_state=random_state, stratify=y_train_val
    )
    
    print(f"    - Training set:   X={X_train.shape}, y={y_train.shape}")
    print(f"    - Validation set: X={X_val.shape}, y={y_val.shape}")
    print(f"    - Test set:       X={X_test.shape}, y={y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_class_mapping(mapping, filepath=None):
    """Saves class mapping dictionary to JSON."""
    if filepath is None:
        root_dir = get_project_root()
        filepath = os.path.join(root_dir, "models", "class_mapping.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        # Convert integer keys to str for JSON serialization
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=4)
    print(f"[Preprocessing] Saved class mapping to: {filepath}")

def prepare_and_cache_dataset(samples_per_class=1200, force_reload=False):
    """
    Loads, preprocesses, and caches data to disk.
    If preprocessed_data.npz exists and force_reload=False, loads directly from disk.
    """
    root_dir = get_project_root()
    cache_file = os.path.join(root_dir, "dataset", "preprocessed_data.npz")
    mapping_file = os.path.join(root_dir, "models", "class_mapping.json")
    
    if os.path.exists(cache_file) and not force_reload:
        print(f"[Preprocessing] Found cached preprocessed data at: {cache_file}")
        data = np.load(cache_file)
        with open(mapping_file, "r") as f:
            mapping = {int(k): v for k, v in json.load(f).items()}
        return (
            (data['X_train'], data['y_train']),
            (data['X_val'], data['y_val']),
            (data['X_test'], data['y_test']),
            mapping
        )
        
    print("[Preprocessing] Preparing fresh dataset...")
    X_raw, y_raw, mapping = load_data(samples_per_class=samples_per_class)
    X_processed = preprocess_images(X_raw)
    
    train_data, val_data, test_data = split_dataset(X_processed, y_raw)
    save_class_mapping(mapping, mapping_file)
    
    print(f"[Preprocessing] Caching preprocessed arrays to: {cache_file}")
    np.savez_compressed(
        cache_file,
        X_train=train_data[0], y_train=train_data[1],
        X_val=val_data[0], y_val=val_data[1],
        X_test=test_data[0], y_test=test_data[1]
    )
    print("[Preprocessing] Caching complete.")
    
    return train_data, val_data, test_data, mapping

if __name__ == "__main__":
    print("Testing preprocessing.py...")
    # Prepare a fast test set of 200 samples per class
    prepare_and_cache_dataset(samples_per_class=200, force_reload=True)
    print("Preprocessing test completed successfully!")
