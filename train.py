"""
Stage 6: Model Training Script
------------------------------
Trains the CNN model on the handwritten character dataset.
Features:
- Uses training and validation sets.
- EarlyStopping and ModelCheckpoint callbacks.
- Saves best weights to models/handwritten_character_cnn.keras.
- Saves training history to outputs/history.json for evaluation plotting.
- Highly optimized for 8 GB RAM CPU environments.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ensure scripts folder is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import get_project_root
from preprocessing import prepare_and_cache_dataset
from model import build_character_cnn

def train_character_model(
    samples_per_class=1000,
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    force_reload=False
):
    """
    Orchestrates data loading, model compilation, training with callbacks,
    and checkpoint saving.
    """
    root_dir = get_project_root()
    models_dir = os.path.join(root_dir, "models")
    outputs_dir = os.path.join(root_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "handwritten_character_cnn.keras")
    history_save_path = os.path.join(outputs_dir, "training_history.json")
    
    print("=" * 70)
    print("STAGE 6: MODEL TRAINING")
    print("=" * 70)
    print(f"Configurations:")
    print(f"  - Samples per class: {samples_per_class} (Total: ~{samples_per_class * 36:,})")
    print(f"  - Epochs:            {epochs}")
    print(f"  - Batch size:        {batch_size}")
    print(f"  - Model save path:   {model_save_path}")
    print("=" * 70)
    
    # 1. Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), mapping = prepare_and_cache_dataset(
        samples_per_class=samples_per_class,
        force_reload=force_reload
    )
    
    num_classes = len(mapping)
    print(f"\n[Training] Total Classes: {num_classes}")
    print(f"[Training] Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}, Test samples: {len(X_test):,}")
    
    # 2. Build CNN model
    print("\n[Training] Building CNN architecture...")
    model = build_character_cnn(input_shape=(28, 28, 1), num_classes=num_classes, learning_rate=learning_rate)
    model.summary()
    
    # 3. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    # 4. Train Model
    print(f"\n[Training] Starting model training for {epochs} epochs on CPU...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Save History
    # Convert numpy floats in history to standard Python floats for JSON serialization
    hist_dict = {}
    for k, v in history.history.items():
        hist_dict[k] = [float(x) for x in v]
        
    with open(history_save_path, "w") as f:
        json.dump(hist_dict, f, indent=4)
        
    print(f"\n[Training] Training history saved to: {history_save_path}")
    print(f"[Training] Best model checkpoint saved to: {model_save_path}")
    print("=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    
    return model, hist_dict

if __name__ == "__main__":
    train_character_model(samples_per_class=800, epochs=8, batch_size=64)
