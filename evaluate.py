"""
Stage 7: Model Evaluation Script
--------------------------------
Evaluates the trained CNN on the unseen test dataset:
1. Calculates Test Loss and Test Accuracy.
2. Computes and prints Scikit-learn Classification Report (Precision, Recall, F1-Score).
3. Generates and saves:
   - outputs/learning_curves.png: Training vs. Validation Accuracy and Loss.
   - outputs/confusion_matrix.png: 36x36 Confusion Matrix Heatmap.
   - outputs/evaluation_metrics.json: Summary metrics.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Ensure scripts folder is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import get_project_root
from preprocessing import prepare_and_cache_dataset

def plot_learning_curves(history_path, output_dir):
    """Plots and saves training vs validation accuracy and loss."""
    if not os.path.exists(history_path):
        print(f"[Evaluation] Warning: History file not found at {history_path}. Skipping curve plot.")
        return
        
    with open(history_path, "r") as f:
        history = json.load(f)
        
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy curve
    ax1.plot(epochs, history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 'r--s', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=10)
    
    # Loss curve
    ax2.plot(epochs, history['loss'], 'b-o', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r--s', label='Validation Loss', linewidth=2)
    ax2.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"[Evaluation] Learning curves saved to: {plot_file}")

def plot_confusion_matrix_heatmap(y_true, y_pred, mapping, output_dir):
    """Plots and saves a 36x36 confusion matrix heatmap."""
    labels = [mapping[i] for i in sorted(mapping.keys())]
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=False, cmap='Blues', fmt='d',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Sample Count'}
    )
    plt.title("Handwritten Character Recognition - Confusion Matrix (36 Classes)", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Predicted Character", fontsize=12)
    plt.ylabel("True Character", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plot_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Confusion matrix saved to: {plot_file}")

def evaluate_trained_model():
    root_dir = get_project_root()
    models_dir = os.path.join(root_dir, "models")
    outputs_dir = os.path.join(root_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "handwritten_character_cnn.keras")
    mapping_path = os.path.join(models_dir, "class_mapping.json")
    history_path = os.path.join(outputs_dir, "training_history.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
        
    print("=" * 70)
    print("STAGE 7: MODEL EVALUATION ON UNSEEN TEST DATA")
    print("=" * 70)
    
    # 1. Load data
    _, _, (X_test, y_test), mapping = prepare_and_cache_dataset()
    
    # 2. Load trained model
    print(f"\n[Evaluation] Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # 3. Evaluate Loss & Accuracy
    print("\n[Evaluation] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n>>> Test Loss:     {test_loss:.4f}")
    print(f">>> Test Accuracy: {test_accuracy * 100:.2f}%\n")
    
    # 4. Predictions
    print("[Evaluation] Generating model predictions on test samples...")
    y_prob = model.predict(X_test, batch_size=128, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    
    # 5. Classification Report
    target_names = [f"'{mapping[i]}'" for i in sorted(mapping.keys())]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Precision, Recall, F1-Score):")
    print("=" * 70)
    print(report)
    print("=" * 70)
    
    # 6. Save Metrics to JSON
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "total_test_samples": int(len(y_test))
    }
    metrics_file = os.path.join(outputs_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[Evaluation] Metrics saved to: {metrics_file}")
    
    # 7. Plot Graphs
    plot_learning_curves(history_path, outputs_dir)
    plot_confusion_matrix_heatmap(y_test, y_pred, mapping, outputs_dir)
    
    print("\n" + "=" * 70)
    print("STAGE 7 EVALUATION COMPLETE: All graphs and reports generated successfully.")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_trained_model()
