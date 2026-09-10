"""
Stage 8: Handwritten Character Prediction Pipeline
--------------------------------------------------
Standalone inference script that accepts any user-provided image
(from disk or numpy array) and performs:
1. Grayscale conversion.
2. Inversion if necessary (paper handwriting is black ink on white paper,
   while MNIST/EMNIST training data is white strokes on black background).
3. Aspect-ratio preserving resize & padding to 28x28.
4. Normalization from [0, 255] to [0.0, 1.0].
5. Forward pass through trained CNN.
6. Returns predicted character, confidence percentage, and top-k probabilities.
"""

import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ensure scripts folder is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import get_project_root

def load_inference_artifacts(model_path=None, mapping_path=None):
    """Loads the trained Keras model and the class label mapping."""
    root_dir = get_project_root()
    if model_path is None:
        model_path = os.path.join(root_dir, "models", "handwritten_character_cnn.keras")
    if mapping_path is None:
        mapping_path = os.path.join(root_dir, "models", "class_mapping.json")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")
        
    model = keras.models.load_model(model_path)
    with open(mapping_path, "r") as f:
        mapping = {int(k): v for k, v in json.load(f).items()}
        
    return model, mapping

def preprocess_user_image(image_input):
    """
    Preprocesses any user-provided image (file path or numpy array)
    to match the 28x28 normalized format expected by the CNN.
    
    Handles:
    - Grayscale conversion
    - Background/foreground polarity check (auto-inverts white background)
    - Aspect-ratio preserving resize with center-padding
    - Normalization to [0.0, 1.0]
    - Reshaping to (1, 28, 28, 1)
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image file at: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("image_input must be a file path string or numpy array.")

    # 1. Grayscale conversion
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            # Blend with white background or drop alpha
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:  # BGR / RGB
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]
    else:
        gray = img.copy()

    # 2. Polarity Detection
    # If the corners/background are bright (mean > 127), it's dark ink on light background -> invert it!
    # Training data has black background (0) and white stroke (> 0).
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    if np.mean(corners) > 127 or np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # 3. Clean noise and enhance stroke contrast using Otsu or adaptive threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Aspect-ratio preserving resize to fit inside 20x20 box, then pad to 28x28
    # Find bounding box of the character stroke
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = thresh[y:y+h, x:x+w]
        
        # Scale to max 20x20 maintaining aspect ratio
        if h > w:
            scale = 20.0 / h
            new_h = 20
            new_w = max(1, int(round(w * scale)))
        else:
            scale = 20.0 / w
            new_w = 20
            new_h = max(1, int(round(h * scale)))
            
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center inside a 28x28 black canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    else:
        # Fallback direct resize
        canvas = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # 5. Normalize to [0.0, 1.0]
    tensor = (canvas / 255.0).astype(np.float32)
    tensor = np.expand_dims(tensor, axis=(0, -1))  # Shape: (1, 28, 28, 1)

    return tensor, canvas

def predict_character(image_input, model=None, mapping=None, top_k=3):
    """
    Executes prediction pipeline on an input image.
    
    Returns:
    --------
    dict with:
      - 'predicted_class': int
      - 'predicted_character': str
      - 'confidence': float (0.0 to 100.0)
      - 'top_predictions': list of tuples [(char, prob_percentage), ...]
      - 'preprocessed_image': 28x28 np.ndarray
    """
    if model is None or mapping is None:
        model, mapping = load_inference_artifacts()
        
    tensor, preprocessed_28x28 = preprocess_user_image(image_input)
    
    # Model inference
    probabilities = model.predict(tensor, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class] * 100.0)
    predicted_char = mapping.get(predicted_class, str(predicted_class))
    
    # Top-K
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        (mapping.get(int(idx), str(idx)), float(probabilities[idx] * 100.0))
        for idx in top_indices
    ]
    
    return {
        'predicted_class': predicted_class,
        'predicted_character': predicted_char,
        'confidence': confidence,
        'top_predictions': top_predictions,
        'preprocessed_image': preprocessed_28x28
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict handwritten character from an image file.")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to input image file.")
    args = parser.parse_args()
    
    root_dir = get_project_root()
    
    # If no image path is given, create a sample test image for demonstration
    test_image_path = args.image_path
    if test_image_path is None:
        print("[Predict] No image path provided. Creating a sample synthetic test image ('A')...")
        sample_dir = os.path.join(root_dir, "outputs", "test_samples")
        os.makedirs(sample_dir, exist_ok=True)
        test_image_path = os.path.join(sample_dir, "sample_A.png")
        
        # Create white background with dark letter 'A'
        sample_canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(sample_canvas, "A", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.imwrite(test_image_path, sample_canvas)
        print(f"[Predict] Sample image created at: {test_image_path}")
        
    print(f"\n[Predict] Running inference on: {test_image_path}")
    result = predict_character(test_image_path)
    
    print("\n" + "=" * 50)
    print("PREDICTION RESULT:")
    print("=" * 50)
    print(f"Predicted Character:  '{result['predicted_character']}'")
    print(f"Confidence Score:     {result['confidence']:.2f}%")
    print("\nTop Candidates:")
    for rank, (char, prob) in enumerate(result['top_predictions'], start=1):
        print(f"  {rank}. Character '{char}': {prob:.2f}%")
    print("=" * 50)
