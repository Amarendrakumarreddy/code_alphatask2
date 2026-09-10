"""
Generates quick test samples and runs batch verification.
"""
import os
import cv2
import numpy as np
from predict import predict_character, load_inference_artifacts, get_project_root

root = get_project_root()
sample_dir = os.path.join(root, "outputs", "test_samples")
os.makedirs(sample_dir, exist_ok=True)

test_chars = ['A', '5', 'Z', '0', 'B', '7']
model, mapping = load_inference_artifacts()

print("Verifying predictions across test samples:")
for char in test_chars:
    canvas = np.ones((120, 120, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, char, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.7, (15, 15, 15), 6, cv2.LINE_AA)
    img_path = os.path.join(sample_dir, f"sample_{char}.png")
    cv2.imwrite(img_path, canvas)
    
    res = predict_character(canvas, model=model, mapping=mapping)
    print(f"Target: '{char}' -> Predicted: '{res['predicted_character']}' (Confidence: {res['confidence']:.2f}%)")
