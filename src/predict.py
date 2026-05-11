import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

# Resolve model path relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pneumonia_model_best.h5")

def predict_pneumonia(image_path):
    """Load the best trained model and predict pneumonia from a chest X-ray."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None

    model = tf.keras.models.load_model(MODEL_PATH)
    img = Image.open(image_path).convert('L')  # Convert to grayscale (chest X-rays)
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)

    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction > 0.5:
        confidence = prediction * 100
        print(f"Prediction: PNEUMONIA (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1 - prediction) * 100
        print(f"Prediction: NORMAL (Confidence: {confidence:.2f}%)")

    return prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)
    else:
        predict_pneumonia(sys.argv[1])
