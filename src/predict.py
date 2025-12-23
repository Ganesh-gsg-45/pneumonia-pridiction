import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import sys
def predict_pneumonia(image_path):
    model=tf.keras.models.load_model('models/pneumonia_model.h5')
    img = Image.open(image_path).convert('L')  # Always convert to grayscale
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        print(f"Prediction: PNEUMONIA (Confidence: {prediction:.2%})")
    else:
        print(f"Prediction: NORMAL (Confidence: {(1-prediction):.2%})")
    
    return prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_pneumonia(sys.argv[1])
