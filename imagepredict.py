import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("mobilenet_sign_model.h5")

# Define the class labels (0-9 and a-z)
class_names = [
    '0','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v', 'w', 'x', 'y', 'z'
]

# Prediction function
def predict_image(image_path):
    IMG_SIZE = 224  # Should match the training input size
    
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found or unreadable:", image_path)
        return

    # Preprocess the image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict using the model
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    print(f"✅ Predicted Class: {class_names[class_index]}")
    print(f"🔍 Confidence: {confidence * 100:.2f}%")

# Example usage
predict_image("trainingdataset/D/0_aug1.jpg")  # Replace with your image path
