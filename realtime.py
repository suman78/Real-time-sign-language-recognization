import cv2
import numpy as np
import pyttsx3
import os
from tensorflow.keras.models import load_model

# ✅ Load trained model
model = load_model('mobilenet_sign_model.h5')

# ✅ Load labels
labels = sorted(os.listdir("dataSet/testingData"))

# ✅ Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 130)  # Speech speed

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))                   # Resize to match model input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # Convert BGR to RGB
    img = preprocess_input(img)                           # Normalize
    img = np.expand_dims(img, axis=0)                     # Add batch dimension
    return img

# ✅ Constants
IMG_SIZE = 224

# ✅ Start video capture
cap = cv2.VideoCapture(0)

sentence = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Region of Interest (ROI)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, "Press 'P' to Predict | 'C' to Clear | 'Q' to Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Sign Language Prediction", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
    elif key == ord('p'):
        # Process and predict on captured ROI
        if roi.size > 0:
            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi_normalized = roi_resized.astype("float32") / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=0)

            predictions = model.predict(roi_expanded, verbose=0)
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions)
            predicted_label = labels[predicted_class]

            if confidence > 0.90:
                sentence += predicted_label + " "
                print("Recognized:", predicted_label)
                engine.say(predicted_label)
                engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
