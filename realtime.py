import cv2
import numpy as np
import pyttsx3
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 📌 Constants
IMG_SIZE = 224
ROI_TOP_LEFT = (100, 100)
ROI_BOTTOM_RIGHT = (324, 324)

# ✅ Load Trained Model
model = load_model('mobilenet_sign_model.h5')

# ✅ Load Class Labels
labels = sorted(['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
# ✅ Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 130)

# 🧼 Preprocessing Function
def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# 🎥 Initialize Webcam
cap = cv2.VideoCapture(0)
sentence = ""

print("🔴 Real-time Sign Language Recognition Started")
print("➡ Press 'P' to Predict | 'C' to Clear | 'Q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 🔳 Draw ROI Box
    x1, y1 = ROI_TOP_LEFT
    x2, y2 = ROI_BOTTOM_RIGHT
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)

    # 🖊 Display instructions and sentence
    cv2.putText(frame, "Press 'P' to Predict | 'C' to Clear | 'Q' to Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 📷 Show Frame
    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        print("❌ Quitting application.")
        break

    elif key == ord('c'):
        print("🔄 Sentence cleared.")
        sentence = ""

    elif key == ord('p'):
        if roi.size > 0:
            processed = preprocess_frame(roi)
            predictions = model.predict(processed, verbose=0)
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions)
            predicted_label = labels[predicted_class]

            if confidence > 0.90:
                sentence += predicted_label + " "
                print(f"✅ Recognized: {predicted_label} ({confidence:.2f})")
                engine.say(predicted_label)
                engine.runAndWait()
            else:
                print(f"⚠️ Low confidence: {confidence:.2f} – No character added.")

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()
