import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import pyttsx3

# Load your model
model = load_model('mobilenet_sign_model.h5')  # Replace with your model file

# Define class labels (adjust according to your dataset)
labels = sorted(os.listdir("dataSet/testingData"))
engine = pyttsx3.init()
engine.setProperty('rate', 130) 

# Function to preprocess the image
def preprocess_image(roi):
    roi = cv2.resize(roi, (224, 224))
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror-like view
    frame = cv2.flip(frame, 1)

    # Define a box (Region Of Interest - ROI) in center
    x1, y1, x2, y2 = 150, 100, 450, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI and predict
    input_data = preprocess_image(roi)
    prediction = model.predict(input_data)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    label = labels[class_index]

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display label and confidence
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Sign Prediction", frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
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
