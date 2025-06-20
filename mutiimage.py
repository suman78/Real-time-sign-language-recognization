import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('mobilenet_sign_model.h5')  # Ensure the model is in the same folder or provide full path

# Label mapping
labels = {
     1:'0', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E',
    7: 'F', 8: 'G', 9: 'H', 10: 'I', 11: 'J',
    12: 'K', 13: 'L', 14: 'M', 15: 'N', 16: 'O',
    17: 'P', 18: 'Q', 19: 'R', 20: 'S', 21: 'T',
    22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y',
    27: 'Z'
}

# Streamlit UI setup
st.set_page_config(page_title="Sign Language Recognition", page_icon="🖐️", layout="centered")
st.markdown("""
    <h1 style="text-align:center; color:#3f51b5;">🖐️ Sign Language Gesture Recognition</h1>
    <h4 style="text-align:center; color:gray;">Upload one or more hand gesture images to get predictions</h4>
    <br>
""", unsafe_allow_html=True)

# Preprocessing function
def preprocess_image(image_pil):
    image = np.array(image_pil)

    # Handle different image modes
    if len(image.shape) == 2:
        pass  # already grayscale
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0

    # MobileNet requires 3 channels
    image = np.stack((image,) * 3, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image

# Upload section
uploaded_files = st.file_uploader("📤 Upload Hand Gesture Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=f"🖼️ {file.name}", use_column_width=True)

        with st.spinner(f"🔍 Predicting for {file.name}..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction) + 1
            confidence = np.max(prediction) * 100
            gesture = labels.get(predicted_class, "Unknown")

        # Display result
        st.markdown(f"""
            <div style='background-color:#f5f5f5;padding:12px;border-radius:10px;margin-bottom:20px;'>
                <h4>✅ <b>Predicted Gesture:</b> <span style="color:#3f51b5;">{gesture}</span></h4>
                <p>🔍 <b>Confidence:</b> <span style="color:#4caf50;">{confidence:.2f}%</span></p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("📎 Upload one or more gesture images to start predicting.")

