import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit.components.v1 import html

# Load model
model = load_model('mobilenet_sign_model2.h5')

# Class labels
labels = {
    1: '0', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E',
    7: 'F', 8: 'G', 9: 'H', 10: 'I', 11: 'J',
    12: 'K', 13: 'L', 14: 'M', 15: 'N', 16: 'O',
    17: 'P', 18: 'Q', 19: 'R', 20: 'S', 21: 'T',
    22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y',
    27: 'Z'
}

# Page config
st.set_page_config(page_title="Sign Language to Sentence", page_icon="🖐️" ,layout="centered")
st.markdown("""
    <div style="text-align:center">
        <h1 style="color:#3f51b5;">🖐️ Sign Language to Sentence Generator</h1>
        <p style="color:gray;">Upload hand gesture images (in correct order) to form a sentence and hear it aloud</p>
    </div>
""", unsafe_allow_html=True)

# Preprocessing function
def preprocess_image(image_pil):
    image = np.array(image_pil)

    if len(image.shape) == 2:
        pass
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.stack((image,) * 3, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image

# File uploader
uploaded_files = st.file_uploader("📤 Upload Multiple Hand Gesture Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

final_sentence = ""

if uploaded_files:
    st.subheader("🖼️ Uploaded Images & Predictions")

    cols = st.columns(3)  # 3 images per row
    for i, file in enumerate(uploaded_files):
        col = cols[i % 3]

        with col:
            image = Image.open(file)
            st.image(image, caption=file.name, use_column_width=False, width=150)

            processed = preprocess_image(image)
            prediction = model.predict(processed)
            predicted_class = np.argmax(prediction) + 1
            gesture = labels.get(predicted_class, "Unknown")
            confidence = np.max(prediction) * 100

            st.markdown(f"""
                <div style='background:#f0f0f0; padding:5px; border-radius:8px; text-align:center; font-size:14px;'>
                <b>{gesture}</b><br>({confidence:.2f}%)
                </div>
            """, unsafe_allow_html=True)

            final_sentence += gesture

    st.markdown("---")
    st.subheader("📘 Final Sentence:")
    st.markdown(f"<h2 style='color:#3f51b5; text-align:center'>{final_sentence}</h2>", unsafe_allow_html=True)

    # Speak the sentence
    html(f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{final_sentence}");
        msg.lang = 'en-US';
        msg.pitch = 1;
        msg.rate = 1;
        msg.volume = 1;
        window.speechSynthesis.speak(msg);
    </script>
    """)
else:
    st.info("📎 Please upload hand gesture images to start prediction.")
