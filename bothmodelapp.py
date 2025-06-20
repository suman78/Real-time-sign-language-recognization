import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit.components.v1 import html

# Load models
mobilenet_model = load_model('mobilenet_sign_model2.h5')
resnet_model = load_model('resnet50_sign_model2.h5')

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
st.set_page_config(page_title="Sign Language - Two Model Comparison", page_icon="🖐️", layout="centered")
st.markdown("""
    <div style="text-align:center">
        <h1 style="color:#3f51b5;">🖐️ Sign Language to text and speech (Two Model Comparison)</h1>
        <p style="color:gray;">Upload hand gesture images and see predictions from both MobileNet and ResNet models</p>
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

final_sentence_mobilenet = ""
final_sentence_resnet = ""

if uploaded_files:
    st.subheader("🖼️ Uploaded Images & Model Predictions")

    cols = st.columns(2)
    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        processed = preprocess_image(image)

        # MobileNet prediction
        pred_mobilenet = mobilenet_model.predict(processed)
        class_mobilenet = np.argmax(pred_mobilenet) + 1
        gesture_mobilenet = labels.get(class_mobilenet, "Unknown")
        conf_mobilenet = np.max(pred_mobilenet) * 100
        final_sentence_mobilenet += gesture_mobilenet

        # ResNet prediction
        pred_resnet = resnet_model.predict(processed)
        class_resnet = np.argmax(pred_resnet) + 1
        gesture_resnet = labels.get(class_resnet, "Unknown")
        conf_resnet = np.max(pred_resnet) * 100
        final_sentence_resnet += gesture_resnet

        with st.expander(f"📷 {file.name}"):
            st.image(image, caption="Uploaded Image", use_column_width=False, width=200)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### MobileNetV2 Prediction")
                st.markdown(f"<b>{gesture_mobilenet}</b><br>({conf_mobilenet:.2f}%)", unsafe_allow_html=True)
            with col2:
                st.markdown("#### ResNet Prediction")
                st.markdown(f"<b>{gesture_resnet}</b><br>({conf_resnet:.2f}%)", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📘 Final Sentences from Both Models")

    st.markdown(f"<h4 style='color:#4caf50;'>MobileNetV2: <b>{final_sentence_mobilenet}</b></h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:#f44336;'>ResNet: <b>{final_sentence_resnet}</b></h4>", unsafe_allow_html=True)

    # Voice Output for both
    html(f"""
        <script>
            var msg1 = new SpeechSynthesisUtterance("MobileNet prediction: {final_sentence_mobilenet}");
            var msg2 = new SpeechSynthesisUtterance("ResNet prediction: {final_sentence_resnet}");
            msg1.lang = 'en-US'; msg2.lang = 'en-US';
            msg1.pitch = 1; msg1.rate = 1; msg1.volume = 1;
            msg2.pitch = 1; msg2.rate = 1; msg2.volume = 1;
            window.speechSynthesis.speak(msg1);
            window.speechSynthesis.speak(msg2);
        </script>
    """)
else:
    st.info("📎 Please upload hand gesture images to see predictions.")
