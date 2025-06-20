# 🤟 Real-Time Sign Language Recognition System

Welcome to the official repository of the **Real-Time Sign Language Recognition System** – an AI-powered application that bridges the communication gap between hearing-impaired individuals and the world using Deep Learning and Computer Vision.

## 🌟 Project Highlights

- 🔍 **Real-Time Recognition** of hand gestures representing numbers (0-9) and alphabets (A-B)
- 🧠 **Deep Learning Models:** MobileNetV2 / ResNet50 for efficient and accurate gesture classification
- 🖼️ **Custom Dataset** with image preprocessing (CLAHE, gamma correction, grayscale, etc.)
- 🎥 Built with **OpenCV** for live video stream processing
- 🛠️ Frameworks Used: TensorFlow, Keras, Streamlit (for UI)

---

## 📽️ Demo Preview

> *Coming Soon: GIF / video demo here showing real-time prediction!*

---

## 🛠️ Tech Stack

| Tool        | Purpose                             |
|-------------|-------------------------------------|
| Python      | Core programming language           |
| TensorFlow  | Deep learning model development     |
| Keras       | High-level neural network API       |
| OpenCV      | Real-time video processing          |
| Streamlit   | Web-based application interface     |
| NumPy       | Numerical operations                |
| Matplotlib  | Visualization of model performance  |

---

## ⚙️ How It Works

1. **Dataset Preparation:** Images are preprocessed using:
   - Grayscale conversion
   - Gamma correction
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Gaussian Blur
   - Image resizing, normalization, and augmentation

2. **Model Training:** Trained MobileNetV2 / ResNet50 on custom gesture dataset.

3. **Real-Time Prediction:** OpenCV captures hand gestures from webcam and passes them to the model for prediction.

4. **Display Output:** Detected gesture is shown in real-time with prediction confidence.

---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
pip install -r requirements.txt
