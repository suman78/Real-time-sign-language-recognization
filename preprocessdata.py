import os
import cv2
import numpy as np

# Input and output directories
input_root = 'asl_dataset'           # Folder containing class-wise subfolders
output_root = 'preprocessed_dataset' # Output folder

# Create output root if not exists
os.makedirs(output_root, exist_ok=True)

# Target size for resized images
target_size = (224, 224)

# Gamma correction function
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Preprocessing pipeline
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma_corrected = adjust_gamma(gray, gamma=1.5)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gamma_corrected)

    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    resized = cv2.resize(blurred, target_size)
    normalized = (resized / 255.0 * 255).astype(np.uint8)  # Normalize then scale back to uint8 for saving

    return normalized

# Process all images class-wise
for class_name in os.listdir(input_root):
    class_input_path = os.path.join(input_root, class_name)
    class_output_path = os.path.join(output_root, class_name)

    if os.path.isdir(class_input_path):
        os.makedirs(class_output_path, exist_ok=True)

        for filename in os.listdir(class_input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_img_path = os.path.join(class_input_path, filename)
                output_img_path = os.path.join(class_output_path, filename)

                processed_img = preprocess_image(input_img_path)

                if processed_img is not None:
                    cv2.imwrite(output_img_path, processed_img)

print("✅ Preprocessing completed and saved in:", output_root)
