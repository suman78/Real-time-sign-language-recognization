import os
import cv2
import random
import shutil

# Source directory with preprocessed images
SOURCE_DIR = 'preprocessed_dataset'
TRAIN_DIR = 'data_split/train'
TEST_DIR = 'data_split/test'

# Split ratio
train_ratio = 0.8

# Create output dirs
for folder in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Loop through each class folder
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)

        # Calculate split index
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Create class folders in train/test
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

        # Move images to train folder
        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(TRAIN_DIR, class_name, img)
            shutil.copy(src, dst)

        # Move images to test folder
        for img in test_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(TEST_DIR, class_name, img)
            shutil.copy(src, dst)

print("✅ Dataset split into training and testing sets.")
