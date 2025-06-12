import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image

# Define paths
original_dataset_dir = "dataSet/testingData"  # Change to your actual path
augmented_dataset_dir = "tesingdataset"
os.makedirs(augmented_dataset_dir, exist_ok=True)

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

# Set number of augmented images per input image
AUGMENT_COUNT = 5

# Loop through each class folder
for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    save_path = os.path.join(augmented_dataset_dir, class_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Processing class '{class_name}'...")

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        try:
            # Load and preprocess
            img = load_img(image_path, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Generate augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1):
                augmented_img = batch[0].astype('uint8')
                out = Image.fromarray(augmented_img)
                out.save(os.path.join(save_path, f"{os.path.splitext(image_name)[0]}_aug{i}.jpg"))
                i += 1
                if i >= AUGMENT_COUNT:
                    break
        except Exception as e:
            print(f"Failed on image {image_path}: {e}")

print("✅ Augmentation complete.")
