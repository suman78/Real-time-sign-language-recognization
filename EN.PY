import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt

# ✅ Constants
IMG_SIZE = 300  # EfficientNetB3 expects 300x300 images
BATCH_SIZE = 32
EPOCHS = 10
train_dir = 'dataSet/testingData'
val_dir = 'dataSet/trainingData'

# ✅ Data Preparation (rescale and augment)
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ✅ Load EfficientNetB3 base
base_model = efn.EfficientNetB3(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # freeze pretrained layers

# ✅ Add custom head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# ✅ Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Optional callback
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# ✅ Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ✅ Save model
model.save("efficientnetb3_sign_model.h5")

# ✅ Evaluate
loss, acc = model.evaluate(val_data)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")

# ✅ Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('EfficientNetB3 Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
