import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report

# ✅ Constants
IMG_SIZE = 224
BATCH_SIZE = 32
test_dir = "dataSet/testingData"

# ✅ Load test data
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ✅ Model paths
models_info = {
    "MobileNetV2": "mobilenet_sign_model.h5",
    "ResNet50": "resnet50_sign_model.h5"
}

# ✅ Store comparison results
accuracies = {}
reports = {}

# ✅ Evaluate each model
for name, path in models_info.items():
    print(f"\n📦 Loading model: {name}")
    model = load_model(path)
    
    print("🔍 Predicting...")
    y_pred = np.argmax(model.predict(test_data, verbose=1), axis=1)
    y_true = test_data.classes

    acc = accuracy_score(y_true, y_pred)
    accuracies[name] = acc * 100  # convert to percentage

    print(f"✅ {name} Accuracy: {acc*100:.2f}%")
    reports[name] = classification_report(y_true, y_pred, target_names=test_data.class_indices.keys())

# ✅ Plot accuracy comparison
plt.figure(figsize=(6, 4))
plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'lightgreen'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# ✅ Print classification reports
for name, report in reports.items():
    print(f"\n📊 Classification Report for {name}:\n{report}")
