import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Constants ===
IMG_SIZE = 224
BATCH_SIZE = 32
TEST_DIR = 'testingdataset'  # Ensure path is correct
MODEL_PATH = 'mobilenet_sign_model2.h5'

# === Load Model ===
model = load_model(MODEL_PATH)

# === Load Test Data ===
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# === Predict ===
predictions = model.predict(test_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# === Accuracy ===
accuracy = np.mean(y_pred == y_true)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)

# === Plot Confusion Matrix ===
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# === Classification Report ===
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# === Extract Metrics ===
precisions = [report[label]["precision"] for label in class_labels]
recalls = [report[label]["recall"] for label in class_labels]
f1_scores = [report[label]["f1-score"] for label in class_labels]

x = np.arange(len(class_labels))
width = 0.25

# === Plot Precision, Recall, F1-Score ===
plt.figure(figsize=(12, 6))
plt.bar(x - width, precisions, width, label='Precision')
plt.bar(x, recalls, width, label='Recall')
plt.bar(x + width, f1_scores, width, label='F1-Score')

plt.xticks(x, class_labels, rotation=45)
plt.ylim(0, 1.05)
plt.ylabel('Score')
plt.title('Classification Metrics per Class')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
