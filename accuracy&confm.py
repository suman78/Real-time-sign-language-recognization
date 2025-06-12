import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Constants
IMG_SIZE = 224
BATCH_SIZE = 32
test_dir = 'dataSet/testingData'

# ✅ Load model
model = load_model('resnet50_sign_model.h5')

# ✅ Load test data
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ✅ Predict
predictions = model.predict(test_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# ✅ Accuracy
accuracy = np.mean(y_pred == y_true)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# ✅ Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ✅ Classification Report
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ✅ Extract precision, recall, and f1-score
precisions = [report[label]["precision"] for label in class_labels]
recalls = [report[label]["recall"] for label in class_labels]
f1_scores = [report[label]["f1-score"] for label in class_labels]

x = np.arange(len(class_labels))
width = 0.25

# ✅ Plot grouped bar chart
plt.figure(figsize=(12, 6))
plt.bar(x - width, precisions, width, label='Precision')
plt.bar(x, recalls, width, label='Recall')
plt.bar(x + width, f1_scores, width, label='F1-Score')

plt.xticks(x, class_labels, rotation=45)
plt.ylim(0, 1.05)
plt.ylabel('Score')
plt.title('Classification Report Metrics per Class')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
