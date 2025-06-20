import matplotlib.pyplot as plt

# Epochs
epochs_model1 = [0, 1, 2, 3]
epochs_model2 = [0, 1]

# Accuracy values
train_acc_model1 = [0.957, 0.995, 0.994, 0.996]
val_acc_model1   = [0.998, 0.998, 0.999, 0.999]

train_acc_model2 = [0.37, 0.57]
val_acc_model2   = [0.78, 0.86]

# Plot
plt.figure(figsize=(10, 6))

# Model 1 (MobileNetV2)
plt.plot(epochs_model1, train_acc_model1, 'o-', label='MobileNetV2 - Train Accuracy')
plt.plot(epochs_model1, val_acc_model1, 'x-', label='MobileNetV2 - Validation Accuracy')

# Model 2 (ResNet50)
plt.plot(epochs_model2, train_acc_model2, 'o--', label='ResNet50 - Train Accuracy')
plt.plot(epochs_model2, val_acc_model2, 'x--', label='ResNet50 - Validation Accuracy')

plt.title('Training and Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.3, 1.01)
plt.legend()
plt.grid(True)
plt.show()
