import matplotlib.pyplot as plt

# Example validation/testing accuracy of both models
model_names = ['MobileNetV2', 'ResNet50']
accuracies = [99.65, 87.37]  # Replace with your actual accuracy values

# Bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies, color=['skyblue', 'salmon'])

# Add accuracy labels above bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f'{yval:.2f}%', ha='center', fontsize=12)

plt.title('Model Accuracy Comparison', fontsize=14)
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
