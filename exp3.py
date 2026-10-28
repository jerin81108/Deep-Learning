"""
Ex. No: 3 - FACE RECOGNITION USING CNN
"""
import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

# 1. Fetch Dataset (with offline synthetic fallback)
target_names = np.array(['Colin Powell', 'Donald Rumsfeld', 'George W Bush', 'Gerhard Schroeder', 'Tony Blair'])
class_count = len(target_names)

try:
    print("Downloading / loading LFW dataset...")
    faces = fetch_lfw_people(min_faces_per_person=100, resize=1.0, slice_=(slice(60, 188), slice(60, 188)), color=True)
    target_names = faces.target_names
    class_count = len(target_names)
    unique_targets, mapped_target = np.unique(faces.target, return_inverse=True)
    mask = np.zeros(len(faces.target), dtype=bool)
    for t in unique_targets:
        mask[np.where(faces.target == t)[0][:100]] = True
    x_faces = faces.images[mask]
    y_faces = mapped_target[mask]
except Exception as e:
    print(f"Could not load LFW dataset online ({e}). Generating synthetic demo data...")
    x_faces = np.random.randint(50, 220, (500, 128, 128, 3), dtype=np.uint8)
    y_faces = np.repeat(np.arange(5), 100)

face_images = x_faces / 255.0
face_labels = to_categorical(y_faces, num_classes=class_count)

# 2. Visualize Sample Faces
fig, ax = plt.subplots(3, 6, figsize=(12, 6))
for i, axi in enumerate(ax.flat):
    if i < len(face_images):
        axi.imshow(face_images[i])
        axi.set(xticks=[], yticks=[], xlabel=target_names[y_faces[i]])
plt.suptitle("Sample Dataset Images", fontsize=14)
plt.tight_layout()
plt.show()

# 3. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(face_images, face_labels, train_size=0.8, stratify=face_labels, random_state=0)
print(f"Training shape: {x_train.shape}, Test shape: {x_test.shape}")

# 4. Build CNN Model
img_shape = face_images.shape[1:]
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=img_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(class_count, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 5. Train Model
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=25, verbose=1)

# 6. Plot Accuracy Curves
epochs_range = range(1, len(hist.history["accuracy"]) + 1)
plt.figure(figsize=(7, 4))
plt.plot(epochs_range, hist.history["accuracy"], "-", label="Training Accuracy")
plt.plot(epochs_range, hist.history["val_accuracy"], ":", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 7. Confusion Matrix
y_predicted = model.predict(x_test)
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_predicted, axis=1)
mat = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(7, 6))
sns.heatmap(mat, square=True, annot=True, fmt="d", cbar=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 8. External Image Inference
img_path = "george.jpg"
if os.path.exists(img_path):
    test_img = image.load_img(img_path, target_size=(img_shape[0], img_shape[1]))
    x = image.img_to_array(test_img) / 255.0
else:
    print(f"\nNotice: '{img_path}' not found. Using a sample from x_test for demo.")
    x = x_test[0]

preds = model.predict(np.expand_dims(x, axis=0))[0]
print("\n--- Prediction Probabilities ---")
for i, name in enumerate(target_names):
    print(f"{name:<25}: {preds[i]*100:.2f}%")
best_idx = np.argmax(preds)
print(f"\nResult: Identified as '{target_names[best_idx]}' with {preds[best_idx]*100:.2f}% confidence.")

