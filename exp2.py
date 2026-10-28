"""
Ex. No: 2 - CHARACTER RECOGNITION USING CNN
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 1. Load Dataset (with synthetic fallback if CSV is absent)
csv_file = "A_Z Handwritten Data.csv"
if os.path.exists(csv_file):
    print("Loading dataset from CSV...")
    data = pd.read_csv(csv_file).astype("float32")
    X = data.drop("0", axis=1)
    y = data["0"]
else:
    print(f"'{csv_file}' not found. Generating sample data for demo execution...")
    np.random.seed(42)
    sample_size = 520  # 20 samples per alphabet
    y = pd.Series(np.repeat(np.arange(26), sample_size // 26))
    X = pd.DataFrame(np.random.randint(0, 256, (sample_size, 784), dtype=np.uint8))

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))
word_dict = {i: chr(65 + i) for i in range(26)}

# 2. Visualize Sample Grid
shuff = shuffle(train_x[:100])
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(shuff[i], cmap="gray")
    ax.axis("off")
plt.suptitle("Sample Training Characters")
plt.show()

# 3. Preprocess for CNN
train_X = train_x.reshape(-1, 28, 28, 1) / 255.0
test_X = test_x.reshape(-1, 28, 28, 1) / 255.0
train_yOHE = to_categorical(train_y, num_classes=26)
test_yOHE = to_categorical(test_y, num_classes=26)

# 4. Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), activation="relu", padding="valid"),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.25),
    Dense(128, activation="relu"),
    Dense(26, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 5. Train
history = model.fit(train_X, train_yOHE, epochs=1, batch_size=128, validation_data=(test_X, test_yOHE))

# 6. Test Predictions Plot
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
preds = model.predict(test_X[:9])
for i, ax in enumerate(axes.flatten()):
    ax.imshow(test_X[i].reshape(28, 28), cmap="gray")
    pred_char = word_dict[np.argmax(preds[i])]
    true_char = word_dict[np.argmax(test_yOHE[i])]
    ax.set_title(f"Pred: {pred_char} | True: {true_char}", color="green" if pred_char == true_char else "red")
    ax.axis("off")
plt.tight_layout()
plt.show()

