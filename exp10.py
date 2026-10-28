"""
Ex. No: 10 - SIGN LANGUAGE RECOGNITION AND SPEECH OUTPUT USING CNN-LSTM
"""
import os
import cv2
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

# Try importing pyttsx3 and mediapipe safely
try:
    import pyttsx3
    speaker = pyttsx3.init()
except Exception:
    speaker = None

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception:
    mp_hands = None

DATASET = "sign_dataset"
SEQUENCE_LENGTH = 30
FEATURES = 63
CLASSES = ["HELLO", "YES", "NO", "THANK_YOU"]

# 1. Prepare Dataset (Auto-generate synthetic gesture data if absent)
def prepare_data_if_needed():
    for label in CLASSES:
        folder = os.path.join(DATASET, label)
        os.makedirs(folder, exist_ok=True)
        if len([f for f in os.listdir(folder) if f.endswith(".npy")]) < 5:
            print(f"Creating sample gesture sequences for '{label}'...")
            for i in range(10):
                seq = np.random.uniform(-0.5, 0.5, (SEQUENCE_LENGTH, FEATURES)).astype(np.float32)
                np.save(os.path.join(folder, f"sample_{i:04d}.npy"), seq)

prepare_data_if_needed()

# 2. Load Dataset
X, labels = [], []
for label in sorted(os.listdir(DATASET)):
    folder = os.path.join(DATASET, label)
    if not os.path.isdir(folder):
        continue
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            seq = np.load(os.path.join(folder, filename))
            if seq.shape == (SEQUENCE_LENGTH, FEATURES):
                X.append(seq)
                labels.append(label)

X = np.array(X, dtype=np.float32)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Build & Train LSTM Model
model = Sequential([
    tf.keras.Input(shape=(SEQUENCE_LENGTH, FEATURES)),
    Masking(mask_value=0.0),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dense(32, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_split=0.2, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}% | Test Loss: {loss:.4f}")

model.save("sign_language_model.keras")
np.save("gesture_labels.npy", encoder.classes_)
print("Model and labels saved successfully.")

# 4. Speech Announcement Helper
def speak_gesture(text):
    print(f"[Speech Output]: {text}")
    if speaker:
        try:
            speaker.say(text)
            speaker.runAndWait()
        except Exception:
            pass

# 5. Test Inference Demonstration
sample_pred = model.predict(X_test[:1], verbose=0)[0]
pred_class = encoder.classes_[np.argmax(sample_pred)]
confidence = np.max(sample_pred) * 100
print(f"\nDemonstration Prediction: '{pred_class}' ({confidence:.2f}% confidence)")
speak_gesture(pred_class)

