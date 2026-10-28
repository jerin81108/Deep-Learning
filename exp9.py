"""
Ex. No: 9 - VIDEO CAPTIONING / ACTION RECOGNITION USING HYBRID CNN-LSTM
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout

# 1. Configuration
DATASET_PATH = "dataset"
FRAMES = 16
IMG_SIZE = 64
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Helper: Auto-create sample videos if dataset folder is missing
def create_demo_videos_if_needed():
    classes = ["Walking", "Running", "Jumping"]
    needed = False
    for c in classes:
        p = os.path.join(DATASET_PATH, c)
        if not os.path.exists(p) or len(os.listdir(p)) < 2:
            needed = True
            break
    if needed:
        print("Dataset videos not found. Creating sample synthetic action videos for demo...")
        for c in classes:
            folder = os.path.join(DATASET_PATH, c)
            os.makedirs(folder, exist_ok=True)
            for v_idx in range(1, 3):
                v_path = os.path.join(folder, f"video{v_idx}.mp4")
                if not os.path.exists(v_path):
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(v_path, fourcc, 10.0, (IMG_SIZE, IMG_SIZE))
                    for f in range(20):
                        frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                        cv2.circle(frame, (10 + f * 2, 30), 8, (255, 255, 255), -1)
                        out.write(frame)
                    out.release()

create_demo_videos_if_needed()

# 2. Extract Equally Spaced Frames
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or total < 1:
        cap.release()
        raise ValueError("Unable to read video: " + video_path)
    indices = np.linspace(0, total - 1, FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Frame extraction failed: " + video_path)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype("float32") / 255.0)
    cap.release()
    return np.array(frames, dtype=np.float32)

# 3. Load Dataset
X, labels = [], []
for class_name in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue
    for filename in sorted(os.listdir(class_path)):
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            p = os.path.join(class_path, filename)
            try:
                X.append(extract_frames(p))
                labels.append(class_name)
                print("Loaded:", p)
            except Exception as e:
                print("Error loading:", p, e)

X = np.array(X, dtype=np.float32)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Hybrid CNN-LSTM Model
model = Sequential([
    tf.keras.Input(shape=(FRAMES, IMG_SIZE, IMG_SIZE, 3)),
    TimeDistributed(Conv2D(16, (3, 3), activation="relu", padding="same")),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same")),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 5. Train & Evaluate
history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.2, verbose=1)
loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Loss:", round(float(loss), 4))
print("Test Accuracy:", round(float(test_accuracy) * 100, 2), "%")

probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)
print("\nClassification Report:\n", classification_report(y_test, predictions, labels=np.arange(len(encoder.classes_)), target_names=encoder.classes_, zero_division=0))

# 6. Predict on Video
def predict_action(video_path):
    video = np.expand_dims(extract_frames(video_path), axis=0)
    probs = model.predict(video, verbose=0)[0]
    best_idx = np.argmax(probs)
    action = encoder.inverse_transform([best_idx])[0]
    conf = probs[best_idx] * 100
    print(f"Predicted Action: {action} | Confidence: {conf:.2f}%")
    return action, conf

# Demo prediction on first test sample video
sample_vid = os.path.join(DATASET_PATH, encoder.classes_[0], "video1.mp4")
if os.path.exists(sample_vid):
    predict_action(sample_vid)

# 7. Plot Performance
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN-LSTM Training Performance")
plt.legend()
plt.tight_layout()
plt.show()

