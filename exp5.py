"""
Ex. No: 5 - SENTIMENT ANALYSIS USING LSTM
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

vocab_size = 10000
max_length = 200

# 1. Load IMDB Dataset (with offline synthetic fallback)
try:
    print("Loading IMDB dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
    word_index = imdb.get_word_index()
except Exception as e:
    print(f"Could not load IMDB online ({e}). Using synthetic dataset...")
    np.random.seed(42)
    X_train = [list(np.random.randint(1, 1000, size=np.random.randint(20, 100))) for _ in range(1000)]
    y_train = np.random.randint(0, 2, size=1000)
    X_test = [list(np.random.randint(1, 1000, size=np.random.randint(20, 100))) for _ in range(200)]
    y_test = np.random.randint(0, 2, size=200)
    word_index = {"amazing": 10, "wonderful": 15, "good": 20, "boring": 30, "terrible": 40, "movie": 5}

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 2. Sequence Padding
X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")

# 3. Model Architecture
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(units=128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 4. Train
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stop])

# 5. Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("\nTest Accuracy =", accuracy)
print("Test Loss =", loss)

# 6. Plot Accuracy & Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

model.save("sentiment_lstm_model.h5")
print("Model Saved Successfully!")

# 7. Sentiment Prediction Helper
def predict_sentiment(text):
    words = text.lower().split()
    encoded_review = [word_index.get(w, 2 - 3) + 3 for w in words]
    padded_review = pad_sequences([encoded_review], maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded_review, verbose=0)[0][0]
    sentiment = "POSITIVE" if prediction >= 0.5 else "NEGATIVE"
    print("\nReview :", text)
    print("Sentiment :", sentiment)
    print("Confidence :", round(float(prediction), 4))

predict_sentiment("this movie was amazing and wonderful")
predict_sentiment("this movie was boring and terrible")

