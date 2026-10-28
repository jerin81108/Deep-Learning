"""
Ex. No: 6 - PARTS OF SPEECH (POS) TAGGING USING SEQUENCE-TO-SEQUENCE ARCHITECTURE
"""
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 1. Load Dataset
try:
    nltk.download('treebank', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    from nltk.corpus import treebank
    sentences = treebank.tagged_sents(tagset='universal')
except Exception as e:
    print(f"NLTK treebank unavailable ({e}). Using sample tagged sentences...")
    sentences = [
        [("the", "DET"), ("quick", "ADJ"), ("brown", "NOUN"), ("fox", "NOUN"), ("jumps", "NOUN"), ("over", "ADP"), ("the", "DET"), ("lazy", "NOUN"), ("dog", "NOUN")],
        [("i", "PRON"), ("am", "VERB"), ("learning", "NOUN"), ("natural", "NOUN"), ("language", "NOUN"), ("processing", "NOUN")]
    ] * 50

words, tags = [], []
for sent in sentences:
    words.append([w.lower() for w, t in sent])
    tags.append([t for w, t in sent])

print("Total Sentences:", len(words))

# 2. Vocabulary & Mappings
word_vocab = sorted(list(set(w for s in words for w in s)))
tag_vocab = sorted(list(set(t for s in tags for t in s)))

word2idx = {w: i + 2 for i, w in enumerate(word_vocab)}
word2idx["PAD"], word2idx["UNK"] = 0, 1

tag2idx = {t: i + 1 for i, t in enumerate(tag_vocab)}
tag2idx["PAD"] = 0
idx2tag = {v: k for k, v in tag2idx.items()}

# 3. Vectorization & Padding
max_len = 100
X = [[word2idx.get(w, word2idx["UNK"]) for w in s] for s in words]
y = [[tag2idx[t] for t in s] for s in tags]

X = pad_sequences(X, maxlen=max_len, padding="post")
y = pad_sequences(y, maxlen=max_len, padding="post")

num_tags = len(tag2idx)
y = np.array([to_categorical(seq, num_classes=num_tags) for seq in y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Seq2Seq Architecture
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=len(word2idx), output_dim=128, mask_zero=True)(input_layer)
lstm = LSTM(units=128, return_sequences=True)(embedding)
output = TimeDistributed(Dense(num_tags, activation="softmax"))(lstm)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 5. Train
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

# 6. Predict POS Function
def predict_pos(sentence):
    tokens = sentence.lower().split()
    sequence = [word2idx.get(w, word2idx["UNK"]) for w in tokens]
    padded = pad_sequences([sequence], maxlen=max_len, padding="post")
    prediction = model.predict(padded, verbose=0)
    pred_idx = np.argmax(prediction, axis=-1)[0]
    
    print("\nPOS TAGS")
    for word, pred in zip(tokens, pred_idx[:len(tokens)]):
        print(f"{word} --> {idx2tag[pred]}")

predict_pos("the quick brown fox jumps over the lazy dog")
predict_pos("i am learning natural language processing")

