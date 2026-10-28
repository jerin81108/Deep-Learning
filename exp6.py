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

# 1. Dataset & Preprocessing
nltk.download('treebank', quiet=True)
nltk.download('universal_tagset', quiet=True)
from nltk.corpus import treebank

sents = treebank.tagged_sents(tagset='universal')
words = [[w.lower() for w, t in s] for s in sents]
tags = [[t for w, t in s] for s in sents]

w2i = {w: i + 2 for i, w in enumerate(sorted({w for s in words for w in s}))}
w2i['PAD'], w2i['UNK'] = 0, 1
t2i = {t: i + 1 for i, t in enumerate(sorted({t for s in tags for t in s}))}
t2i['PAD'] = 0
i2t = {v: k for k, v in t2i.items()}

max_len, n_tags = 100, len(t2i)
X = pad_sequences([[w2i.get(w, 1) for w in s] for s in words], maxlen=max_len, padding='post')
y = to_categorical(pad_sequences([[t2i[t] for t in s] for s in tags], maxlen=max_len, padding='post'), num_classes=n_tags)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Seq2Seq Architecture
inp = Input(shape=(max_len,))
x = Embedding(len(w2i), 128, mask_zero=True)(inp)
x = LSTM(128, return_sequences=True)(x)
out = TimeDistributed(Dense(n_tags, activation='softmax'))(x)

model = Model(inp, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train & Evaluate
model.fit(X_tr, y_tr, batch_size=32, epochs=5, validation_split=0.1)
print("\nTest Accuracy:", model.evaluate(X_te, y_te)[1])

# 4. Predict
def predict_pos(sent):
    toks = sent.lower().split()
    seq = pad_sequences([[w2i.get(w, 1) for w in toks]], maxlen=max_len, padding='post')
    preds = np.argmax(model.predict(seq, verbose=0)[0], axis=-1)
    print("\nPOS TAGS")
    for w, p in zip(toks, preds[:len(toks)]):
        print(f"{w} --> {i2t[p]}")

predict_pos("the quick brown fox jumps over the lazy dog")
predict_pos("i am learning natural language processing")
