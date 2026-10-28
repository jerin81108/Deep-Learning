"""
Ex. No: 7 - MACHINE TRANSLATION USING ENCODER-DECODER ARCHITECTURE
"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 1. Dataset
english_texts = [
    "hello", "how are you", "good morning", "good night", "thank you",
    "i love you", "see you later", "what is your name", "where are you", "nice to meet you"
]
french_texts = [
    "bonjour", "comment allez vous", "bonjour matin", "bonne nuit", "merci",
    "je t aime", "a plus tard", "quel est votre nom", "ou etes vous", "ravi de vous rencontrer"
]

decoder_input_texts = ["startseq " + t for t in french_texts]
decoder_target_texts = [t + " endseq" for t in french_texts]

# 2. Tokenization & Sequence Preparation
input_tokenizer = Tokenizer(filters="")
input_tokenizer.fit_on_texts(english_texts)
target_tokenizer = Tokenizer(filters="")
target_tokenizer.fit_on_texts(decoder_input_texts + decoder_target_texts)

encoder_input_sequences = input_tokenizer.texts_to_sequences(english_texts)
decoder_input_sequences = target_tokenizer.texts_to_sequences(decoder_input_texts)
decoder_target_sequences = target_tokenizer.texts_to_sequences(decoder_target_texts)

max_encoder_len = max(len(seq) for seq in encoder_input_sequences)
max_decoder_len = max(len(seq) for seq in decoder_input_sequences)

encoder_input_data = pad_sequences(encoder_input_sequences, maxlen=max_encoder_len, padding="post")
decoder_input_data = pad_sequences(decoder_input_sequences, maxlen=max_decoder_len, padding="post")
decoder_target_data = pad_sequences(decoder_target_sequences, maxlen=max_decoder_len, padding="post")

num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1
decoder_target_data = to_categorical(decoder_target_data, num_classes=num_decoder_tokens)

# 3. Build Training Model
latent_dim = 256
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, latent_dim)
decoder_emb = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 4. Train Model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=2, epochs=100, verbose=1)

# 5. Build Inference Model
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=decoder_states_inputs)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2, state_h2, state_c2])

reverse_target_word_index = {idx: word for word, idx in target_tokenizer.word_index.items()}

# 6. Translation Inference
def translate_sentence(sentence):
    seq = input_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_encoder_len, padding="post")
    states_value = encoder_model.predict(seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index["startseq"]
    
    decoded_sentence = ""
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, "")
        if sampled_word == "endseq" or sampled_word == "":
            break
        decoded_sentence += sampled_word + " "
        if len(decoded_sentence.split()) > max_decoder_len:
            break
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()

for s in ["hello", "thank you", "good night", "i love you"]:
    print(f"\nEnglish : {s}")
    print(f"French  : {translate_sentence(s)}")

