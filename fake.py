# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 01:06:46 2025

@author: hp
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# ===================== #
# 📌 Download NLTK Resources
# ===================== #
nltk.download("stopwords")
nltk.download("punkt")
nltk.data.path.append("C:/Users/hp/AppData/Roaming/nltk_data")

# ===================== #
# 📌 Load and Preprocess Dataset
# ===================== #
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/hp/Desktop/news.csv")  # Ensure 'news.csv' has 'text' & 'label' columns
    df.dropna(inplace=True)  # Remove missing values
    return df

# Function to Clean and Process Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))  # Load stopwords
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load and preprocess data
df = load_data()
df["text"] = df["text"].apply(clean_text)

# ===================== #
# 📌 Tokenization & Padding
# ===================== #
MAX_NB_WORDS = 10000  # Increased Vocabulary size
MAX_SEQUENCE_LENGTH = 500  # Max length of input text
EMBEDDING_DIM = 100  # Increased embedding dimensions

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df["text"])
X = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

y = df["label"].map({"fake": 0, "real": 1}).values  # Convert labels to 0 (Fake) & 1 (Real)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== #
# 📌 Load Pretrained Glove Embeddings (Optional)
# ===================== #
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < MAX_NB_WORDS:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# ===================== #
# 📌 Build Improved LSTM Model
# ===================== #
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(150, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Save Model & Tokenizer
model.save("fake_news_lstm_model.h5")
joblib.dump(tokenizer, "tokenizer.pkl")

# ===================== #
# 📌 Streamlit UI for Fake News Detection
# ===================== #
st.title("📰 Fake News Detector (LSTM) 🔥")
st.markdown("## AI-Powered Fake News Detection with Enhanced Accuracy 🚀")

st.sidebar.header("⚙️ About")
st.sidebar.write("This AI model uses an **LSTM-based deep learning approach** with **Glove embeddings** to detect fake news.")

# Load Model & Tokenizer
model = tf.keras.models.load_model("fake_news_lstm_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# User Input
user_input = st.text_area("📌 Enter a news article", "")

if st.button("🔍 Detect Fake News"):
    if user_input:
        with st.spinner("Analyzing the article..."):
            # Preprocess Input
            cleaned_text = clean_text(user_input)
            transformed_text = tokenizer.texts_to_sequences([cleaned_text])
            transformed_text = pad_sequences(transformed_text, maxlen=MAX_SEQUENCE_LENGTH)

            # Predict
            prediction = model.predict(transformed_text)[0][0]
            result = "🟢 Real News" if prediction > 0.5 else "🔴 Fake News"
            confidence = prediction if prediction > 0.5 else (1 - prediction)

        # Display Results
        st.markdown(f"### Prediction: {result}")
        st.progress(float(confidence))  # Show confidence level
        st.write(f"**Confidence Level: {confidence:.2%}**")

        # Confidence Visualization
        fig, ax = plt.subplots()
        ax.bar(["Real", "Fake"], [prediction, 1 - prediction], color=["green", "red"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please enter some text for analysis.")

# ===================== #
# 📌 Footer
# ===================== #
st.sidebar.info("Developed by Abhinav Rao 🚀")
