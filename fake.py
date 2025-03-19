# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 01:06:46 2025

@author: hp
"""

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
def load_data():
    df = pd.read_csv("C:/Users/hp/Desktop/news.csv")  # Update with correct filename
    df.columns = df.columns.str.strip().str.lower()
    if 'label' not in df.columns:
        st.error("Dataset does not contain a 'label' column!")
        st.stop()
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Preprocess Data
def preprocess_data(df):
    df['text'] = df['text'].astype(str).apply(clean_text)
    df['label'] = df['label'].map({'fake': 0, 'real': 1})
    return df

# Tokenization & Padding
def prepare_sequences(texts, max_words=5000, max_len=200):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, tokenizer

# Train LSTM Model
def train_lstm(X_train, y_train):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=200),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
    return model

# Streamlit UI
st.title("📰 Fake News Detection with LSTM")

df = load_data()
df = preprocess_data(df)
X, tokenizer = prepare_sequences(df['text'])
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.button("Train Model"):
    st.write("Training LSTM Model... Please wait!")
    model = train_lstm(X_train, y_train)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Trained! Accuracy: {acc:.2f}")

# User Input for Prediction
user_input = st.text_area("Enter news text to check if it's fake or real:")
if st.button("Predict"):
    input_seq = tokenizer.texts_to_sequences([clean_text(user_input)])
    input_pad = pad_sequences(input_seq, maxlen=200, padding='post')
    prediction = model.predict(input_pad)
    label = "Real News" if prediction > 0.5 else "Fake News"
    st.subheader(f"Prediction: {label}")
