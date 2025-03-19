# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import requests

# API URL
API_URL = "http://127.0.0.1:8000"

st.title("📰 Fake News Detection System")

# User input
option = st.radio("Choose input type:", ("Text", "URL"))

if option == "Text":
    text_input = st.text_area("Enter news text:")
    if st.button("Check News"):
        response = requests.post(f"{API_URL}/predict/", params={"text": text_input})
        result = response.json()
        st.write(f"**Prediction:** {result['prediction']}")
elif option == "URL":
    url_input = st.text_input("Enter news URL:")
    if st.button("Check URL"):
        response = requests.post(f"{API_URL}/predict_url/", params={"url": url_input})
        result = response.json()
        if "error" in result:
            st.error("Could not process the URL. Try another one.")
        else:
            st.write(f"**Prediction:** {result['prediction']}")

