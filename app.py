import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")

st.title("📩 Spam Detection AI")

text = st.text_area("Enter your message")

if st.button("Predict"):
    result = model.predict([text])[0]
    
    if result == 1:
        st.error("🚫 Spam")
    else:
        st.success("✅ Not Spam")