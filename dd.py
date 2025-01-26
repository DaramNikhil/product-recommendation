import streamlit as st
from transformers import pipeline

# Load fitness text model
model = pipeline('text-classification', model='bert-base-uncased')

st.title("Fitness Predictor")
user_input = st.text_input("Enter your fitness habits:")
if user_input:
    result = model(user_input)
    st.write("Predicted Fitness Level:", result[0]['label'])
