import streamlit as st
import numpy as np
import pickle

# Load model, scaler, and columns
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details below:")

# Create input fields dynamically
input_data = []

for col in columns:
    value = st.number_input(f"{col}", value=0.0)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("✅ Legit Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")