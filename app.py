import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Banking Fraud Detection Dashboard")
st.markdown("### Predict and analyze fraudulent transactions in real-time")

# -------------------------------
# LOAD MODEL (You must save model as model.pkl)
# -------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🔧 Enter Transaction Details")

time = st.sidebar.slider("Transaction Time", 0, 172800, 100000)
amount = st.sidebar.slider("Transaction Amount", 0.0, 5000.0, 100.0)

# PCA features (simplified selection)
v_features = {}
for i in range(1, 6):  # using V1–V5 for demo
    v_features[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0)

# -------------------------------
# PREPARE INPUT DATA
# -------------------------------
input_data = {
    'Time': time,
    'Amount': amount
}

input_data.update(v_features)

input_df = pd.DataFrame([input_data])

# Fill missing V6–V28 with 0 (since model expects all)
for i in range(6, 29):
    input_df[f'V{i}'] = 0

# Reorder columns properly
columns_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_df = input_df[columns_order]

# -------------------------------
# SCALE AMOUNT (IMPORTANT)
# -------------------------------
scaler = StandardScaler()
input_df['Amount'] = scaler.fit_transform(input_df[['Amount']])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Fraud"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric("Fraud Probability", f"{probability*100:.2f}%")

# -------------------------------
# DASHBOARD SECTION
# -------------------------------
st.markdown("---")
st.subheader("📊 Transaction Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💰 Amount Analysis")
    st.bar_chart([amount])

with col2:
    st.markdown("### ⏱ Time Indicator")
    st.line_chart([time])

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built for Applied Business Analytics Project 🚀")