import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('fraud_model.pkl')

st.set_page_config(page_title="💳 FraudCard Detector", layout="centered")
st.title("💳 FraudCard Detector")
st.write("Upload a CSV file with transaction data to detect frauds.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Uploaded Data:")
    st.dataframe(data.head())

    if st.button("🔍 Predict"):
        predictions = model.predict(data)
        data['Prediction'] = predictions
        st.success(f"✅ Prediction complete. Fraudulent transactions: {sum(predictions)}")
        st.write(data)
