import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page settings
st.set_page_config(page_title="💳 FraudCard Detector", layout="centered")

# Title
st.title("💳 FraudCard Detector")
st.markdown("Upload a CSV file containing transaction data to detect fraudulent activities.")

# Load the trained model
model = joblib.load("fraud_model.pkl")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(data.head())

    if st.button("🚨 Predict Fraud"):
        try:
            # Copy and clean data before prediction
            data_clean = data.copy()

            # Drop columns not used in model (like names or IDs)
            columns_to_drop = ['nameOrig', 'nameDest']  # Adjust if needed
            data_clean = data_clean.drop(columns=columns_to_drop, errors='ignore')

            # Replace inf with NaN and drop rows with NaN
            data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
            data_clean = data_clean.dropna()

            # Ensure numeric data
            data_clean = data_clean.select_dtypes(include=[np.number])

            # Predict
            predictions = model.predict(data_clean)

            # Add predictions to original data
            data['Prediction'] = predictions

            # Show results
            st.success(f"✅ Prediction complete! Fraudulent transactions detected: {sum(predictions)}")
            st.dataframe(data)

            # Optional: show counts
            st.subheader("📈 Fraud vs. Non-Fraud Count")
            st.bar_chart(data['Prediction'].value_counts())

        except Exception as e:
            st.error("❌ Error during prediction. Please check your file format and try again.")
            st.exception(e)
