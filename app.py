import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------
# 💾 Load model and features
# ----------------------
model_data = joblib.load("fraud_model.pkl")
model = model_data["model"]
expected_cols = model_data["features"]

# ----------------------
# 🎨 Streamlit App UI
# ----------------------
st.set_page_config(page_title="💳 FraudCard Detector", layout="centered")

st.title("💳 FraudCard Detector")
st.markdown("Upload a CSV file of transactions and detect fraudulent activity using a trained ML model.")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

# ----------------------
# 🧪 Prediction Block
# ----------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())

    if st.button("🚨 Predict Fraud"):
        try:
            # ✅ Clean Data
            data_clean = data.copy()

            # Drop columns not used in training (e.g., names, IDs)
            columns_to_drop = ['nameOrig', 'nameDest']
            data_clean = data_clean.drop(columns=columns_to_drop, errors='ignore')

            # Replace inf, drop missing
            data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
            data_clean = data_clean.dropna()

            # ✅ Match the feature columns used during training
            data_clean = data_clean[expected_cols]

            # ✅ Predict
            predictions = model.predict(data_clean)
            data['Prediction'] = predictions

            # ✅ Output
            st.success(f"✅ Prediction complete! Fraudulent transactions detected: {sum(predictions)}")
            st.dataframe(data)

            # 📈 Chart
            st.subheader("📈 Fraud vs Non-Fraud Count")
            st.bar_chart(data['Prediction'].value_counts())

        except Exception as e:
            st.error("❌ Error during prediction. Make sure your file matches the model's expected format.")
            st.exception(e)
