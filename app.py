import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# 🧠 Load model + expected columns
# -------------------------------
model_data = joblib.load("fraud_model.pkl")
model = model_data["model"]
expected_cols = model_data["features"]

# -------------------------------
# 🎨 Page Config
# -------------------------------
st.set_page_config(
    page_title="💳 FraudCard Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("💳 FraudCard Detector")
st.markdown("Let AI analyze your transactions and detect frauds in real-time 🚨")

st.markdown("---")
st.subheader("📂 Step 1: Upload Your Transaction CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# -------------------------------
# 🧪 Prediction Logic
# -------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(data.head())

    if st.button("🚨 Predict Fraud"):
        try:
            # Clean and prepare data
            data_clean = data.copy()
            columns_to_drop = ['nameOrig', 'nameDest']
            data_clean = data_clean.drop(columns=columns_to_drop, errors='ignore')
            data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
            data_clean = data_clean.dropna()
            data_clean = data_clean[expected_cols]

            # Predict
            predictions = model.predict(data_clean)
            data['Prediction'] = predictions

            # Metrics
            total = len(predictions)
            fraud = sum(predictions)
            non_fraud = total - fraud
            fraud_percent = (fraud / total) * 100

            st.markdown("---")
            st.subheader("📊 Step 2: Fraud Summary Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔍 Total Transactions", total)
            col2.metric("🚨 Frauds Detected", fraud)
            col3.metric("📉 Fraud %", f"{fraud_percent:.2f}%")

            # Chart
            st.subheader("📈 Fraud vs Non-Fraud Chart")
            fig, ax = plt.subplots()
            labels = ['Genuine', 'Fraud']
            sizes = [non_fraud, fraud]
            colors = ['#00cc99', '#ff4d4d']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Download button
            st.subheader("📥 Step 3: Download Results")
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction Results as CSV",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv',
            )

            st.markdown("---")
            st.subheader("📋 Full Prediction Results")
            st.dataframe(data)

        except Exception as e:
            st.error("❌ Something went wrong. Please make sure your file is in the correct format.")
            st.exception(e)

# -------------------------------
# 🙋‍♀️ Footer
# -------------------------------
st.markdown("---")
st.caption("Made with ❤️ by **Jhalak Yadav** | ML Intern @ Suvidha Foundation")
