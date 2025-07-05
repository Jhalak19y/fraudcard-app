import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

# -------------------------------
# 🎨 Page Config
# -------------------------------
st.set_page_config(
    page_title="💳 FraudCard Detector",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# 💡 Sidebar Info
# -------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**FraudCard Detector** uses a trained ML model to detect fraudulent transactions.

- Built with 💖 using Streamlit
- Model: Random Forest
- Upload your CSV to begin
- View fraud stats, charts, and download results
""")

# -------------------------------
# 🎬 Animated Lottie Header
# -------------------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_fraud = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_s1ofpcqn.json")
st_lottie(lottie_fraud, height=200, key="fraud")

st.title("💳 FraudCard Detector")
st.markdown("Upload a CSV and let AI catch the frauds! 🚨")

# -------------------------------
# 💾 Load Model & Features
# -------------------------------
model_data = joblib.load("fraud_model.pkl")
model = model_data["model"]
expected_cols = model_data["features"]

# -------------------------------
# 📂 Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("📥 Upload your CSV file", type=["csv"])

# -------------------------------
# 🧠 Prediction Logic
# -------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(data.head())

    if st.button("🚨 Predict Fraud"):
        try:
            data_clean = data.copy()

            # Drop unused columns
            columns_to_drop = ['nameOrig', 'nameDest']
            data_clean = data_clean.drop(columns=columns_to_drop, errors='ignore')

            data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
            data_clean = data_clean.dropna()

            # Ensure correct feature order
            data_clean = data_clean[expected_cols]

            # Predict
            predictions = model.predict(data_clean)
            data['Prediction'] = predictions
            data['Status'] = data['Prediction'].apply(lambda x: '🟢 Genuine' if x == 0 else '🔴 Fraud')

            # Metrics
            total = len(predictions)
            fraud = sum(predictions)
            non_fraud = total - fraud
            fraud_percent = (fraud / total) * 100

            st.markdown("---")
            st.subheader("📊 Fraud Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔍 Total", total)
            col2.metric("🚨 Frauds", fraud)
            col3.metric("📉 Fraud %", f"{fraud_percent:.2f}%")

            # Pie chart
            st.subheader("📈 Fraud vs Non-Fraud")
            fig, ax = plt.subplots()
            labels = ['Genuine', 'Fraud']
            sizes = [non_fraud, fraud]
            colors = ['#00cc99', '#ff4d4d']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Download CSV
            st.subheader("📥 Download Results")
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV with Predictions",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

            # Final output
            st.subheader("🧾 Prediction Results")
            st.dataframe(data)

        except Exception as e:
            st.error("❌ Error during prediction. Check if your CSV matches the model format.")
            st.exception(e)

# -------------------------------
# 🙋‍♀️ Footer
# -------------------------------
st.markdown("---")
st.caption("Made with ❤️ by Jhalak Yadav")
