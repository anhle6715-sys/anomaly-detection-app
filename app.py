import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")  # Thay bằng file model của bạn
    return model

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Anomaly Detection App")

st.write("Upload your CSV file to detect anomalies.")

# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Nút dự đoán
    if st.button("Detect Anomalies"):
        # Giả sử model là IsolationForest hoặc tương tự
        predictions = model.predict(data)

        # -1 = anomaly, 1 = normal
        data["Anomaly"] = ["Anomaly" if x == -1 else "Normal" for x in predictions]

        st.write("### Detection Results")
        st.dataframe(data)

        # Tải về file kết quả
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )
