import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import joblib
import altair as alt

st.set_page_config(page_title="Anomaly Detection App", layout="wide")

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data
def read_csv(file, **kwargs):
    return pd.read_csv(file, **kwargs)

@st.cache_resource
def load_uploaded_model(file_bytes: bytes):
    """Load a user-uploaded sklearn/joblib model from bytes."""
    file_like = io.BytesIO(file_bytes)
    model = joblib.load(file_like)
    return model

@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# -------------------------------
# UI – Sidebar
# -------------------------------
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio(
    "Model mode",
    [
        "Use uploaded model (.pkl)",
        "Train IsolationForest now",
    ],
)

st.title("🔎 Anomaly Detection App – Streamlit version")
st.caption(
    "Upload a CSV, choose a model, detect anomalies, visualize, and download results."
)

# -------------------------------
# Data upload
# -------------------------------
uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_csv is None:
    st.info("👆 Upload a CSV to get started.")
    st.stop()

# Read and preview data
try:
    df = read_csv(uploaded_csv)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded CSV is empty.")
    st.stop()

st.subheader("📄 Data preview")
st.dataframe(df.head(100), use_container_width=True)

# Column selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Please upload a CSV with numeric features.")
    st.stop()

with st.expander("Select features for modeling", expanded=True):
    feat_cols = st.multiselect(
        "Numeric columns to use as features",
        options=numeric_cols,
        default=numeric_cols,
    )

if not feat_cols:
    st.warning("Please select at least one numeric column to proceed.")
    st.stop()

X = df[feat_cols].copy()

# -------------------------------
# Model selection / training
# -------------------------------
model = None
if mode == "Use uploaded model (.pkl)":
    model_file = st.sidebar.file_uploader("Upload model .pkl (sklearn joblib)", type=["pkl", "joblib"])
    if model_file is not None:
        try:
            model = load_uploaded_model(model_file.read())
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
else:
    st.sidebar.markdown("### IsolationForest parameters")
    contamination = st.sidebar.slider(
        "Contamination (expected outlier proportion)", 0.0, 0.5, 0.05, 0.01
    )
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    max_samples = st.sidebar.select_slider(
        "max_samples",
        options=["auto", 64, 128, 256, 512, 1024, 2048, 4096],
        value="auto",
    )
    random_state = st.sidebar.number_input("random_state", value=42, step=1)

    if st.sidebar.button("Train model"):
        try:
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                max_samples=max_samples,
                random_state=int(random_state),
            )
            model.fit(X)
            st.sidebar.success("Model trained!")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# Guard if model still None
if model is None:
    st.warning("📦 Please upload a trained model or click 'Train model' in the sidebar.")
    st.stop()

# -------------------------------
# Predict anomalies
# -------------------------------
with st.spinner("Detecting anomalies..."):
    try:
        y_pred = model.predict(X)  # 1 for normal, -1 for anomaly
        scores = None
        # If available, get anomaly scores (lower = more anomalous for IsolationForest)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

result = df.copy()
result["AnomalyFlag"] = (y_pred == -1)
result["AnomalyLabel"] = np.where(result["AnomalyFlag"], "Anomaly", "Normal")
if scores is not None:
    result["AnomalyScore"] = scores

# Summary
n_anom = int(result["AnomalyFlag"].sum())
ratio = n_anom / len(result)
st.success(f"✅ Detection done: {n_anom} anomalies out of {len(result)} rows (ratio {ratio:.2%}).")

# Show table
st.subheader("🧾 Results table (first 1,000 rows)")
st.dataframe(result.head(1000), use_container_width=True)

# Download
st.download_button(
    label="⬇️ Download results as CSV",
    data=to_csv_bytes(result),
    file_name="anomaly_detection_results.csv",
    mime="text/csv",
)

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📊 Visualization")

if len(feat_cols) >= 2:
    x_col = st.selectbox("X axis", feat_cols, index=0)
    y_col = st.selectbox("Y axis", feat_cols, index=1)

    chart_df = result[[x_col, y_col, "AnomalyLabel"]].rename(
        columns={"AnomalyLabel": "Label"}
    )

    scatter = (
        alt.Chart(chart_df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            color=alt.Color("Label", scale=alt.Scale(scheme="set1")),
            tooltip=[x_col, y_col, "Label"],
        )
        .interactive()
    )

    st.altair_chart(scatter, use_container_width=True)
else:
    st.info("Select at least two numeric columns to see a scatter plot.")

# -------------------------------
# Tips
# -------------------------------
with st.expander("ℹ️ Tips & Notes"):
    st.markdown(
        """
        - **This is a pure Streamlit app**. There's no Flask dependency, so it runs fine on Streamlit Cloud.
        - To use your own model, export it with `joblib.dump(model, "model.pkl")` and upload it in the sidebar.
        - IsolationForest convention: **-1 = anomaly**, **1 = normal**. We convert to labels for readability.
        - If your dataset has non-numeric columns, select only the numeric ones in the feature selector above.
        """
    )
