# app.py
import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import joblib
import altair as alt

st.set_page_config(page_title="Anomaly Detection (Streamlit)", layout="wide")

# ---- Dark mode CSS (simple) ----
def enable_dark_mode():
    dark_css = """
    <style>
    /* background */
    .reportview-container, .css-18e3th9 {background-color:#0e1116;}
    .css-1d391kg { background-color: #0e1116; }
    /* text */
    .stMarkDown, .css-1v3fvcr, .css-1ln0q3q { color: #e6edf3; }
    .st-bf { color: #e6edf3; }
    /* inputs */
    .stTextInput>div>div>input, .stFileUploader>div>div>input { background:#111418; color:#e6edf3; }
    .stButton>button { background-color:#1f6feb; color: white; }
    .stDownloadButton>button { background-color:#1f6feb; color: white; }
    /* table */
    .element-container .stDataFrame { background: #0b0d10; color:#e6edf3; }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

# default: dark mode enabled
enable_dark_mode()

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data
def read_csv(file, **kwargs):
    return pd.read_csv(file, **kwargs)

@st.cache_resource
def load_uploaded_model_bytes(b: bytes):
    f = io.BytesIO(b)
    return joblib.load(f)

@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# -------------------------------
# Sidebar settings (defaults)
# -------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.caption("Defaults: Use uploaded model (.pkl)")

model_mode = st.sidebar.radio(
    "Model mode",
    ["Use uploaded model (.pkl)", "Train IsolationForest now"],
    index=0  # default to uploaded model
)

st.title("🔎 Anomaly Detection (Streamlit) — Dark mode")
st.caption("Upload CSV → choose model (upload .pkl) or train IsolationForest → detect anomalies → download results")

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_csv is None:
    st.info("Upload a CSV to start.")
    st.stop()

try:
    df = read_csv(uploaded_csv)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

if df.empty:
    st.warning("Uploaded CSV is empty.")
    st.stop()

st.subheader("📄 Data preview")
st.dataframe(df.head(100), use_container_width=True)

# numeric features selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Please upload CSV with numeric features.")
    st.stop()

with st.expander("Select numeric features for model", expanded=True):
    feat_cols = st.multiselect("Numeric columns", options=numeric_cols, default=numeric_cols)

if not feat_cols:
    st.warning("Select at least one numeric feature.")
    st.stop()

X = df[feat_cols].copy()

# -------------------------------
# Model selection / upload / train
# -------------------------------
model = None
if model_mode == "Use uploaded model (.pkl)":
    model_file = st.sidebar.file_uploader("Upload model (.pkl or .joblib)", type=["pkl", "joblib"])
    if model_file is not None:
        try:
            model = load_uploaded_model_bytes(model_file.read())
            st.sidebar.success("Model loaded from upload.")
        except Exception as e:
            st.sidebar.error(f"Failed loading model: {e}")
else:
    st.sidebar.markdown("### IsolationForest parameters")
    contamination = st.sidebar.slider("Contamination (outlier proportion)", 0.0, 0.5, 0.05, 0.01)
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    max_samples = st.sidebar.select_slider("max_samples", options=["auto", 64, 128, 256, 512, 1024], value="auto")
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
            st.sidebar.success("Model trained.")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# Guard: model must be present
if model is None:
    st.warning("Please upload a trained model file (.pkl) OR train IsolationForest in the sidebar.")
    st.stop()

# -------------------------------
# Prediction
# -------------------------------
with st.spinner("Detecting anomalies..."):
    try:
        y_pred = model.predict(X)  # IsolationForest: 1 normal, -1 anomaly
        scores = None
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

n_anom = int(result["AnomalyFlag"].sum())
st.success(f"✅ Detection done: {n_anom} anomalies / {len(result)} rows ({n_anom/len(result):.2%})")

# results table
st.subheader("🧾 Results (preview)")
st.dataframe(result.head(1000), use_container_width=True)

# download
st.download_button("⬇️ Download results (CSV)", data=to_csv_bytes(result), file_name="anomaly_results.csv", mime="text/csv")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📊 Visualization")
if len(feat_cols) >= 2:
    x_col = st.selectbox("X axis", feat_cols, index=0)
    y_col = st.selectbox("Y axis", feat_cols, index=1)
    chart_df = result[[x_col, y_col, "AnomalyLabel"]].rename(columns={"AnomalyLabel": "Label"})
    scatter = (
        alt.Chart(chart_df)
        .mark_circle(size=60, opacity=0.8)
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
    st.info("Select at least two numeric features to draw scatter plot.")

# tips
with st.expander("ℹ️ Tips & notes"):
    st.markdown("""
    - Upload a trained sklearn model exported with `joblib.dump(model, 'model.pkl')` to use your model.
    - If using IsolationForest: convention is `-1 = anomaly`, `1 = normal`.
    - If your model expects scaled input, scale/features must match what the model was trained on.
    """)
