import streamlit as st
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

st.set_page_config(page_title="Advanced Plant Disease Dashboard", layout="wide")

# ─── Load Metadata ─────────────────────────────────────
if os.path.exists("dataset_metadata.json"):
    with open("dataset_metadata.json") as f:
        meta = json.load(f)
else:
    meta = {"num_classes": 0, "classes": [], "split_counts": {}}

# ─── Sidebar ───────────────────────────────────────────
st.sidebar.title("🚀 Navigation")
page = st.sidebar.selectbox("Select Page", [
    "Overview",
    "Dataset Explorer",
    "EDA Insights",
    "Data Augmentation Demo",
    "Model Architecture",
    "Training Monitor",
    "Evaluation Metrics",
    "Confusion Matrix",
    "Predictions (Upload Image)",
    "Logs & Analytics",
    "Settings"
])

# ─── 1. Overview ───────────────────────────────────────
if page == "Overview":
    st.title("🌿 Advanced Plant Disease Detection Dashboard")
    st.markdown("""
    ### Features Included:
    1. Dataset summary
    2. Class visualization
    3. Image preview
    4. Data augmentation demo
    5. CNN architecture visualization
    6. Training monitoring
    7. Metrics tracking
    8. Confusion matrix
    9. Image prediction
    10. Logs analysis
    11. Interactive filters
    12. Data tables
    13. Graph insights
    14. Model loading
    15. Real-time prediction
    16. Accuracy tracking
    17. Recall & Precision
    18. CSV log viewer
    19. UI navigation
    20. Expandable sections
    """)

# ─── 2. Dataset Explorer ────────────────────────────────
elif page == "Dataset Explorer":
    st.header("📊 Dataset Explorer")

    st.metric("Total Classes", meta.get("num_classes", 0))

    if meta["classes"]:
        selected_class = st.selectbox("Select Class", meta["classes"])
        st.write(f"Selected: {selected_class}")

    st.write("### Data Split")
    st.json(meta.get("split_counts", {}))

# ─── 3. EDA Insights ───────────────────────────────────
elif page == "EDA Insights":
    st.header("📈 EDA Insights")

    if os.path.exists("eda_class_distribution.png"):
        st.image("eda_class_distribution.png")

    if os.path.exists("eda_sample_images.png"):
        st.image("eda_sample_images.png")

# ─── 4. Data Augmentation Demo ─────────────────────────
elif page == "Data Augmentation Demo":
    st.header("🔄 Augmentation Demo")
    uploaded = st.file_uploader("Upload Image")

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Original")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img.rotate(45), caption="Rotated")
        with col2:
            st.image(img.transpose(Image.FLIP_LEFT_RIGHT), caption="Flipped")

# ─── 5. Model Architecture ─────────────────────────────
elif page == "Model Architecture":
    st.header("🧠 CNN Architecture")

    st.markdown("""
    - Conv Blocks: 32 → 64 → 128 → 256
    - BatchNorm + ReLU
    - MaxPooling + Dropout
    - Dense Layer (512)
    - Softmax Output
    """)

# ─── 6. Training Monitor ───────────────────────────────
elif page == "Training Monitor":
    st.header("📊 Training Monitor")

    if os.path.exists("logs/custom_cnn_history.csv"):
        df = pd.read_csv("logs/custom_cnn_history.csv")

        st.line_chart(df[["accuracy", "val_accuracy"]])
        st.line_chart(df[["loss", "val_loss"]])

# ─── 7. Evaluation Metrics ─────────────────────────────
elif page == "Evaluation Metrics":
    st.header("📊 Metrics")

    if os.path.exists("logs/custom_cnn_history.csv"):
        df = pd.read_csv("logs/custom_cnn_history.csv")
        st.write(df.describe())

# ─── 8. Confusion Matrix ───────────────────────────────
elif page == "Confusion Matrix":
    st.header("📉 Confusion Matrix")
    st.info("Add confusion matrix computation here")

# ─── 9. Prediction ─────────────────────────────────────
elif page == "Predictions (Upload Image)":
    st.header("🔍 Predict Disease")

    uploaded = st.file_uploader("Upload Leaf Image")

    if uploaded:
        img = Image.open(uploaded).resize((224,224))
        st.image(img)

        if os.path.exists("models/custom_cnn_best.keras"):
            model = tf.keras.models.load_model("models/custom_cnn_best.keras")

            arr = np.array(img)/255.0
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr)
            class_idx = np.argmax(pred)

            st.success(f"Prediction: {meta['classes'][class_idx]}")
        else:
            st.error("Model not found")

# ─── 10. Logs & Analytics ──────────────────────────────
elif page == "Logs & Analytics":
    st.header("📂 Logs Analysis")

    if os.path.exists("logs/custom_cnn_history.csv"):
        df = pd.read_csv("logs/custom_cnn_history.csv")
        st.dataframe(df)

# ─── 11. Settings ──────────────────────────────────────
elif page == "Settings":
    st.header("⚙️ Settings")
    st.write("Customize dashboard features here")
