from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

from src.metrics import dice_score, iou
from src.predict import predict_image
from src.utils import CLASS_NAMES, MODEL_DIR, PLOTS_DIR


st.set_page_config(page_title="Glaucoma Detection", layout="wide")


@st.cache_resource
def get_models():
    unet_path = MODEL_DIR / "unet_finetuned.h5"
    cnn_path = MODEL_DIR / "cnn_classifier.h5"
    if not unet_path.exists() or not cnn_path.exists():
        return None, None
    unet = load_model(unet_path, custom_objects={"dice_score": dice_score, "iou": iou})
    cnn = load_model(cnn_path)
    return unet, cnn


def decode_upload(uploaded_file):
    bytes_data = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_temp_upload(uploaded_file):
    temp_dir = Path("outputs") / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded_file.name
    temp_path.write_bytes(uploaded_file.getbuffer())
    return temp_path


page = st.sidebar.radio(
    "Navigation",
    ["Home", "Upload Eye Image", "Prediction", "Metrics"],
)

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if page == "Home":
    st.title("Retrained U-Net Neural Network to Detect Glaucoma Retinal Disorder with OCTA Eye Images")
    st.write("Transfer Learning U-Net segmentation followed by a CNN classifier.")

elif page == "Upload Eye Image":
    st.title("Upload Eye Image")
    uploaded = st.file_uploader(
        "Upload retinal image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
    if uploaded is not None:
        st.session_state.uploaded_path = save_temp_upload(uploaded)
        st.image(decode_upload(uploaded), caption="Uploaded image", use_column_width=True)

elif page == "Prediction":
    st.title("Prediction")
    unet_model, cnn_model = get_models()
    if st.session_state.uploaded_path is None:
        st.info("Upload an eye image first.")
    elif unet_model is None or cnn_model is None:
        st.error("Train and save models before prediction.")
    else:
        result = predict_image(st.session_state.uploaded_path, unet_model, cnn_model)
        st.session_state.prediction = result
        col1, col2 = st.columns(2)
        with col1:
            st.image(result["image"], caption="Uploaded image", use_column_width=True)
        with col2:
            st.image(result["segmented"], caption="Segmented image", use_column_width=True)
        st.subheader(f"Predicted class: {result['predicted_class']}")

elif page == "Metrics":
    st.title("Metrics")
    result = st.session_state.prediction
    if result is not None:
        st.metric("Confidence score", f"{result['confidence'] * 100:.2f}%")
        st.write(
            {CLASS_NAMES[index]: float(value) for index, value in enumerate(result["probabilities"])}
        )
    else:
        st.info("Run a prediction to see confidence score.")

    accuracy_plot = PLOTS_DIR / "cnn_classifier_accuracy.png"
    loss_plot = PLOTS_DIR / "cnn_classifier_loss.png"
    col1, col2 = st.columns(2)
    with col1:
        if accuracy_plot.exists():
            st.image(str(accuracy_plot), caption="Model accuracy")
    with col2:
        if loss_plot.exists():
            st.image(str(loss_plot), caption="Model loss")
