import streamlit as st
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
import cv2
import numpy as np
from PIL import Image

# ==========================================
# Fix for PyTorch 2.6+
# ==========================================
torch.serialization.add_safe_globals([
    DetectionModel,
    nn.Sequential,
    Conv
])

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="🌾 Wheat Detection",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Wheat Detection using YOLO")
st.write("Upload an image to detect wheat using your trained YOLO model.")

# ==========================================
# Load Model (cached)
# ==========================================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.to("cpu")
    return model

model = load_model()

# ==========================================
# Upload Image
# ==========================================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("Detect"):

        with st.spinner("Running inference..."):

            img = np.array(image)

            results = model.predict(
                img,
                conf=0.25,
                verbose=False
            )

            annotated = results[0].plot()

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            result_image = Image.fromarray(annotated)

        st.success("Detection Complete!")

        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)