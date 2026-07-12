import streamlit as st
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

import cv2
import numpy as np

from PIL import Image

import tempfile
import shutil
import os
import time
import glob

from io import BytesIO

# ==========================================
# PyTorch Fix
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
    layout="wide"
)

# ==========================================
# Load YOLO Model
# ==========================================

@st.cache_resource
def load_model():

    model = YOLO("best.pt")

    model.to("cpu")

    return model


model = load_model()

# ==========================================
# Helper Functions
# ==========================================

def count_boxes(results):

    count = 0

    for r in results:

        if r.boxes is not None:

            count += len(r.boxes)

    return count


def clear_predict_folder():

    if os.path.exists("runs/detect"):

        shutil.rmtree("runs/detect")
# ==========================================
# Layout
# ==========================================

left, right = st.columns([1, 1])

image = None
video_path = None
is_video = False

with left:

    st.markdown("## 📤 Upload Image or Video")

    uploaded_file = st.file_uploader(
        "Choose an image or video",
        type=[
            "jpg",
            "jpeg",
            "png",
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        extension = uploaded_file.name.split(".")[-1].lower()

        # ==========================
        # IMAGE
        # ==========================

        if extension in ["jpg", "jpeg", "png"]:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                width="stretch"
            )

        # ==========================
        # VIDEO
        # ==========================

        else:

            is_video = True

            temp_dir = tempfile.mkdtemp()

            video_path = os.path.join(
                temp_dir,
                uploaded_file.name
            )

            with open(video_path, "wb") as f:

                f.write(uploaded_file.read())

            st.video(video_path)

with right:

    st.markdown("## ℹ️ Instructions")

    st.info("""

### Supported Files

🖼 JPG

🖼 PNG

🎥 MP4

🎥 AVI

🎥 MOV

🎥 MKV

---

### Steps

1️⃣ Upload an image or video

2️⃣ Click Detect Wheat

3️⃣ Wait for AI processing

4️⃣ Download the result

---

Model : **YOLOv11**

Device : **CPU**

Confidence : **0.25**

""")

# ==========================================
# Detect Button
# ==========================================

detect = False

if uploaded_file is not None:

    detect = st.button(
        "🚀 Detect Wheat",
        use_container_width=True
    )

# ==========================================
# Progress
# ==========================================

progress = st.empty()

status = st.empty()

preview = st.empty()
# ==========================================
# Detection
# ==========================================

if detect:

    clear_predict_folder()

    start_time = time.time()

    with st.spinner("🤖 AI is processing..."):

        # ===================================================
        # IMAGE
        # ===================================================

        if not is_video:

            img = np.array(image)

            results = model.predict(

                source=img,

                conf=0.25,

                save=False,

                verbose=False

            )

            annotated = results[0].plot()

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            result_image = Image.fromarray(
                annotated
            )

            total_boxes = count_boxes(results)

            elapsed = time.time() - start_time

            st.success("✅ Detection Completed!")

            c1, c2 = st.columns(2)

            with c1:

                st.subheader("📷 Original")

                st.image(
                    image,
                    width="stretch"
                )

            with c2:

                st.subheader("🎯 Prediction")

                st.image(
                    result_image,
                    width="stretch"
                )

            st.markdown("### 📊 Statistics")

            a, b, c = st.columns(3)

            a.metric(
                "🌾 Wheat Heads",
                total_boxes
            )

            b.metric(
                "⏱ Time",
                f"{elapsed:.2f}s"
            )

            c.metric(
                "💻 Device",
                "CPU"
            )

            buffer = BytesIO()

            result_image.save(
                buffer,
                format="PNG"
            )

            st.download_button(

                "⬇️ Download Result",

                buffer.getvalue(),

                file_name="prediction.png",

                mime="image/png"

            )

        # ===================================================
        # VIDEO
        # ===================================================

        else:

            status.info("🎥 Processing video...")

            model.predict(

                source=video_path,

                conf=0.25,

                save=True,

                stream=False,

                verbose=False

            )

            elapsed = time.time() - start_time

            videos = glob.glob(
                "runs/detect/predict/*.mp4"
            )

            if len(videos) == 0:

                videos = glob.glob(
                    "runs/detect/predict*/*.mp4"
                )

            if len(videos) == 0:

                st.error(
                    "❌ Output video was not found."
                )

            else:

                output_video = videos[-1]

                st.success(
                    "✅ Video Detection Completed!"
                )

                st.balloons()

                st.subheader(
                    "🎥 Detection Result"
                )

                st.video(
                    output_video
                )

                st.markdown(
                    "### 📊 Statistics"
                )

                x, y = st.columns(2)

                x.metric(
                    "⏱ Time",
                    f"{elapsed:.2f}s"
                )

                y.metric(
                    "Model",
                    "YOLOv11"
                )

                with open(
                    output_video,
                    "rb"
                ) as f:

                    st.download_button(

                        "⬇️ Download Processed Video",

                        data=f,

                        file_name="prediction.mp4",

                        mime="video/mp4"

                    )# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.markdown(
    """
    <center>

    <h4 style="color:#2d6a4f;">
        🌾 Wheat Detection using YOLOv11
    </h4>

    <p>
        Built with ❤️ using Streamlit & Ultralytics YOLO
    </p>

    </center>
    """,
    unsafe_allow_html=True
)

# ==========================================
# Cleanup Temporary Files
# ==========================================

try:

    if is_video:

        if video_path is not None:

            temp_folder = os.path.dirname(video_path)

            if os.path.exists(temp_folder):

                shutil.rmtree(
                    temp_folder,
                    ignore_errors=True
                )

except Exception:

    pass
