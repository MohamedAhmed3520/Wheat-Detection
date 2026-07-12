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
import os
import time

from io import BytesIO

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
    layout="wide"
)

# ==========================================
# Custom CSS
# ==========================================
st.markdown("""
<style>

/* ---------------- Background ---------------- */

.stApp{
    background: linear-gradient(135deg,#eef7ff,#dff6ff,#ffffff);
}

/* Hide Streamlit Branding */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Container */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Title */

.title{
    text-align:center;
    font-size:55px;
    font-weight:800;
    color:#2d6a4f;
    animation: fadeDown 1s ease;
}

/* Subtitle */

.subtitle{
    text-align:center;
    font-size:20px;
    color:#555;
    margin-bottom:35px;
    animation: fadeUp 1.2s ease;
}

/* Floating Shapes */

.circle1{
    position:fixed;
    width:180px;
    height:180px;
    border-radius:50%;
    background:rgba(34,197,94,.15);
    top:40px;
    left:-70px;
    animation: float 8s infinite;
}

.circle2{
    position:fixed;
    width:130px;
    height:130px;
    border-radius:50%;
    background:rgba(59,130,246,.12);
    bottom:70px;
    right:-40px;
    animation: float2 9s infinite;
}

.circle3{
    position:fixed;
    width:70px;
    height:70px;
    border-radius:50%;
    background:rgba(251,191,36,.2);
    top:300px;
    right:180px;
    animation: float 7s infinite;
}

/* Upload Area */

[data-testid="stFileUploader"]{
    border:2px dashed #2d6a4f;
    border-radius:20px;
    padding:20px;
    background:white;
}

/* Buttons */

.stButton>button{
    width:100%;
    background:#2d6a4f;
    color:white;
    border-radius:15px;
    font-size:18px;
    font-weight:bold;
    transition:.3s;
}

.stButton>button:hover{
    transform:scale(1.04);
    box-shadow:0 8px 18px rgba(0,0,0,.2);
}

/* Cards */

.card{
    background:white;
    padding:20px;
    border-radius:18px;
    box-shadow:0px 10px 25px rgba(0,0,0,.08);
}

/* Images */

img{
    border-radius:18px;
}

/* Animations */

@keyframes float{
0%{transform:translateY(0);}
50%{transform:translateY(-25px);}
100%{transform:translateY(0);}
}

@keyframes float2{
0%{transform:translateY(0);}
50%{transform:translateY(20px);}
100%{transform:translateY(0);}
}

@keyframes fadeDown{
from{opacity:0;transform:translateY(-40px);}
to{opacity:1;transform:translateY(0);}
}

@keyframes fadeUp{
from{opacity:0;transform:translateY(40px);}
to{opacity:1;transform:translateY(0);}
}

</style>

<div class="circle1"></div>
<div class="circle2"></div>
<div class="circle3"></div>

<div class="title">
🌾 Wheat Detection using YOLOv8
</div>

<div class="subtitle">
Upload an image or a video and let AI automatically detect wheat heads.
</div>

""", unsafe_allow_html=True)

# ==========================================
# Load Model
# ==========================================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.to("cpu")
    return model

model = load_model()

# ==========================================
# Layout
# ==========================================

left, right = st.columns([1,1])

with left:

    st.markdown("## 📤 Upload Image / Video")

    uploaded_file = st.file_uploader(
        "Upload an Image or Video",
        type=["jpg","jpeg","png","mp4","avi","mov","mkv"],
        label_visibility="collapsed"
        )

image = None
video_path = None
is_video = False

if uploaded_file is not None:

    extension = uploaded_file.name.split(".")[-1].lower()

    # ---------------- Image ----------------

    if extension in ["jpg","jpeg","png"]:

        image = Image.open(uploaded_file).convert("RGB")

        with left:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

    # ---------------- Video ----------------

    else:

        is_video = True

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{extension}"
        )

        temp_video.write(uploaded_file.read())
        temp_video.close()

        video_path = temp_video.name

        with left:
            st.video(video_path)

# ==========================================
# Instructions
# ==========================================

with right:

    st.markdown("## ℹ️ Instructions")

    st.info("""

### Supported Files

✅ JPG

✅ PNG

✅ MP4

✅ AVI

✅ MOV

✅ MKV

---

1. Upload an image or video.

2. Click Detect Wheat.

3. Wait for processing.

4. Download the result.

---

Model: **YOLOv8**

Device: **CPU**

""")

# ==========================================
# Helper Function
# ==========================================

def count_detections(results):
    total = 0

    for r in results:
        if r.boxes is not None:
            total += len(r.boxes)

    return total
# ==========================================
# Detection
# ==========================================

if uploaded_file is not None:

    if st.button("🚀 Detect Wheat"):

        progress = st.progress(0)

        start_time = time.time()

        # ======================================================
        # IMAGE DETECTION
        # ======================================================

        if not is_video:

            with st.spinner("🤖 AI is analyzing image..."):

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

                result = Image.fromarray(annotated)

                total_boxes = count_detections(results)

            progress.progress(100)

            elapsed = time.time() - start_time

            st.success("✅ Detection Completed!")

            st.balloons()

            st.markdown("---")

            c1, c2 = st.columns(2)

            with c1:

                st.subheader("📷 Original Image")

                st.image(
                    image,
                    use_container_width=True
                )

            with c2:

                st.subheader("🎯 Detection Result")

                st.image(
                    result,
                    use_container_width=True
                )

            st.markdown("### 📊 Statistics")

            a, b, c = st.columns(3)

            a.metric("🌾 Wheat Heads", total_boxes)
            b.metric("⏱ Time", f"{elapsed:.2f} sec")
            c.metric("💻 Device", "CPU")

            buffer = BytesIO()

            result.save(
                buffer,
                format="PNG"
            )

            st.download_button(
                "⬇️ Download Result",
                data=buffer.getvalue(),
                file_name="prediction.png",
                mime="image/png"
            )

        # ======================================================
        # VIDEO DETECTION
        # ======================================================

        else:

            with st.spinner("🎥 Processing Video..."):

                cap = cv2.VideoCapture(video_path)

                fps = cap.get(cv2.CAP_PROP_FPS)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                output_path = "prediction_video.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                frame_counter = 0

                total_detections = 0

                preview = st.empty()

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.predict(
                        frame,
                        conf=0.25,
                        verbose=False
                    )

                    total_detections += count_detections(results)

                    annotated = results[0].plot()

                    writer.write(annotated)

                    if frame_counter % 10 == 0:

                        rgb = cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        )

                        preview.image(
                            rgb,
                            caption=f"Processing Frame {frame_counter}",
                            use_container_width=True
                        )

                    frame_counter += 1

                    if total_frames > 0:

                        percent = int(
                            frame_counter /
                            total_frames *
                            100
                        )

                        progress.progress(
                            min(percent, 100)
                        )

                cap.release()

                writer.release()

                preview.empty()

            elapsed = time.time() - start_time

            progress.empty()

            st.success("✅ Video Detection Completed!")

            st.balloons()

            st.markdown("---")

            st.subheader("🎥 Processed Video")

            video_file = open(
                output_path,
                "rb"
            )

            video_bytes = video_file.read()

            st.video(video_bytes)

            st.markdown("### 📊 Statistics")

            x, y, z = st.columns(3)

            x.metric(
                "🌾 Total Detections",
                total_detections
            )

            y.metric(
                "🎞 Frames",
                frame_counter
            )

            z.metric(
                "⏱ Time",
                f"{elapsed:.2f} sec"
            )

            with open(
                output_path,
                "rb"
            ) as file:

                st.download_button(

                    "⬇️ Download Processed Video",

                    data=file,

                    file_name="prediction_video.mp4",

                    mime="video/mp4"
                )

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.markdown(
    """
    <center>
        <h5>
            🌾 Built with Streamlit + YOLOv8 + Ultralytics
        </h5>
    </center>
    """,
    unsafe_allow_html=True
)

# ==========================================
# Cleanup Temporary Video
# ==========================================

if is_video:

    try:

        if os.path.exists(video_path):
            os.remove(video_path)

    except:
        pass
