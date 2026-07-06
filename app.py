import streamlit as st
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
import cv2
import numpy as np
from PIL import Image
import time

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

/* Images */

img{
    border-radius:18px;
}

/* Card */

.card{
    background:white;
    padding:20px;
    border-radius:18px;
    box-shadow:0px 10px 25px rgba(0,0,0,.08);
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
🌾 Wheat Detection using YOLOv11
</div>

<div class="subtitle">
Upload an image and let AI automatically detect wheat heads.
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

    st.markdown("## 📤 Upload Image")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

with right:

    st.markdown("## ℹ️ Instructions")

    st.info("""
### Steps

1. Upload your wheat image.

2. Click **Detect Wheat**.

3. Wait a few seconds.

4. View the detection result.

---

Model: **YOLOv11**

Device: **CPU**

Confidence: **0.25**
""")

# ==========================================
# Detection
# ==========================================

if uploaded_file is not None:

    if st.button("🚀 Detect Wheat"):

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        with st.spinner("🤖 AI is analyzing the image..."):

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

        progress.empty()

        st.success("✅ Detection Completed!")

        st.balloons()

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📷 Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("🎯 Detection Result")

            st.image(
                result,
                use_container_width=True
            )

        # Optional: Download Button
        from io import BytesIO

        buffer = BytesIO()
        result.save(buffer, format="PNG")

        st.download_button(
            "⬇️ Download Result",
            data=buffer.getvalue(),
            file_name="prediction.png",
            mime="image/png"
        )

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.markdown(
    "<center><h5>🌾 Built with Streamlit + YOLO + Ultralytics</h5></center>",
    unsafe_allow_html=True
)
