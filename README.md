# 🌾 Wheat Detection using YOLOv8

A deep learning project for detecting wheat heads in field images using **YOLOv8**. This project was developed as part of one of Kaggle's challenging computer vision competitions, aiming to accurately localize wheat heads under different lighting conditions, scales, and levels of occlusion.

The project includes a modern **Streamlit** web application that allows users to upload an image and visualize the detection results instantly.

---

## 🚀 Live Demo

👉 **Try the application here:**

**https://cf49by5t8hbfuchmq7y9zk.streamlit.app/**

---

## 📌 Features

- 🌾 Wheat head detection using **YOLOv8**
- 📷 Upload any field image for inference
- ⚡ Fast prediction with Ultralytics YOLO
- 🎯 Bounding box visualization
- 🎨 Modern Streamlit user interface
- 📥 Download prediction results
- 💻 CPU compatible deployment

---

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Pillow

---
## Data Set Link 
https://www.kaggle.com/competitions/global-wheat-detection/data 
## 📂 Project Structure

```text
Wheat-Detection/
│
├── app.py                 # Streamlit application
├── best.pt                # Trained YOLOv8 model
├── requirements.txt
├── README.md
└── images/                # (Optional) Sample images
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Wheat-Detection.git
cd Wheat-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🖼️ How It Works

1. Upload a wheat field image.
2. The trained YOLOv8 model performs object detection.
3. Wheat heads are identified with bounding boxes.
4. The annotated image is displayed.
5. Download the prediction if desired.

---

## 📊 Model

- **Architecture:** YOLOv8
- **Framework:** Ultralytics
- **Task:** Object Detection
- **Deployment:** Streamlit

---

## 🎯 Future Improvements

- Train on larger datasets
- Improve detection accuracy for dense wheat fields
- Deploy GPU inference
- Add confidence threshold control
- Support video inference

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Mohamed Ahmed**

If you found this project helpful, consider giving it a ⭐ on GitHub!
