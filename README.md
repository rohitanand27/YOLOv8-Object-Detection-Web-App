# 🔍 YOLOv8 Object Detection Web App

A user-friendly web application for object detection using the powerful YOLOv8 models. Built with **Streamlit**, this app supports image uploads, webcam capture, and real-time video object detection.

![YOLOv8 Detection](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg) ![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red) ![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)

---

## 🚀 Features

- 📷 **Image & Webcam Detection**: Upload or capture images for object detection.
- 🎥 **Real-time Video Detection**: Use webcam for live object detection with FPS display.
- 🔧 **Model Selection**: Choose from YOLOv8 variants (`n`, `s`, `m`, `l`, `x`) for speed vs. accuracy.
- 📊 **Detection Insights**:
  - Object counts with interactive bar chart.
  - Most frequently detected class.
  - JSON output viewer.
- 🌗 **Light/Dark Theme Toggle**.
- 📥 **Download Annotated Image** after detection.

---

## 🧰 Technologies Used

- [Streamlit](https://streamlit.io/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Altair](https://altair-viz.github.io/)
- [Pillow (PIL)](https://python-pillow.org/)
- [NumPy](https://numpy.org/)

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yolov8-streamlit-app.git
   cd yolov8-streamlit-app
