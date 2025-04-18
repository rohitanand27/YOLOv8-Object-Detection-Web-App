
# 🔍 YOLOv8 Object Detection Web App

An interactive and modern web application for real-time and image-based object detection using **YOLOv8**. Built with **Streamlit**, it allows users to run powerful object detection models directly in the browser with just a few clicks.

<div align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg" />
  <img src="https://img.shields.io/badge/Built%20with-Streamlit-red" />
  <img src="https://img.shields.io/badge/Python-3.8+-yellow.svg" />
</div>

---

## 🖼️ Preview

| Image Mode | Real-Time Mode |
|------------|----------------|
| ![image_preview](https://via.placeholder.com/350x220.png?text=Image+Detection+Preview) | ![video_preview](https://via.placeholder.com/350x220.png?text=Webcam+Detection+Preview) |

> _Replace the preview images above with actual screenshots or demo GIFs._

---

## 🚀 Features

- **🔍 Object Detection** using YOLOv8 on uploaded or webcam-captured images.
- **🎥 Real-Time Detection** using webcam video stream.
- **📊 Visual Insights**:
  - Detected object count and class chart.
  - Most frequent detected object.
  - Raw JSON output toggle for advanced insights.
- **🌓 Theme Toggle**: Light/Dark mode switch.
- **📥 Download Results**: Download annotated image with detected objects.

---

## 📦 Supported YOLOv8 Models

Choose from various YOLOv8 model sizes based on speed vs accuracy tradeoffs:

- `yolov8n`: Nano — Fastest, least accurate.
- `yolov8s`: Small — Good for mobile.
- `yolov8m`: Medium — Balanced.
- `yolov8l`: Large — More accurate.
- `yolov8x`: Extra-large — Most accurate, slower.

---

## 🛠️ Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| [Streamlit](https://streamlit.io/) | Frontend web interface |
| [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) | Object detection backbone |
| [OpenCV](https://opencv.org/) | Image and webcam processing |
| [Pillow](https://python-pillow.org/) | Image loading and manipulation |
| [Altair](https://altair-viz.github.io/) | Interactive object count visualization |
| [NumPy](https://numpy.org/) | Numerical computations |

---

## 📁 Project Structure

```
📂 yolov8-streamlit-app
├── app.py                # Main Streamlit app
├── yolov8n.pt            # Example model weight (place other variants as needed)
├── requirements.txt      # All Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/yolov8-streamlit-app.git
   cd yolov8-streamlit-app
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 Weights**  
   Get `.pt` model files from [Ultralytics YOLOv8 Releases](https://github.com/ultralytics/ultralytics/releases).
   Place the weights (e.g. `yolov8n.pt`) in the project directory.

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Example Use Cases

- Prototype for surveillance or retail monitoring.
- Educational tool to understand object detection.
- Dataset annotation assistance for researchers.
- Real-time smart camera preview.

---

## 🔒 Known Limitations

- Real-time detection may vary in performance depending on system specs.
- Webcam may not work in certain browser/security configurations.

---

## 🌐 Live Demo (Optional)

> Deploy this app using [Streamlit Community Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces/). Add the link below:

**[🧪 Try the Live App Here](https://your-app-link.streamlit.app)**

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.

---

## 🙋‍♂️ Author

**Rohit Anand**  
MCA Student | AI & Software Development Enthusiast  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | 🌐 [Portfolio](https://yourportfolio.com)

---

## ⭐️ Show Your Support

If you like this project, please consider ⭐️ starring the repo and sharing it with others!
