import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import random
import io
import time
from collections import Counter
import altair as alt

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    layout="wide"
)

# --- Custom CSS for Themes and Enhanced UI ---
st.markdown("""<style>
    .stApp { background-color: #f5f5f5; }
    .title { text-align: center; font-size: 2.5em; color: #4A90E2; font-weight: bold; }
    .description { text-align: center; font-size: 1.2em; color: #555; margin-bottom: 20px; }
</style>""", unsafe_allow_html=True)

# --- Theme Toggle ---
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown('<style>.stApp { background-color: #333; color: #fff; }</style>', unsafe_allow_html=True)

# --- Sidebar Settings ---
st.sidebar.header("Settings")
mode = st.sidebar.radio("Detection Mode", ["Image", "Real-time Video"])
model_choice = st.sidebar.selectbox("Select YOLOv8 Model", ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

with st.sidebar.expander("ℹ️ Model Info"):
    st.markdown("""
    - **yolov8n**: Nano - Fastest, least accurate.
    - **yolov8s**: Small - Fast, good for mobile.
    - **yolov8m**: Medium - Balanced.
    - **yolov8l**: Large - More accurate.
    - **yolov8x**: X-Large - Most accurate, slower.
    """)

# --- Load YOLOv8 Model ---
@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(f"{model_name}.pt")

yolo_model = load_yolo_model(model_choice)

# --- Object Detection Function ---
def detect_objects_yolo(image, model, confidence_threshold):
    image_np = np.array(image)
    results = model.predict(source=image_np, conf=confidence_threshold, save=False, show=False)
    draw_image = np.array(image)
    label_counts = {}

    if results and len(results) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        labels = [model.names[int(cls_id)] for cls_id in class_ids]
        label_counts = Counter(labels)

        for i in range(len(boxes)):
            if confidences[i] >= confidence_threshold:
                box = [int(b) for b in boxes[i]]
                label = model.names[int(class_ids[i])]
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                draw_image = cv2.rectangle(draw_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                draw_image = cv2.putText(draw_image, f"{label}: {confidences[i]:.2f}",
                                         (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return Image.fromarray(draw_image), label_counts, results

# --- UI Layout ---
st.markdown('<p class="title">🔍 YOLOv8 Object Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Upload an image or use the webcam for object detection.</p>', unsafe_allow_html=True)

# --- Image Mode ---
if mode == "Image":
    image_source = st.radio("Select Image Source:", ["Upload", "Webcam"])
    col1, col2 = st.columns([1, 1])

    with col1:
        image = None
        if image_source == "Upload":
            uploaded_file = st.file_uploader("📤 Drag and Drop an Image", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            image = st.camera_input("📷 Capture Image from Webcam")
            if image:
                image = Image.open(image).convert("RGB")
                st.image(image, caption="Captured Image", use_container_width=True)

    with col2:
        if st.button("🚀 Detect Objects") and image is not None:
            with st.spinner("Processing... Please wait."):
                start_time = time.time()
                result_image, label_counts, results = detect_objects_yolo(image, yolo_model, confidence_threshold)
                end_time = time.time()

                st.image(result_image, caption="Detected Objects", use_container_width=True)
                st.success(f"✅ Inference Time: {end_time - start_time:.2f} seconds")

                if label_counts:
                    top_label = label_counts.most_common(1)[0]
                    st.info(f"🔝 Most Detected: **{top_label[0]}** ({top_label[1]} times)")

                    chart = alt.Chart(pd.DataFrame({
                        "Object": list(label_counts.keys()),
                        "Count": list(label_counts.values())
                    })).mark_bar().encode(
                        x="Object",
                        y="Count",
                        color="Object"
                    ).interactive()

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No objects detected with the current confidence threshold.")

                if st.checkbox("🔍 Show Raw Detection Output"):
                    st.json(results[0].tojson())

                img_byte_arr = io.BytesIO()
                result_image.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()

                st.download_button(
                    label="📥 Download Image",
                    data=img_byte_arr,
                    file_name="detected_objects.png",
                    mime="image/png"
                )

elif mode == "Real-time Video":
    st.markdown("📹 **Real-time Object Detection via Webcam**")
    run = st.checkbox("▶️ Start Webcam")

    if run:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not open webcam. Please check permissions or availability.")
            else:
                stframe = st.empty()
                fps_display = st.empty()
                frame_count = 0
                frame_skip = 2
                prev_time = time.time()

                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Failed to grab frame.")
                        break

                    if frame_count % frame_skip == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)

                        result_image, _, _ = detect_objects_yolo(pil_image, yolo_model, confidence_threshold)
                        stframe.image(result_image, channels="RGB", use_column_width=True)

                        # Calculate FPS
                        curr_time = time.time()
                        fps = frame_skip / (curr_time - prev_time)
                        prev_time = curr_time
                        fps_display.text(f"⚡ FPS: {fps:.2f}")

                    frame_count += 1

                cap.release()
                st.info("🛑 Webcam stream stopped.")

        except Exception as e:
            st.error(f"❌ Error accessing webcam: {e}")

st.markdown("---")
st.markdown("💡 **Tip:** YOLOv8 works best with clear, high-resolution inputs.")
