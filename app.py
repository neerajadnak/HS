import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # auto-downloads

model = load_model()

st.title("🚀 Object Detection (No OpenCV)")

# -----------------------------
# Session state
# -----------------------------
if "detected_objects" not in st.session_state:
    st.session_state.detected_objects = set()

# -----------------------------
# Mode Selection
# -----------------------------
mode = st.radio("Choose Input Type", ["📸 Camera", "🎥 Upload Image"])

# =============================
# 📸 CAMERA INPUT
# =============================
if mode == "📸 Camera":
    image_file = st.camera_input("Take a picture")

# =============================
# 🎥 IMAGE UPLOAD
# =============================
else:
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# =============================
# PROCESS IMAGE
# =============================
if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    img_array = np.array(image)

    results = model(img_array)

    # Draw using PIL
    draw = ImageDraw.Draw(image)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Skip person
            if label == "person":
                continue

            st.session_state.detected_objects.add(label)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, y1 - 10), label, fill="green")

    st.image(image, caption="Detected Image")

# -----------------------------
# OUTPUT
# -----------------------------
st.subheader("📋 Detected Objects (Summary)")
st.write(list(st.session_state.detected_objects))
