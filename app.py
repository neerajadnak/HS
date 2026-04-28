import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Load YOLO Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

st.title("🚀 Object Detection App (YOLO)")

# -----------------------------
# Session State for objects
# -----------------------------
if "detected_objects" not in st.session_state:
    st.session_state.detected_objects = set()

# -----------------------------
# Mode Selection
# -----------------------------
mode = st.radio("Choose Input Type", ["📸 Camera", "🎥 Video Upload"])

# =============================
# 📸 CAMERA INPUT
# =============================
if mode == "📸 Camera":
    image_file = st.camera_input("Take a picture")

    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame)

        annotated = frame.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Skip person
                if label == "person":
                    continue

                st.session_state.detected_objects.add(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated, caption="Detected Image")

# =============================
# 🎥 VIDEO UPLOAD
# =============================
elif mode == "🎥 Video Upload":
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if video_file is not None:
        tfile = open("temp.mp4", "wb")
        tfile.write(video_file.read())

        cap = cv2.VideoCapture("temp.mp4")

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = frame.copy()

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    if label == "person":
                        continue

                    st.session_state.detected_objects.add(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(annotated, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated)

        cap.release()

# -----------------------------
# Output Detected Objects
# -----------------------------
st.subheader("📋 Detected Objects (Summary)")
st.write(list(st.session_state.detected_objects))
