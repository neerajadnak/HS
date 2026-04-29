import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

st.title("🛋️ Room Object Detection (No Person)")

st.write("Upload a room image. The app will detect all objects except people.")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(frame, caption="Uploaded Image", use_container_width=True)

    # Run YOLO
    results = model(frame)

    annotated_frame = frame.copy()
    detected_labels = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # ❌ SKIP PERSON
            if label == "person":
                continue

            detected_labels.append(label)

            # Get box coords
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated_frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    st.image(annotated_frame, caption="Detected Objects (No Person)", use_container_width=True)

    # Count objects
    st.subheader("🧾 Detected Objects Summary")

    if detected_labels:
        counts = Counter(detected_labels)
        for obj, count in counts.items():
            st.write(f"**{obj}** : {count}")
    else:
        st.write("No objects detected.")
