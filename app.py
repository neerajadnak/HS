import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.title("🛋️ Room Detection (Segmentation - No Person)")

st.write("Upload an image. People will NOT be detected.")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolo11m-seg.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file → OpenCV (NO RGB conversion)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Run YOLO directly on image
    results = model(img)

    annotated = img.copy()

    detected_labels = []

    for r in results:
        boxes = r.boxes
        masks = r.masks  # segmentation masks

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # ❌ SKIP PERSON
            if label == "person":
                continue

            detected_labels.append(label)

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Optional: draw segmentation mask
            if masks is not None:
                mask = masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                annotated[mask > 0.5] = annotated[mask > 0.5] * 0.5 + np.array([0,255,0]) * 0.5

    st.image(annotated, caption="Detected (No Person)", use_container_width=True)

    # Show detected objects
    st.subheader("🧾 Detected Objects")

    if detected_labels:
        unique = set(detected_labels)
        for obj in unique:
            count = detected_labels.count(obj)
            st.write(f"**{obj}** : {count}")
    else:
        st.write("No objects detected.")
