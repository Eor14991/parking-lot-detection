# streamlit_app.py
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# -----------------------------
# Load YOLOv8 model
# -----------------------------
model = YOLO("weights/best.pt")

# Class colors
class_colors = {"empty": (0, 255, 0), "occupay": (255, 0, 0)}

st.title("🚗 Parking Lot Detection with YOLOv8")

# -----------------------------
# Choose input type
# -----------------------------
mode = st.radio("Select input type:", ["Image", "Video"])

# -----------------------------
# IMAGE MODE
# -----------------------------
if mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Predict
        results = model.predict(frame, conf=0.5, save=False, show=False)
        result = results[0]

        # Count classes
        counts = {"empty": 0, "occupay": 0}
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            color = class_colors.get(label, (255, 255, 0))
            if label in counts:
                counts[label] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Overlay counts
        cv2.putText(frame, f"Empty: {counts['empty']}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, class_colors["empty"], 4)
        cv2.putText(frame, f"Occupay: {counts['occupay']}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, class_colors["occupay"], 4)

        # Show result
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB")

# -----------------------------
# VIDEO MODE
# -----------------------------
elif mode == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.image([])  # placeholder for frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict
            results = model.predict(frame, conf=0.5, save=False, show=False)
            result = results[0]

            # Count classes
            counts = {"empty": 0, "occupay": 0}
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                color = class_colors.get(label, (255, 255, 0))
                if label in counts:
                    counts[label] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Overlay counts
            cv2.putText(frame, f"Empty: {counts['empty']}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, class_colors["empty"], 4)
            cv2.putText(frame, f"Occupay: {counts['occupay']}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, class_colors["occupay"], 4)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        cap.release()
