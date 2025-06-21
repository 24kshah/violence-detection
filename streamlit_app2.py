import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
from tempfile import NamedTemporaryFile
from ultralytics import YOLO

# Load models
model = tf.keras.models.load_model("best_violence_model.h5")
yolo_model = YOLO("yolov8n.pt")  # Replace with your own YOLO model if available

# Constants
NUM_FRAMES = 30
FRAME_SIZE = (224, 224)
LABEL_MAP = {0: "NonViolence", 1: "Violence"}

# --- Preprocessing ---
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=NUM_FRAMES, dtype=np.int32)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, FRAME_SIZE)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
                rgb_frame.astype(np.float32)
            )
            frames.append(preprocessed)
        else:
            frames.append(np.zeros((FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.float32))
    cap.release()

    if len(frames) < NUM_FRAMES:
        padding = np.zeros((NUM_FRAMES - len(frames), *FRAME_SIZE, 3), dtype=np.float32)
        frames.extend(padding.tolist())

    return np.expand_dims(np.array(frames), axis=0)

# --- Predict from uploaded video ---
def predict_from_video(video_path):
    video_array = preprocess_video(video_path)
    prediction = model.predict(video_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Stream video with bounding boxes
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(source=frame, conf=0.4, verbose=False)[0]
        frame = results.plot()

        # Add classification label
        cv2.putText(
            frame,
            f"{LABEL_MAP[predicted_class]} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0) if predicted_class == 0 else (0, 0, 255),
            3,
            cv2.LINE_AA
        )

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

    return LABEL_MAP[predicted_class], confidence

# --- Predict from webcam ---
def predict_from_webcam():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    buffer = []
    label = "Analyzing..."
    color = (255, 255, 0)  # Yellow

    stop_button = st.button("Stop Webcam", key="stop_webcam_button")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, FRAME_SIZE)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
            rgb_frame.astype(np.float32)
        )
        buffer.append(preprocessed)

        # Every 30 frames → run classification
        if len(buffer) == NUM_FRAMES:
            input_array = np.expand_dims(np.array(buffer), axis=0)
            prediction = model.predict(input_array)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            label = f"{LABEL_MAP[predicted_class]} ({confidence:.2f})"
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
            buffer = []

        # YOLO detection
        results = yolo_model.predict(source=frame, conf=0.4, verbose=False)[0]
        frame = results.plot()

        # Add classification label
        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA
        )

        # Show stream
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if stop_button:
            break

    cap.release()
    #cv2.destroyAllWindows()

# --- Streamlit App ---
st.title("🔍 Violence Detection System")
option = st.radio("Choose Input Type", ("Upload Video", "Webcam Feed"))

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])
    if uploaded_file is not None:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        st.video(uploaded_file)
        if st.button("Predict Violence", key="predict_button"):
            label, confidence = predict_from_video(video_path)
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2f}")

elif option == "Webcam Feed":
    st.warning("Click 'Stop Webcam' to end live detection.")
    predict_from_webcam()
