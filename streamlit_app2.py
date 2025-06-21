import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
import requests
from tempfile import NamedTemporaryFile
from ultralytics import YOLO

# --- Download model from Hugging Face if not present ---
MODEL_URL = "https://huggingface.co/24kshah/violencemodel/resolve/main/best_violence_model.h5"
MODEL_PATH = "best_violence_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading violence detection model..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded!")

# --- Load Models ---
model = tf.keras.models.load_model(MODEL_PATH)
yolo_model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is in your repo

# --- Constants ---
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
            resized = cv2.resize(frame, FRAME_SIZE)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            norm = tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype(np.float32))
            frames.append(norm)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.float32))
    cap.release()

    return np.expand_dims(np.array(frames), axis=0)

# --- Video Upload Detection ---
def predict_from_video(video_path):
    video_array = preprocess_video(video_path)
    prediction = model.predict(video_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Play video with YOLO boxes and classification
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(source=frame, conf=0.4, verbose=False)[0]
        frame = results.plot()

        label_color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        cv2.putText(
            frame,
            f"{LABEL_MAP[predicted_class]} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            label_color,
            3,
            cv2.LINE_AA
        )

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()

    return LABEL_MAP[predicted_class], confidence

# --- Webcam Detection ---
def predict_from_webcam():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    buffer = []
    label = "Analyzing..."
    color = (255, 255, 0)

    stop_button = st.button("Stop Webcam", key="stop_webcam_button")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, FRAME_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype(np.float32))
        buffer.append(norm)

        if len(buffer) == NUM_FRAMES:
            input_array = np.expand_dims(np.array(buffer), axis=0)
            prediction = model.predict(input_array)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            label = f"{LABEL_MAP[predicted_class]} ({confidence:.2f})"
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
            buffer = []

        results = yolo_model.predict(source=frame, conf=0.4, verbose=False)[0]
        frame = results.plot()

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

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if stop_button:
            break

    cap.release()

# --- Streamlit UI ---
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
