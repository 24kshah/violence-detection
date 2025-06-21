from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

# Flask App
app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("best_violence_model.h5")

# Constants
NUM_FRAMES = 30
FRAME_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"mp4"}

# Label Map
LABEL_MAP = {0: "NonViolence", 1: "Violence"}

# Check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess uploaded video
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
            preprocessed_frame = tf.keras.applications.mobilenet_v2.preprocess_input(rgb_frame.astype(np.float32))
            frames.append(preprocessed_frame)
        else:
            frames.append(np.zeros((FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.float32))
    cap.release()

    if len(frames) < NUM_FRAMES:
        padding = np.zeros((NUM_FRAMES - len(frames), *FRAME_SIZE, 3), dtype=np.float32)
        frames.extend(padding.tolist())

    return np.expand_dims(np.array(frames), axis=0)  # Shape: (1, NUM_FRAMES, 224, 224, 3)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Preprocess and predict
        video_array = preprocess_video(filepath)
        prediction = model.predict(video_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        result = {
            "label": LABEL_MAP[predicted_class],
            "confidence": f"{confidence:.2f}"
        }
        return jsonify(result)
    else:
        return jsonify({"error": "Invalid file format. Only MP4 allowed."})

if __name__ == "__main__":
    app.run(debug=True)
