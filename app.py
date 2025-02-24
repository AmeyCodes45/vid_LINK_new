from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import requests
import gdown
import os
import subprocess
from keras.models import load_model
from collections import Counter

app = Flask(__name__)
CORS(app)

# Load FER2013 Emotion Model
FER_MODEL_PATH = "face_model.h5"  # Ensure this file is in the deployment directory
emotion_model = load_model(FER_MODEL_PATH)

# Define FER2013 Emotion Labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to download video from a URL or Google Drive
def download_video(video_url, output_path="downloaded_video.webm"):
    if "drive.google.com" in video_url:
        file_id = video_url.split("/d/")[1].split("/")[0]
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    else:
        response = requests.get(video_url, stream=True)
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    return output_path

# Convert WebM to MP4 using FFmpeg
def convert_webm_to_mp4(input_path, output_path="converted_video.mp4"):
    subprocess.run(["ffmpeg", "-i", input_path, "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", output_path], check=True)
    return output_path

# Analyze video using MediaPipe + FER2013
def analyze_video(video_path):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames, engagement_frames = 0, 0
    emotions_detected = []
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) // 3) or 1  # Adaptive frame skipping
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % frame_interval != 0:
            continue

        total_frames += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            engagement_frames += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = gray_frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (48, 48)) / 255.0
                emotion_prediction = emotion_model.predict(np.expand_dims(resized_face, axis=(0, -1)))[0]
                detected_emotion = EMOTION_LABELS[np.argmax(emotion_prediction)]
                emotions_detected.append(detected_emotion)

    cap.release()
    
    if engagement_frames == 0:
        return {
            "Overall Confidence Level": None,
            "Nervousness": None,
            "Engagement Level": None,
            "Most Frequent Emotion": None,
            "Face Detected": False
        }

    engagement_ratio = engagement_frames / total_frames if total_frames > 0 else 0
    nervousness_score = max(0, (1 - engagement_ratio) * 100)
    nervous_emotions = {"Fear", "Sad", "Surprise"}
    nervous_count = sum(1 for e in emotions_detected if e in nervous_emotions)
    total_emotions = len(emotions_detected)

    if total_emotions > 0:
        emotion_based_nervousness = (nervous_count / total_emotions) * 100
        nervousness_score = max(nervousness_score, emotion_based_nervousness)

    most_frequent_emotion = Counter(emotions_detected).most_common(1)[0][0] if emotions_detected else "Neutral"

    return {
        "Overall Confidence Level": "High" if engagement_ratio > 0.7 else ("Moderate" if engagement_ratio > 0.4 else "Low"),
        "Nervousness": "Low" if nervousness_score < 30 else ("Moderate" if nervousness_score < 60 else "High"),
        "Engagement Level": "High" if engagement_ratio > 0.7 else ("Moderate" if engagement_ratio > 0.4 else "Low"),
        "Most Frequent Emotion": most_frequent_emotion,
        "Face Detected": True
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    video_url = data.get("video_url")
    if not video_url:
        return jsonify({"error": "Missing video_url parameter"}), 400

    try:
        downloaded_video = download_video(video_url)
        converted_video = convert_webm_to_mp4(downloaded_video)
        results = analyze_video(converted_video)
        
        os.remove(downloaded_video)
        os.remove(converted_video)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
