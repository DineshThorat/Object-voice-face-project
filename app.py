from collections import defaultdict
import os
import shutil
import librosa
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from detection import load_image, detect_objects, save_detected_objects, draw_boxes, extract_frames, main as detect_objects_main, yolo_model
from face import *

app = Flask(__name__)

app.config['UPLOAD_FOLDER_OBJ'] = 'static/videos'
app.config['DETECTED_FOLDER_OBJ'] = 'static/detected_objects'
app.config['FRAMES_FOLDER_OBJ'] = 'obj_frames'
app.config['UPLOAD_VOICE_FOLDER'] = 'upload_voices'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FRAMES_FOLDER'] = 'frames'
app.config['ENCODINGS_FOLDER'] = 'encodings'
app.config['MATCHES_FOLDER'] = 'matches'

for folder in [app.config['UPLOAD_FOLDER_OBJ'], app.config['DETECTED_FOLDER_OBJ'], app.config['UPLOAD_VOICE_FOLDER'], app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER'], app.config['ENCODINGS_FOLDER'], app.config['MATCHES_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/object-detection')
def object_detection():
    return render_template('object.html')


@app.route('/voice-recognition')
def voice_recognition():
    return render_template('voice.html')


@app.route('/face-recognition')
def face_recognition_page():
    return render_template('face.html')


# Voice Recognition
def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


def match_voice(input_audio_path, model, scaler, threshold=0.25):
    feature = extract_features(input_audio_path).reshape(1, -1)
    feature = scaler.transform(feature)
    prediction = model.predict(feature)
    prediction_prob = model.predict_proba(feature)
    max_prob = np.max(prediction_prob)
    if max_prob > threshold:
        return prediction[0]
    else:
        return "no match"


with open('model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(
                app.config['UPLOAD_VOICE_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            result = match_voice(file_path, ensemble_model, scaler)
            if result != "no match":
                matched_file = result
                return render_template('voice_result.html', matched_file=matched_file)
            else:
                return render_template('voice_result.html', matched_file=None)
    return redirect(url_for('voice_recognition'))


# Face Recognition
@app.route('/upload-face', methods=['POST'])
def upload_face():
    if request.method == 'POST':
        if 'video' not in request.files or 'image' not in request.files:
            return redirect(request.url)

        video = request.files['video']
        image = request.files['image']

        if video.filename == '' or image.filename == '':
            return redirect(request.url)

        if video and image:
            clear_folders()

            video_path = os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(video.filename))
            video.save(video_path)

            image_path = os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
            image.save(image_path)

            extract_frames(video_path, app.config['FRAMES_FOLDER'])
            known_encodings, labels = train_face_recognition_model(
                app.config['FRAMES_FOLDER'])
            encodings_filename = os.path.join(
                app.config['ENCODINGS_FOLDER'], secure_filename(video.filename) + '.pkl')
            save_encodings(known_encodings, labels, encodings_filename)

            match_result = match_image(image_path)
            if not match_result:
                return redirect(url_for('output_page', match_found=False))

            return redirect(url_for('output_page', match_found=True))
    return render_template('index.html')


@app.route('/output')
def output_page():
    matched_image = os.path.exists(os.path.join(
        app.config['MATCHES_FOLDER'], 'matched_image.jpg'))
    matched_frame = os.path.exists(os.path.join(
        app.config['MATCHES_FOLDER'], 'matched_frame.jpg'))
    match_found = matched_image or matched_frame

    return render_template('output.html', matched_image=matched_image, matched_frame=matched_frame, match_found=match_found)


@app.route('/matches/<filename>')
def send_matched_file(filename):
    return send_from_directory(app.config['MATCHES_FOLDER'], filename)


# Object Detection
def process_video(video_path):
    detected_objects_folder = app.config['DETECTED_FOLDER_OBJ']
    if os.path.exists(detected_objects_folder):
        shutil.rmtree(detected_objects_folder)
    os.makedirs(detected_objects_folder)

    extract_frames(video_path, detected_objects_folder)

    for frame_file in os.listdir(detected_objects_folder):
        frame_path = os.path.join(detected_objects_folder, frame_file)
        image = load_image(frame_path)

        labels, boxes, _ = detect_objects(yolo_model, image)

        frame_output_dir = os.path.join(
            detected_objects_folder, os.path.splitext(frame_file)[0])
        os.makedirs(frame_output_dir, exist_ok=True)

        save_detected_objects(image, boxes, frame_output_dir)

        class_names = yolo_model.names
        object_counts = defaultdict(int)
        for label in labels:
            object_counts[class_names[label]] += 1

        with open(os.path.join(frame_output_dir, 'object_counts.txt'), 'w') as f:
            for object_name, count in object_counts.items():
                f.write(f"{object_name} - {count}\n")


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if not os.path.exists(app.config['UPLOAD_FOLDER_OBJ']):
                os.makedirs(app.config['UPLOAD_FOLDER_OBJ'])
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER_OBJ'], "uploaded_video.mp4")
            file.save(file_path)
            process_video(file_path)
            return redirect(url_for('display_results', filename="uploaded_video.mp4"))
    return "Error occurred!"


@app.route('/results/<filename>')
def display_results(filename):
    frames_data = []

    detected_objects_folder = app.config['DETECTED_FOLDER_OBJ']
    if os.path.exists(detected_objects_folder):
        frames_folders = os.listdir(detected_objects_folder)

        for frame_folder in frames_folders:
            frame_path = os.path.join(detected_objects_folder, frame_folder)
            if os.path.isdir(frame_path):
                objects = os.listdir(frame_path)
                objects = [os.path.join(frame_folder, obj).replace(
                    "\\", "/") for obj in objects if obj != 'object_counts.txt']
                with open(os.path.join(frame_path, 'object_counts.txt'), 'r') as f:
                    object_counts = f.readlines()
                frames_data.append({
                    'frame': frame_folder,
                    'objects': objects,
                    'object_counts': object_counts
                })

    return render_template('results.html', frames_data=frames_data)


if __name__ == '__main__':
    app.run(debug=True)
