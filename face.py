import os
import shutil
import cv2
import face_recognition
import pickle

UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
ENCODINGS_FOLDER = 'encodings'
MATCHES_FOLDER = 'matches'


def match_image(image_path):
    image = face_recognition.load_image_file(image_path)
    image_encodings = face_recognition.face_encodings(image)
    if len(image_encodings) == 0:
        clear_matches()
        return False

    image_encoding = image_encodings[0]

    encodings_files = os.listdir(ENCODINGS_FOLDER)
    for enc_file in encodings_files:
        encodings_path = os.path.join(ENCODINGS_FOLDER, enc_file)
        known_encodings, labels = load_encodings(encodings_path)

        matches = face_recognition.compare_faces(
            known_encodings, image_encoding)
        if any(matches):
            match_index = matches.index(True)
            matched_frame = labels[match_index]
            matched_frame_path = os.path.join(
                FRAMES_FOLDER, matched_frame)

            matched_image_path = os.path.join(
                MATCHES_FOLDER, 'matched_image.jpg')
            shutil.copy(image_path, matched_image_path)
            matched_frame_copy_path = os.path.join(
                MATCHES_FOLDER, 'matched_frame.jpg')
            shutil.copy(matched_frame_path, matched_frame_copy_path)
            return True

    clear_matches()
    return False


def extract_frames(video_path, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    success = True

    while success:
        success, frame = video_capture.read()
        if success:
            if frame_count % int(fps) == 0:
                resized_frame = cv2.resize(frame, (700, 450))
                frame_path = os.path.join(
                    output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, resized_frame)
            frame_count += 1

    video_capture.release()


def train_face_recognition_model(data_dir):
    known_encodings = []
    labels = []

    for image_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if len(encoding) > 0:
            known_encodings.append(encoding[0])
            labels.append(image_name)

    return known_encodings, labels


def save_encodings(encodings, labels, filename):
    with open(filename, 'wb') as file:
        pickle.dump((encodings, labels), file)


def load_encodings(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def clear_folders():
    if os.path.exists(ENCODINGS_FOLDER):
        shutil.rmtree(ENCODINGS_FOLDER)
    if os.path.exists(FRAMES_FOLDER):
        shutil.rmtree(FRAMES_FOLDER)
    if os.path.exists(MATCHES_FOLDER):
        shutil.rmtree(MATCHES_FOLDER)
    os.makedirs(ENCODINGS_FOLDER)
    os.makedirs(FRAMES_FOLDER)
    os.makedirs(MATCHES_FOLDER)


def clear_matches():
    if os.path.exists(MATCHES_FOLDER):
        shutil.rmtree(MATCHES_FOLDER)
    os.makedirs(MATCHES_FOLDER)
