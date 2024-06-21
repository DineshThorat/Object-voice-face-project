import os
import pickle
import face_recognition


def train_face_recognition_model(data_dir):
    known_encodings = []
    labels = []

    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                if len(encoding) > 0:
                    known_encodings.append(encoding[0])
                    labels.append(label_name)

    return known_encodings, labels


data_dir = "Photo Dataset"
known_encodings, labels = train_face_recognition_model(data_dir)

with open('face_encodings.pkl', 'wb') as f:
    pickle.dump((known_encodings, labels), f)

print("Training completed and data saved to face_encodings.pkl")
