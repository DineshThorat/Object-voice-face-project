import os
import numpy as np
import librosa
import soundfile as sf
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = 'Audio dataset'


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


def augment_data(y, sr):
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise

    y_shift = np.roll(y, sr // 10)

    y_stretch = librosa.effects.time_stretch(y, rate=1.1)

    return [y, y_noise, y_shift, y_stretch]


def load_data(data_path):
    features = []
    file_paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.mp3'):
                print(f"Processing file: {file_path}")
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    augmented_audios = augment_data(y, sr)
                    for audio in augmented_audios:
                        temp_path = 'temp.wav'
                        sf.write(temp_path, audio, sr)
                        feature = extract_features(temp_path)
                        features.append(feature)
                        file_paths.append(file_path)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return np.array(features), np.array(file_paths)


X, file_paths = load_data(DATA_PATH)

if len(X) == 0:
    raise ValueError(
        "No features extracted. Please check the data loading process.")

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_encoded = file_paths

model1 = SVC(kernel='linear', probability=True)
model2 = RandomForestClassifier(n_estimators=100)
ensemble_model = VotingClassifier(
    estimators=[('svc', model1), ('rf', model2)], voting='soft')

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=0)

ensemble_model.fit(X_train, y_train)

test_predictions = ensemble_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test accuracy: {test_accuracy}")

with open('model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
