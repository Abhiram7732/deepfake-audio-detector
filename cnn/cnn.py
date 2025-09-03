
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# Config
# -----------------------------
DATASET_DIR = "release_in_the_wild"
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")

SAMPLE_RATE = 16000
N_MELS = 64
FIXED_FRAMES = 128   # crop/pad to this many frames

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_melspectrogram(file_path, sr=SAMPLE_RATE, n_mels=N_MELS, max_len=FIXED_FRAMES):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        # mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

        # pad or crop along time axis
        if mel_db.shape[1] < max_len:
            pad_width = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]

        return mel_db
    except Exception as e:
        print(f"Could not process {file_path}: {e}")
        return np.zeros((n_mels, max_len))

def load_dataset(real_dir, fake_dir):
    X, y = [], []

    for f in os.listdir(real_dir):
        path = os.path.join(real_dir, f)
        if os.path.isfile(path):
            feat = extract_melspectrogram(path)
            X.append(feat)
            y.append(0)  # real

    for f in os.listdir(fake_dir):
        path = os.path.join(fake_dir, f)
        if os.path.isfile(path):
            feat = extract_melspectrogram(path)
            X.append(feat)
            y.append(1)  # fake

    X = np.array(X)[..., np.newaxis]  # add channel dim
    y = np.array(y)
    return X, y

# -----------------------------
# Load Data
# -----------------------------
X, y = load_dataset(REAL_DIR, FAKE_DIR)
print("Dataset shape:", X.shape, y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# Model (CNN)
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(N_MELS, FIXED_FRAMES, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=12,
    batch_size=16,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
print("Test Accuracy:", accuracy_score(y_test, y_pred))


# -----------------------------
# Save the model
# -----------------------------
model.save("deepfake_audio_model.keras")   # saves in new Keras format
print("Model saved as deepfake_audio_model.keras")