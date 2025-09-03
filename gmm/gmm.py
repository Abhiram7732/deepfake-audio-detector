import os
import numpy as np
import librosa
import scipy.fftpack
import warnings
import joblib
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# Config
# -----------------------------
DATASET_DIR = "release_in_the_wild"
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")

SAMPLE_RATE = 16000
N_CQCC = 20   # number of cepstral coefficients
MAX_LEN = 400 # maximum frames per utterance

# -----------------------------
# Feature Extraction (CQCC)
# -----------------------------
def extract_cqcc(file_path, sr=SAMPLE_RATE, n_cqcc=N_CQCC, max_len=MAX_LEN):
    try:
        y, sr = librosa.load(file_path, sr=sr)

        # Pad very short signals (to avoid n_fft warnings in librosa)
        min_len = 2048  # safe padding length for CQT
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)), mode="constant")

        # Constant-Q Transform
        cqt = librosa.cqt(y, sr=sr, hop_length=256, fmin=20,
                          n_bins=96, bins_per_octave=12)
        cqt_power = np.abs(cqt) ** 2

        # Log power spectrum
        log_cqt = librosa.power_to_db(cqt_power)

        # DCT to get cepstral coefficients (like MFCC but on CQT)
        cqcc = scipy.fftpack.dct(log_cqt, axis=0, norm="ortho")[:n_cqcc, :]

        # Normalize
        cqcc = (cqcc - cqcc.mean()) / (cqcc.std() + 1e-9)

        # Pad/crop to fixed length
        if cqcc.shape[1] < max_len:
            pad_width = max_len - cqcc.shape[1]
            cqcc = np.pad(cqcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            cqcc = cqcc[:, :max_len]

        return cqcc.T  # shape: (frames, features)

    except Exception as e:
        print(f"Could not process {file_path}: {e}")
        return np.zeros((max_len, n_cqcc))

# -----------------------------
# Load Dataset with Progress Bar
# -----------------------------
def load_dataset(real_dir, fake_dir):
    X, y = [], []

    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, f))]
    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]

    for path, label in tqdm(all_files, desc="Extracting features", unit="file"):
        feat = extract_cqcc(path)
        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Optional: suppress only librosa warnings (not errors)
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

    print("Loading dataset... (this may take a while)")
    X, y = load_dataset(REAL_DIR, FAKE_DIR)
    print("Dataset shape:", X.shape, y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Flatten utterances into frames for GMM training
    X_train_real = np.vstack([X_train[i] for i in range(len(X_train)) if y_train[i] == 0]).astype(np.float64)
    X_train_fake = np.vstack([X_train[i] for i in range(len(X_train)) if y_train[i] == 1]).astype(np.float64)

    print("Training data shapes:", X_train_real.shape, X_train_fake.shape)

    # Train GMMs (stable settings)
    gmm_real = GaussianMixture(
        n_components=8,
        covariance_type='diag',
        reg_covar=1e-3,
        max_iter=200,
        random_state=42
    )

    gmm_fake = GaussianMixture(
        n_components=8,
        covariance_type='diag',
        reg_covar=1e-3,
        max_iter=200,
        random_state=42
    )

    print("Training GMM for real...")
    gmm_real.fit(X_train_real)

    print("Training GMM for fake...")
    gmm_fake.fit(X_train_fake)

    # -----------------------------
    # Save Models
    # -----------------------------
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(gmm_real, "saved_models/gmm_real.pkl")
    joblib.dump(gmm_fake, "saved_models/gmm_fake.pkl")
    print("✅ Models saved to 'saved_models/'")

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = []
    for i in tqdm(range(len(X_test)), desc="Evaluating", unit="utterance"):
        features = X_test[i]
        ll_real = gmm_real.score(features)  # log-likelihood
        ll_fake = gmm_fake.score(features)
        pred = 0 if ll_real > ll_fake else 1
        y_pred.append(pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print("Test Accuracy:", accuracy_score(y_test, y_pred))