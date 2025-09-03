import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ML
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

# ---------- Configuration ----------
DATA_DIR = "release_in_the_wild"   # folder containing 'real' and 'fake' subfolders
SR = 16000                         # sampling rate
N_MFCC = 40
RANDOM_SEED = 42
SVM_SAVE = "svm_mfcc.pkl"
# -----------------------------------

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Utilities ----------
def list_audio_files(base_dir):
    p_real = os.path.join(base_dir, "real")
    p_fake = os.path.join(base_dir, "fake")
    real_files = glob.glob(os.path.join(p_real, "**", "*.wav"), recursive=True)
    fake_files = glob.glob(os.path.join(p_fake, "**", "*.wav"), recursive=True)
    return real_files, fake_files

def load_audio(path, sr=SR):
    wav, fs = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if fs != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=fs, target_sr=sr)
    return wav.astype(np.float32)

def extract_mfcc_mean(wave, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # shape: (n_mfcc,)

# ---------- Build dataset ----------
print("Listing files...")
real_files, fake_files = list_audio_files(DATA_DIR)
print(f"Found {len(real_files)} real files and {len(fake_files)} fake files.")

files = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
random.shuffle(files)

mfcc_X, labels = [], []
print("Extracting MFCC features (may take a while)...")
for path, lbl in tqdm(files):
    try:
        wav = load_audio(path)
        mf = extract_mfcc_mean(wav)
        mfcc_X.append(mf)
        labels.append(lbl)
    except Exception as e:
        print("Failed on", path, e)

mfcc_X = np.vstack(mfcc_X)
labels = np.array(labels)
print("MFCC matrix shape:", mfcc_X.shape)

# ---------- Train & Save SVM ----------
print("Training SVM baseline on MFCC means...")
X_train, X_test, y_train, y_test = train_test_split(
    mfcc_X, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
)
svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED)
svm.fit(X_train, y_train)

# Save model
joblib.dump(svm, SVM_SAVE)
print("Saved SVM to", SVM_SAVE)

# ---------- Load model & Evaluate ----------
svm = joblib.load(SVM_SAVE)
y_pred = svm.predict(X_test)
y_score = svm.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_score)

print("\n📊 SVM MFCC Evaluation Report")
print("================================")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ---------- Plot ROC ----------
fpr, tpr, _ = roc_curve(y_test, y_score)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - SVM on MFCC")
plt.legend()
plt.grid(True)
plt.show()

