
# SVM , CNN , GMM , graphs with mp3 
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import scipy.fftpack

# -----------------------------
# Config
# -----------------------------
SAMPLE_RATE = 16000
N_MELS = 64
FIXED_FRAMES = 128
MFCC_COUNT = 40

SVM_PATH = "svm/svm_mfcc.pkl"
KERAS_PATH = "cnn/deepfake_audio_model.keras"
GMM_REAL_PATH = "gmm/gmm_real.pkl"
GMM_FAKE_PATH = "gmm/gmm_fake.pkl"

# Pre-computed test accuracies
MODEL_ACCURACIES = {
    "SVM (MFCC)": 0.965,
    "CNN (Mel-spectrogram)": 0.996,
    "CQCC + GMM (LLR)": 0.904
}

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Audio Deepfake Detector", 
    page_icon="🎙", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .algorithm-section {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-positive {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 0.7rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .result-negative {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 0.7rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .compact-plot {
        max-width: 900px;
        margin: 0 auto;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🎙️ Audio Deepfake Detector</h1>
    <p>Advanced AI-powered detection using SVM, CNN, and GMM algorithms</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Lazy model loading (cache)
# -----------------------------
@st.cache_resource
def load_svm():
    return joblib.load(SVM_PATH)

@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model(KERAS_PATH)

@st.cache_resource
def load_gmm():
    return joblib.load(GMM_REAL_PATH), joblib.load(GMM_FAKE_PATH)

# -----------------------------
# Feature extractors
# -----------------------------
def extract_mfcc(file, sr=SAMPLE_RATE, n_mfcc=MFCC_COUNT):
    y, sr = librosa.load(file, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)
    return mfcc_scaled, y, sr, mfcc

def extract_melspectrogram(file, sr=SAMPLE_RATE, n_mels=N_MELS, max_len=FIXED_FRAMES):
    y, sr = librosa.load(file, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    if mel_db_norm.shape[1] < max_len:
        pad_width = max_len - mel_db_norm.shape[1]
        mel_db_norm = np.pad(mel_db_norm, ((0, 0), (0, pad_width)), mode="constant")
        mel_db_for_plot = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db_norm = mel_db_norm[:, :max_len]
        mel_db_for_plot = mel_db[:, :max_len]

    return mel_db_norm, y, sr, mel_db_for_plot

def extract_cqcc(file, sr=SAMPLE_RATE, n_cqcc=20, max_len=400):
    y, sr = librosa.load(file, sr=sr, mono=True)

    min_len = 2048
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)), mode="constant")

    cqt = librosa.cqt(y, sr=sr, hop_length=256, fmin=20, n_bins=96, bins_per_octave=12)
    cqt_power = np.abs(cqt) ** 2
    cqt_db = librosa.power_to_db(cqt_power)
    cqcc = scipy.fftpack.dct(cqt_db, axis=0, norm="ortho")[:n_cqcc, :]
    cqcc_norm = (cqcc - cqcc.mean()) / (cqcc.std() + 1e-9)

    if cqcc_norm.shape[1] < max_len:
        pad_width = max_len - cqcc_norm.shape[1]
        cqcc_norm = np.pad(cqcc_norm, ((0, 0), (0, pad_width)), mode="constant")
        cqt_db_for_plot = np.pad(cqt_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        cqcc_norm = cqcc_norm[:, :max_len]
        cqt_db_for_plot = cqt_db[:, :max_len]

    features_for_gmm = cqcc_norm.T
    return features_for_gmm, y, sr, cqt_db_for_plot, cqcc_norm

# -----------------------------
# Main Interface Layout
# -----------------------------

# Create two main columns for input and model info
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=["wav", "mp3"],
        help="Supported formats: WAV, MP3"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        # Get file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size} bytes"
        }
        with st.expander("📋 File Details"):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="algorithm-section">', unsafe_allow_html=True)
    st.subheader("🧠 Detection Algorithm")
    algo = st.radio(
        "Select detection method:",
        (
            "SVM (MFCC)",
            "CNN (Mel-spectrogram)", 
            "CQCC + GMM (LLR)",
        ),
        help="Each algorithm uses different audio features for detection"
    )
    
    # Display model accuracy
    accuracy = MODEL_ACCURACIES[algo]
    st.metric(
        label="Model Accuracy",
        value=f"{accuracy*100:.1f}%",
        help="Test accuracy on validation dataset"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Algorithm descriptions
with st.expander("ℹ️ Algorithm Information"):
    algo_info = {
        "SVM (MFCC)": "Support Vector Machine using Mel-Frequency Cepstral Coefficients - captures spectral characteristics of speech",
        "CNN (Mel-spectrogram)": "Convolutional Neural Network analyzing mel-spectrograms - uses deep learning to detect visual patterns in audio",
        "CQCC + GMM (LLR)": "Gaussian Mixture Models with Constant-Q Cepstral Coefficients - compares likelihood ratios between real/fake distributions"
    }
    st.info(algo_info[algo])

# Analysis button (centered)
if uploaded_file is not None:
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Audio", type="primary", use_container_width=True)
else:
    analyze_button = False

# -----------------------------
# Results and Visualizations - IMPROVED LAYOUT
# -----------------------------
if uploaded_file is not None and analyze_button:
    with st.spinner("🔄 Processing audio and running analysis..."):
        
        # Run the selected algorithm
        if algo == "SVM (MFCC)":
            features, y, sr, mfcc = extract_mfcc(uploaded_file)
            svm_model = load_svm()
            pred = svm_model.predict(features)[0]
            proba = svm_model.predict_proba(features)[0]
            prob_real, prob_fake = float(proba[0]), float(proba[1])
            label = "🟢 REAL" if pred == 0 else "🔴 FAKE"
            result_class = "result-positive" if pred == 0 else "result-negative"

        elif algo == "CNN (Mel-spectrogram)":
            mel_norm, y, sr, mel_db_for_plot = extract_melspectrogram(uploaded_file)
            cnn_model = load_cnn()
            X = mel_norm[np.newaxis, ..., np.newaxis]
            pred_prob_fake = float(cnn_model.predict(X, verbose=0)[0][0])
            prob_fake = pred_prob_fake
            prob_real = 1 - prob_fake
            label = "🟢 REAL" if prob_real > prob_fake else "🔴 FAKE"
            result_class = "result-positive" if prob_real > prob_fake else "result-negative"

        else:  # CQCC + GMM (LLR)
            cqcc_frames, y, sr, cqt_db_for_plot, cqcc_for_plot = extract_cqcc(uploaded_file)
            gmm_real, gmm_fake = load_gmm()

            ll_real = float(gmm_real.score(cqcc_frames))
            ll_fake = float(gmm_fake.score(cqcc_frames))

            ll_real_frames = gmm_real.score_samples(cqcc_frames)
            ll_fake_frames = gmm_fake.score_samples(cqcc_frames)
            llr_frames = ll_real_frames - ll_fake_frames

            pred = 0 if ll_real > ll_fake else 1
            label = "🟢 REAL" if pred == 0 else "🔴 FAKE"
            result_class = "result-positive" if pred == 0 else "result-negative"

            exp_real, exp_fake = np.exp([ll_real, ll_fake])
            prob_real = float(exp_real / (exp_real + exp_fake))
            prob_fake = float(exp_fake / (exp_real + exp_fake))

    # Results section with better layout
    st.markdown("---")
    st.subheader("🎯 Detection Results")
    
    # Main result display - smaller size
    st.markdown(f'<div class="{result_class}">{label}</div>', unsafe_allow_html=True)
    
    # Probability metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Real Probability",
            value=f"{prob_real:.1%}",
            delta=f"{prob_real-0.5:.1%}" if prob_real > 0.5 else None
        )
    
    with col2:
        st.metric(
            label="Fake Probability", 
            value=f"{prob_fake:.1%}",
            delta=f"{prob_fake-0.5:.1%}" if prob_fake > 0.5 else None
        )
    
    with col3:
        confidence = max(prob_real, prob_fake)
        st.metric(
            label="Confidence",
            value=f"{confidence:.1%}"
        )

    # Progress bar for confidence
    st.progress(confidence)

    # Visualizations section - waveform now as a tab
    st.subheader("📊 Audio Analysis Visualizations")
    
    # Create tabs for different visualizations
    if algo == "SVM (MFCC)":
        tab1, tab2, tab3 = st.tabs(["📈 Waveform", "🔥 MFCC Features", "📊 Probabilities"])
        
        with tab1:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Audio Waveform")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            img = librosa.display.specshow(mfcc, x_axis="time", ax=ax, sr=sr)
            fig.colorbar(img, ax=ax)
            ax.set_title("Mel-Frequency Cepstral Coefficients (MFCCs)")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(["Real", "Fake"], [prob_real, prob_fake], 
                         color=['#4CAF50', '#F44336'])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Detection Confidence")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2%}', ha='center', va='bottom')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    elif algo == "CNN (Mel-spectrogram)":
        tab1, tab2, tab3 = st.tabs(["📈 Waveform", "🌈 Mel-Spectrogram", "📊 Probabilities"])
        
        with tab1:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude") 
            ax.set_title("Audio Waveform")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            img = librosa.display.specshow(mel_db_for_plot, sr=sr, x_axis="time", 
                                         y_axis="mel", ax=ax)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title("Mel-Spectrogram (Input to CNN)")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(["Real", "Fake"], [prob_real, prob_fake],
                         color=['#4CAF50', '#F44336'])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Detection Confidence")
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2%}', ha='center', va='bottom')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    else:  # CQCC + GMM
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Waveform", "🎵 CQT Spectrum", "🔧 CQCC Features", 
            "📈 LLR Timeline", "📊 Probabilities"
        ])
        
        with tab1:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Audio Waveform")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            img = librosa.display.specshow(cqt_db_for_plot, x_axis="time", 
                                         y_axis="cqt_note", ax=ax)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title("Constant-Q Transform Log-Power Spectrogram")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            img = librosa.display.specshow(cqcc_for_plot, x_axis="time", ax=ax)
            fig.colorbar(img, ax=ax)
            ax.set_ylabel("CQCC Coefficient Index")
            ax.set_title("Constant-Q Cepstral Coefficients (Normalized)")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(llr_frames, linewidth=2, color='#2196F3')
            ax.axhline(0.0, linestyle="--", color='red', alpha=0.7, label="Decision Boundary")
            ax.fill_between(range(len(llr_frames)), llr_frames, 0, 
                           where=(llr_frames > 0), alpha=0.3, color='green', label='Real')
            ax.fill_between(range(len(llr_frames)), llr_frames, 0,
                           where=(llr_frames < 0), alpha=0.3, color='red', label='Fake')
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Log-Likelihood Ratio")
            ax.set_title("Per-frame LLR Analysis (Real - Fake)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<div class="compact-plot">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(["Real", "Fake"], [prob_real, prob_fake],
                         color=['#4CAF50', '#F44336'])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Detection Confidence")
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2%}', ha='center', va='bottom')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer with model performance
# -----------------------------
st.markdown("---")
st.subheader("📈 Model Performance Comparison")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.metric(
        label="SVM (MFCC)",
        value=f"{MODEL_ACCURACIES['SVM (MFCC)']*100:.1f}%",
        help="Accuracy on test dataset"
    )

with perf_col2:
    st.metric(
        label="CNN (Mel-spectrogram)", 
        value=f"{MODEL_ACCURACIES['CNN (Mel-spectrogram)']*100:.1f}%",
        help="Accuracy on test dataset"
    )

with perf_col3:
    st.metric(
        label="CQCC + GMM (LLR)",
        value=f"{MODEL_ACCURACIES['CQCC + GMM (LLR)']*100:.1f}%", 
        help="Accuracy on test dataset"
    )

# Additional info in sidebar
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown("""
    This application uses three different machine learning approaches to detect deepfake audio:
    
    **🔸 SVM with MFCCs**: Traditional machine learning approach using spectral features
    
    **🔸 CNN with Mel-spectrograms**: Deep learning model that analyzes visual patterns in audio spectrograms
    
    **🔸 GMM with CQCCs**: Statistical model comparing likelihood distributions of real vs fake audio
    """)
    
    st.header("📋 Usage Tips")
    st.markdown("""
    - Upload clear audio files (WAV/MP3 format)
    - Longer audio samples may provide better accuracy
    - Different algorithms may perform better on different types of deepfakes
    - Check multiple algorithms for consensus
    """)
    
    st.header("⚠️ Disclaimer")
    st.markdown("""
    This tool is for research and educational purposes. Detection accuracy may vary based on:
    - Audio quality and length
    - Type of deepfake generation method
    - Background noise and compression
    """)

    st.header("📸 Layout Tips")
    st.markdown("""
    **Perfect for Screenshots:**
    - Result box and metrics at top
    - Choose any visualization tab below
    - Waveform now included as first tab
    - All graphs are compact and centered
    """)
