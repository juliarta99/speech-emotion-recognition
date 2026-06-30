"""
==========================================================================
 STREAMLIT APP - SPEECH EMOTION RECOGNITION (CREMA-D)
==========================================================================
Aplikasi ini memungkinkan pengguna mengunggah file audio (.wav), lalu
secara otomatis:
  1. Melakukan PREPROCESSING (trim silence, normalisasi amplitudo,
     penyeragaman durasi) -- identik dengan preprocessing yang dipakai
     saat training di notebook.
  2. Mengekstrak FITUR (domain waktu, domain frekuensi, MFCC+Delta+Delta2)
     -- identik kolom & urutannya dengan yang dipakai saat training.
  3. Melakukan KLASIFIKASI EMOSI menggunakan model yang sudah dilatih
     (SVM, Random Forest, dan/atau CNN) dari 'model_bundle.pkl'.

Cara menjalankan:
    streamlit run app.py

Pastikan file-file berikut berada di FOLDER YANG SAMA dengan app.py:
    - model_bundle.pkl       (WAJIB, hasil ekspor dari notebook bagian 7)
    - cnn_model.keras        (WAJIB jika ingin opsi model CNN aktif)
    - cnn_label_encoder.pkl  (WAJIB jika ingin opsi model CNN aktif,
                               pasangan dari cnn_model.keras)

Jika opsi CNN tidak muncul di aplikasi, cek bagian sidebar -- akan
dijelaskan secara spesifik file mana yang belum ditemukan/gagal dimuat.
==========================================================================
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================================================
# KONFIGURASI HALAMAN
# ==========================================================================
st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="centered",
)

MODEL_BUNDLE_PATH = "model_bundle.pkl"
CNN_MODEL_PATH = "cnn_model.keras"
CNN_LABEL_ENCODER_PATH = "cnn_label_encoder.pkl"

EMOTION_LABELS_ID = {
    "ANG": "Marah (Angry)",
    "DIS": "Jijik (Disgust)",
    "FEA": "Takut (Fear)",
    "HAP": "Senang (Happy)",
    "NEU": "Netral (Neutral)",
    "SAD": "Sedih (Sad)",
}


# ==========================================================================
# FUNGSI: load_model_bundle / load_cnn_model
# ----------------------------------------------------------------------
# Di-cache (st.cache_resource) agar model tidak dimuat ulang dari disk
# setiap kali ada interaksi pengguna -- hanya dimuat sekali per sesi.
# ==========================================================================
@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_BUNDLE_PATH):
        return None
    with open(MODEL_BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


@st.cache_resource
def load_cnn_model():
    """Memuat model CNN + label encoder-nya. Mengembalikan tiga nilai:
    (model, label_encoder, pesan_error). Jika berhasil, pesan_error=None.
    Jika gagal, model & label_encoder=None dan pesan_error berisi alasan
    SPESIFIK kegagalannya (bukan cuma 'tidak ditemukan' secara umum)."""
    missing = []
    if not os.path.exists(CNN_MODEL_PATH):
        missing.append(f"'{CNN_MODEL_PATH}'")
    if not os.path.exists(CNN_LABEL_ENCODER_PATH):
        missing.append(f"'{CNN_LABEL_ENCODER_PATH}'")

    if missing:
        return None, None, (
            f"File berikut tidak ditemukan di folder app.py: {', '.join(missing)}. "
            f"Pastikan kalian sudah MENYALIN kedua file hasil ekspor notebook "
            f"(bagian 'Simpan CNN secara terpisah') ke folder yang sama dengan "
            f"app.py ini, bukan hanya 'model_bundle.pkl' saja."
        )

    try:
        import tensorflow as tf
    except ImportError:
        return None, None, (
            "TensorFlow belum terpasang di environment Python yang menjalankan "
            "Streamlit ini. Jalankan: pip install tensorflow"
        )

    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        with open(CNN_LABEL_ENCODER_PATH, "rb") as f:
            cnn_label_encoder = pickle.load(f)
        return cnn_model, cnn_label_encoder, None
    except Exception as e:
        return None, None, f"Gagal memuat model CNN: {e}"


# ==========================================================================
# FUNGSI PREPROCESSING & EKSTRAKSI FITUR
# ----------------------------------------------------------------------
# Identik dengan fungsi di notebook training, agar fitur yang dihasilkan
# konsisten dengan yang dipelajari model. Parameter (SR, FRAME_LENGTH,
# HOP_LENGTH, dst) diambil dari config yang disimpan di dalam model_bundle.
# ==========================================================================
def load_and_preprocess_audio(filepath, sr, top_db, duration):
    y, sr = librosa.load(filepath, sr=sr)

    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trimmed) > 0:
        y = y_trimmed

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    return y, sr


def extract_time_domain_features(y, frame_length, hop_length):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    return {
        "rms_mean": np.mean(rms), "rms_std": np.std(rms),
        "zcr_mean": np.mean(zcr), "zcr_std": np.std(zcr),
    }


def extract_frequency_domain_features(y, sr, frame_length, hop_length):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    return {
        "centroid_mean": np.mean(centroid), "centroid_std": np.std(centroid),
        "bandwidth_mean": np.mean(bandwidth), "bandwidth_std": np.std(bandwidth),
        "rolloff_mean": np.mean(rolloff), "rolloff_std": np.std(rolloff),
    }


def extract_mfcc_features(y, sr, n_mfcc, frame_length, hop_length):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    feats = {}
    for i in range(n_mfcc):
        feats[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        feats[f"mfcc_{i+1}_std"] = np.std(mfcc[i])
        feats[f"delta_mfcc_{i+1}_mean"] = np.mean(delta_mfcc[i])
        feats[f"delta_mfcc_{i+1}_std"] = np.std(delta_mfcc[i])
        feats[f"delta2_mfcc_{i+1}_mean"] = np.mean(delta2_mfcc[i])
        feats[f"delta2_mfcc_{i+1}_std"] = np.std(delta2_mfcc[i])
    return feats


def extract_mel_spectrogram(y, sr, n_mels, max_pad_len, frame_length, hop_length):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=frame_length, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :max_pad_len]
    return mel_db


def extract_all_features(filepath, config):
    """Menjalankan seluruh pipeline preprocessing + ekstraksi fitur untuk
    satu file audio, mengembalikan: dict fitur tabular, mel spectrogram,
    dan sinyal audio bersih (untuk visualisasi)."""
    sr = config["SR"]
    top_db = config.get("TOP_DB", 25)
    duration = config.get("FIXED_DURATION", 4.0)
    frame_length = config.get("FRAME_LENGTH", 2048)
    hop_length = config.get("HOP_LENGTH", 512)
    n_mfcc = config.get("N_MFCC", 13)
    n_mels = config.get("N_MELS", 128)
    max_pad_len = config.get("MAX_PAD_LEN", 174)

    y, sr = load_and_preprocess_audio(filepath, sr, top_db, duration)

    feats = {}
    feats.update(extract_time_domain_features(y, frame_length, hop_length))
    feats.update(extract_frequency_domain_features(y, sr, frame_length, hop_length))
    feats.update(extract_mfcc_features(y, sr, n_mfcc, frame_length, hop_length))

    mel_db = extract_mel_spectrogram(y, sr, n_mels, max_pad_len, frame_length, hop_length)

    return feats, mel_db, y, sr


# ==========================================================================
# FUNGSI PREDIKSI
# ----------------------------------------------------------------------
# Menyusun fitur sesuai urutan kolom yang dipakai saat training, lalu
# menjalankan scaler + model untuk menghasilkan prediksi + probabilitas.
# ==========================================================================
def predict_emotion(feats_dict, bundle, model_choice):
    feature_columns = bundle["feature_columns"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]

    X = np.array([[feats_dict[col] for col in feature_columns]])
    X_scaled = scaler.transform(X)

    model = bundle["svm_model"] if model_choice == "SVM" else bundle["rf_model"]
    pred_encoded = model.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba_values = model.predict_proba(X_scaled)[0]
        proba = dict(zip(label_encoder.classes_, proba_values))

    return pred_label, proba


def predict_emotion_cnn(mel_db, cnn_model, cnn_label_encoder):
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    mel_input = mel_norm[np.newaxis, ..., np.newaxis]
    proba_values = cnn_model.predict(mel_input, verbose=0)[0]
    pred_label = cnn_label_encoder.inverse_transform([np.argmax(proba_values)])[0]
    proba = dict(zip(cnn_label_encoder.classes_, proba_values))
    return pred_label, proba


def plot_waveform_and_melspec(y, sr, mel_db):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5))

    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="steelblue")
    axes[0].set_title("Waveform (setelah preprocessing)")
    axes[0].set_xlabel("Waktu (detik)")

    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1])
    axes[1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    return fig


# ==========================================================================
# UI UTAMA
# ==========================================================================
st.title("Speech Emotion Recognition")
st.caption("Klasifikasi emosi dari suara berdasarkan dataset CREMA-D "
           "(ANG, DIS, FEA, HAP, NEU, SAD)")

bundle = load_model_bundle()

if bundle is None:
    st.error(
        f"File model `{MODEL_BUNDLE_PATH}` tidak ditemukan di folder ini. "
        "Pastikan kalian sudah menyalin file hasil ekspor dari notebook "
        "(`model_bundle.pkl`) ke folder yang sama dengan `app.py` ini."
    )
    st.stop()

cnn_model, cnn_label_encoder, cnn_error_msg = load_cnn_model()

with st.sidebar:
    st.header("Pengaturan")
    model_options = ["SVM", "Random Forest"]
    if cnn_model is not None:
        model_options.append("CNN (Mel Spectrogram)")
    model_choice_display = st.radio("Pilih model klasifikasi:", model_options, index=0)

    st.markdown("---")
    st.markdown(
        "**Tentang model:**\n"
        f"- Skenario fitur: `{bundle.get('scenario', 'C_Kombinasi')}`\n"
        f"- Jumlah fitur tabular: `{len(bundle['feature_columns'])}`\n"
        f"- Kelas emosi: `{', '.join(bundle['label_encoder'].classes_)}`"
    )

    # if cnn_model is None:
    #     st.warning("Model CNN belum aktif.")
    #     with st.expander("Lihat alasan & cara mengaktifkan CNN"):
    #         st.write(cnn_error_msg)
    #         st.markdown(
    #             "**Langkah mengaktifkan CNN:**\n"
    #             "1. Buka kembali notebook training kalian.\n"
    #             "2. Pastikan sel CNN (bagian 6) sudah pernah dijalankan sampai selesai.\n"
    #             "3. Pastikan sel **'Simpan CNN secara terpisah'** (bagian 7) sudah "
    #             "dijalankan -- sel ini menghasilkan `cnn_model.keras` dan "
    #             "`cnn_label_encoder.pkl`.\n"
    #             "4. Salin (copy) KEDUA file tersebut dari folder kerja notebook ke "
    #             "folder yang sama dengan `app.py` ini.\n"
    #             "5. Restart aplikasi Streamlit (`streamlit run app.py` lagi)."
    #         )
    # else:
    #     st.success("Model CNN aktif dan siap digunakan.")

uploaded_file = st.file_uploader(
    "Unggah file audio (.wav)", type=["wav"],
    help="File akan diproses otomatis: trim silence, normalisasi amplitudo, "
         "lalu diekstrak fiturnya sebelum diklasifikasi."
)

if uploaded_file is not None:
    # Simpan file upload ke file sementara, karena librosa.load butuh path
    # atau file-like object yang seekable -- cara paling aman lewat tempfile.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Memproses audio (preprocessing + ekstraksi fitur)..."):
        try:
            feats_dict, mel_db, y_clean, sr_clean = extract_all_features(
                tmp_path, bundle["config"]
            )
        except Exception as e:
            st.error(f"Gagal memproses file audio: {e}")
            st.stop()
        finally:
            os.unlink(tmp_path)

    st.success("Preprocessing & ekstraksi fitur selesai.")

    # ----------------------------------------------------------------
    # VISUALISASI HASIL PREPROCESSING
    # ----------------------------------------------------------------
    with st.expander("Lihat visualisasi waveform & Mel Spectrogram"):
        fig = plot_waveform_and_melspec(y_clean, sr_clean, mel_db)
        st.pyplot(fig)

    # ----------------------------------------------------------------
    # KLASIFIKASI
    # ----------------------------------------------------------------
    st.subheader("Hasil Klasifikasi Emosi")

    if model_choice_display == "CNN (Mel Spectrogram)":
        pred_label, proba = predict_emotion_cnn(mel_db, cnn_model, cnn_label_encoder)
    else:
        model_key = "SVM" if model_choice_display == "SVM" else "RandomForest"
        pred_label, proba = predict_emotion(feats_dict, bundle, model_key)

    pred_display = EMOTION_LABELS_ID.get(pred_label, pred_label)
    st.markdown(f"## {pred_display}")
    st.caption(f"Model yang digunakan: **{model_choice_display}**")

    if proba is not None:
        st.markdown("#### Probabilitas per kelas emosi")
        df_proba = pd.DataFrame({
            "Emosi": [EMOTION_LABELS_ID.get(k, k) for k in proba.keys()],
            "Probabilitas": list(proba.values()),
        }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

        st.bar_chart(df_proba.set_index("Emosi")["Probabilitas"])
        st.dataframe(
            df_proba.style.format({"Probabilitas": "{:.2%}"}),
            use_container_width=True,
        )

    with st.expander("Lihat detail nilai fitur yang diekstrak"):
        df_feats = pd.DataFrame(
            list(feats_dict.items()), columns=["Fitur", "Nilai"]
        )
        st.dataframe(df_feats, use_container_width=True, height=300)

else:
    st.info("Unggah file audio (.wav) untuk mulai klasifikasi emosi.")
    st.markdown(
        """
        **Catatan:**
        - File audio idealnya berisi ucapan manusia (bukan musik/noise).
        - Durasi file akan otomatis diseragamkan, jadi tidak perlu memotong
          manual sebelum diunggah.
        - Hasil terbaik didapat dari rekaman yang relatif bersih (minim
          noise latar belakang).
        """
    )