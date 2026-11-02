"""Feature extraction, copied from the training notebooks.

Each function below reproduces the extraction code of one notebook line for
line (sample rate, duration, MFCC count and librosa defaults for n_fft=2048,
hop_length=512, win_length=2048). Do not change parameters here without
retraining the matching model: the saved model only understands features
produced exactly this way.
"""
import numpy as np
import librosa


# --- Nepali LSTM: TrainingCodes/LSTM(1).ipynb, cells 14 and 22 ------------------
def extract_mfcc_sequence(file_path, n_mfcc=40, duration=1.0):
    """(frames, 40) MFCC sequence of the first second at librosa's default 22,050 Hz."""
    y, sr = librosa.load(file_path, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    return mfcc


def pad_mfcc_sequence(mfcc_seq, maxlen):
    """Pad to the model's time-step count exactly as training did.

    Training called pad_sequences(padding='post', dtype='float32') over the whole
    dataset, which zero-padded every clip at the end up to the longest clip
    (44 frames). The notebook's predict_digit() padded a single file only to its
    own length, so clips shorter than 1 s produced fewer time steps than the
    model was trained on. Passing maxlen fixes that and nothing else.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    return pad_sequences([mfcc_seq], maxlen=maxlen, padding="post", dtype="float32")


# --- Nepali SVM: TrainingCodes/SVM(1).ipynb, cells 1 and 12 --------------------
def extract_mfcc_mean_native_sr(file_path, n_mfcc=40):
    """40-dim MFCC mean, loaded at the file's native sample rate (sr=None)."""
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)


# --- Nepali Random Forest: TrainingCodes/RandomForest(3).ipynb, cells 2 and 9 --
def extract_mfcc_zcr_centroid(file_path):
    """42 MFCC means + mean zero-crossing rate + mean spectral centroid (44 dims)."""
    y, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=42)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T)

    features = np.concatenate([mfcc_mean, [zcr, centroid]])
    return features.reshape(1, -1)

