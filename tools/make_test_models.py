"""Build PLACEHOLDER models on synthetic tones, for testing the web app only.

These models are NOT the real trained models and predict nothing meaningful
on speech. They exist so the app and its tests can run before the real
.keras / .joblib files from the notebooks are copied into models/.
They use the same feature extractors and the same model types/hyperparameters
as the notebooks, but are trained on fake data with few epochs.

Usage:  python tools/make_test_models.py OUT_DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from preprocessing import features

def tone(freq, seconds, sr, rng):
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(len(t))).astype("float32")


def write_wavs(root, n_classes, per_class, sr, rng):
    paths = []
    for c in range(n_classes):
        for r in range(per_class):
            p = root / f"c{c}_r{r}.wav"
            sf.write(p, tone(150 + 25 * c, rng.uniform(0.5, 1.2), sr, rng), sr)
            paths.append((p, c))
    return paths


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wav_dir = out / "wavs"
    wav_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    data = write_wavs(wav_dir, 100, 2, 22050, rng)
    y = np.array([c for _, c in data])

    # SVM (Nepali): same SVC settings as the saved notebook model
    X = np.vstack([features.extract_mfcc_mean_native_sr(p) for p, _ in data])
    scaler = StandardScaler().fit(X)
    joblib.dump(svm.SVC(kernel="rbf", C=10, gamma=0.1).fit(scaler.transform(X), y), out / "NEPALI_SVM_model.joblib")
    joblib.dump(scaler, out / "NEPALI_SVM_scaler.joblib")

    # Random Forest (Nepali)
    X = np.vstack([features.extract_mfcc_zcr_centroid(p) for p, _ in data])
    scaler = StandardScaler().fit(X)
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=3,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    joblib.dump(rf.fit(scaler.transform(X), y), out / "NEP_RF_model.joblib")
    joblib.dump(scaler, out / "NEP_RF_scaler.joblib")

    # LSTM (Nepali): same architecture as LSTM(1).ipynb cell 17, 2 epochs only
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical

    seqs = [features.extract_mfcc_sequence(p) for p, _ in data]
    Xs = pad_sequences(seqs, padding="post", dtype="float32")
    model = Sequential()
    model.add(LSTM(64, input_shape=(Xs.shape[1], Xs.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(Xs, to_categorical(y, 100), epochs=2, batch_size=32, verbose=0)
    model.save(out / "NEPALI_LSTM_model.keras")
    print(f"Placeholder models written to {out} (LSTM input shape {model.input_shape})")


if __name__ == "__main__":
    main(sys.argv[1])
