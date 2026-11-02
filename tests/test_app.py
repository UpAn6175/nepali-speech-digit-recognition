"""Checks that the web app predicts exactly what the notebook code predicts.

Runs against placeholder models (tools/make_test_models.py), or against the
real models if DIGIT_MODELS_DIR points at them and TEST_WAV_DIR holds WAVs.

    python tools/make_test_models.py /tmp/digit_test_models
    DIGIT_MODELS_DIR=/tmp/digit_test_models python -m pytest tests -q
"""
import io
import os
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS_DIR = Path(os.environ.get("DIGIT_MODELS_DIR", ROOT / "models"))
if not (MODELS_DIR / "NEPALI_LSTM_model.keras").exists():
    pytest.skip("No models available; see module docstring", allow_module_level=True)

from app import app  # noqa: E402  (loads models at import, like production)

WAV_DIR = Path(os.environ.get("TEST_WAV_DIR", MODELS_DIR / "wavs"))
WAVS = sorted(WAV_DIR.glob("*.wav"))[:6]
MODEL_FILES = {"nepali_svm": "NEPALI_SVM_model.joblib", "nepali_rf": "NEP_RF_model.joblib"}
NEPALI_SORTED = sorted(str(i) for i in range(1, 101))


# ---- Reference code copied from the notebooks' prediction cells -------------
def nb_lstm(file_path):  # LSTM(1).ipynb cells 21-22
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    label_names = [str(i) for i in range(1, 101)]
    model = load_model(MODELS_DIR / "NEPALI_LSTM_model.keras")
    y, sr = librosa.load(file_path, duration=1.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    padded_seq = pad_sequences([mfcc], padding="post", dtype="float32")
    prediction = model.predict(padded_seq, verbose=0)
    return label_names[np.argmax(prediction)], float(np.max(prediction))


def nb_svm(file_path):  # SVM(1).ipynb cell 12
    model = joblib.load(MODELS_DIR / "NEPALI_SVM_model.joblib")
    scaler = joblib.load(MODELS_DIR / "NEPALI_SVM_scaler.joblib")
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feats = np.mean(mfcc.T, axis=0).reshape(1, -1)
    return NEPALI_SORTED[model.predict(scaler.transform(feats))[0]]


def nb_rf(file_path):  # RandomForest(3).ipynb cell 9
    model = joblib.load(MODELS_DIR / "NEP_RF_model.joblib")
    scaler = joblib.load(MODELS_DIR / "NEP_RF_scaler.joblib")
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=42)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T)
    feats = np.concatenate([mfcc_mean, [zcr, centroid]]).reshape(1, -1)
    return NEPALI_SORTED[model.predict(scaler.transform(feats))[0]]


@pytest.fixture
def client():
    return app.test_client()


def post(client, path, model):
    with open(path, "rb") as f:
        return client.post("/api/predict", data={"audio": (f, path.name), "model": model})


@pytest.mark.parametrize("wav", WAVS, ids=lambda p: p.name)
def test_matches_notebook_sklearn(client, wav):
    for key, ref in [("nepali_svm", nb_svm), ("nepali_rf", nb_rf)]:
        if not (MODELS_DIR / MODEL_FILES[key]).exists():
            continue
        r = post(client, wav, key)
        assert r.status_code == 200, r.json
        assert r.json["digit"] == ref(wav), key


@pytest.mark.parametrize("wav", WAVS, ids=lambda p: p.name)
def test_matches_notebook_lstm(client, wav):
    if librosa.get_duration(path=wav) < 1.0:
        pytest.skip("notebook code pads short clips to their own length (fixed in app)")
    r = post(client, wav, "nepali_lstm")
    assert r.status_code == 200, r.json
    digit, conf = nb_lstm(wav)
    assert r.json["digit"] == digit
    assert r.json["confidence"] == pytest.approx(conf, rel=1e-4)
    assert len(r.json["top"]) == 5


def test_lstm_pads_short_clip_to_training_length(tmp_path):
    from preprocessing import features

    p = tmp_path / "short.wav"
    sf.write(p, 0.3 * np.sin(np.linspace(0, 900, 22050 // 2)), 22050)
    x = features.pad_mfcc_sequence(features.extract_mfcc_sequence(p), 44)
    assert x.shape == (1, 44, 40)
    assert np.all(x[0, 30:] == 0)  # post-padding with zeros, as in training


def test_rejects_wrong_extension(client):
    r = client.post("/api/predict", data={"audio": (io.BytesIO(b"hello"), "notes.txt")})
    assert r.status_code == 400 and "Unsupported" in r.json["error"]


def test_rejects_corrupt_wav(client):
    r = client.post("/api/predict", data={"audio": (io.BytesIO(b"not really audio"), "x.wav")})
    assert r.status_code == 400 and "could not be read" in r.json["error"]


def test_rejects_silence(client, tmp_path):
    p = tmp_path / "silent.wav"
    sf.write(p, np.zeros(22050), 22050)
    r = post(client, p, "nepali_lstm")
    assert r.status_code == 400 and "silent" in r.json["error"]


def test_missing_file(client):
    r = client.post("/api/predict", data={})
    assert r.status_code == 400


def test_models_endpoint(client):
    r = client.get("/api/models")
    assert r.json["default"] == "nepali_lstm"
    assert {m["key"] for m in r.json["models"]} == {"nepali_lstm", "nepali_svm", "nepali_rf"}


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Predict" in r.data
