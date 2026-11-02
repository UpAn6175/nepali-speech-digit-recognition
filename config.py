"""App settings and the model registry.

To switch the default model, set DIGIT_MODEL to one of the keys in MODELS
(for example `DIGIT_MODEL=nepali_svm`). To add a model, add an entry here and
put its files in models/.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("DIGIT_MODELS_DIR", BASE_DIR / "models"))
LABELS_DIR = BASE_DIR / "models" / "labels"

DEFAULT_MODEL = os.environ.get("DIGIT_MODEL", "nepali_lstm")

ALLOWED_EXTENSIONS = {"wav", "flac", "ogg", "mp3"}
MAX_UPLOAD_MB = 10
MAX_AUDIO_SECONDS = 30

# Each entry describes one saved model from the notebooks.
#   kind       "keras" (softmax probabilities) or "sklearn"
#   extractor  function name in preprocessing/features.py
#   labels     JSON list mapping class index -> spoken number
#   accuracy   held-out accuracy printed in the notebook
MODELS = {
    "nepali_lstm": {
        "name": "LSTM (Nepali, 1-100)",
        "notebook": "TrainingCodes/LSTM(1).ipynb",
        "kind": "keras",
        "model_file": "NEPALI_LSTM_model.keras",
        "scaler_file": None,
        "labels": "nepali_lstm_labels.json",
        "extractor": "extract_mfcc_sequence",
        "features": "40 MFCCs per frame, first 1 s at 22,050 Hz, zero-padded to 44 frames",
        "accuracy": 0.6861,
    },
    "nepali_svm": {
        "name": "SVM (Nepali, 1-100)",
        "notebook": "TrainingCodes/SVM(1).ipynb",
        "kind": "sklearn",
        "model_file": "NEPALI_SVM_model.joblib",
        "scaler_file": "NEPALI_SVM_scaler.joblib",
        "labels": "nepali_sorted_folder_labels.json",
        "extractor": "extract_mfcc_mean_native_sr",
        "features": "Mean of 40 MFCCs, native sample rate, StandardScaler",
        "accuracy": 0.6639,
    },
    "nepali_rf": {
        "name": "Random Forest (Nepali, 1-100)",
        "notebook": "TrainingCodes/RandomForest(3).ipynb",
        "kind": "sklearn",
        "model_file": "NEP_RF_model.joblib",
        "scaler_file": "NEP_RF_scaler.joblib",
        "labels": "nepali_sorted_folder_labels.json",
        "extractor": "extract_mfcc_zcr_centroid",
        "features": "Mean of 42 MFCCs + ZCR + spectral centroid at 22,050 Hz, StandardScaler",
        "accuracy": 0.4917,
    },
}

# train_models.py records the test accuracy of a fresh training run here;
# it replaces the notebook figure so the page shows the accuracy of the files in use.
_metrics = MODELS_DIR / "metrics.json"
if _metrics.exists():
    import json

    for _key, _acc in json.loads(_metrics.read_text()).items():
        if _key in MODELS:
            MODELS[_key]["accuracy"] = _acc
