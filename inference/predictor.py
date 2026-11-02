"""Load the trained models once and run predictions with them.

A model whose files are missing or fail to load is kept in the registry with
an error message instead of crashing the app, so the website can explain what
is missing.
"""
import json
import logging
import threading

import joblib
import numpy as np

from config import LABELS_DIR, MODELS, MODELS_DIR
from preprocessing import features

log = logging.getLogger(__name__)


class ModelUnavailable(Exception):
    pass


class LoadedModel:
    def __init__(self, key, spec):
        self.key = key
        self.spec = spec
        self.model = None
        self.scaler = None
        self.labels = None
        self.timesteps = None
        self.error = None
        self._lock = threading.Lock()

    @property
    def ready(self):
        return self.model is not None

    def load(self):
        spec = self.spec
        model_path = MODELS_DIR / spec["model_file"]
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: models/{spec['model_file']}")
            if spec["scaler_file"]:
                scaler_path = MODELS_DIR / spec["scaler_file"]
                if not scaler_path.exists():
                    raise FileNotFoundError(f"Scaler file not found: models/{spec['scaler_file']}")
                self.scaler = joblib.load(scaler_path)
            if spec["labels"]:
                with open(LABELS_DIR / spec["labels"]) as f:
                    self.labels = json.load(f)

            if spec["kind"] == "keras":
                from tensorflow.keras.models import load_model

                model = load_model(model_path)
                # (None, time_steps, n_mfcc): the padding length used in training
                _, self.timesteps, n_feat = model.input_shape
                if n_feat != 40:
                    raise ValueError(f"Model expects {n_feat} features per frame, extractor gives 40")
                if self.labels is not None and model.output_shape[-1] != len(self.labels):
                    raise ValueError(f"Model has {model.output_shape[-1]} outputs but label file has {len(self.labels)}")
            else:
                model = joblib.load(model_path)
                # classes_ are label indices; a class missing from the training split is allowed
                if self.labels is not None and int(max(model.classes_)) >= len(self.labels):
                    raise ValueError(f"Model predicts class {max(model.classes_)} but label file has {len(self.labels)}")
            self.model = model
            log.info("Loaded model %s from %s", self.key, model_path)
        except Exception as e:  # keep the app running, report on the page
            self.model = None
            self.error = str(e)
            log.warning("Model %s unavailable: %s", self.key, e)
        return self

    def _label(self, class_value):
        if self.labels is None:
            return str(class_value)
        return self.labels[int(class_value)]

    def predict(self, audio_path, top_k=5):
        if not self.ready:
            raise ModelUnavailable(self.error or "Model not loaded")
        extract = getattr(features, self.spec["extractor"])

        with self._lock:
            if self.spec["kind"] == "keras":
                seq = extract(audio_path)
                x = features.pad_mfcc_sequence(seq, self.timesteps)
                probs = self.model.predict(x, verbose=0)[0]
                classes = np.arange(len(probs))
            else:
                x = extract(audio_path)
                if self.scaler is not None:
                    x = self.scaler.transform(x)
                pred = self.model.predict(x)[0]
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(x)[0]
                    classes = self.model.classes_
                else:
                    # SVC trained without probability=True: label only
                    return {"digit": self._label(pred), "confidence": None, "top": []}

        order = np.argsort(probs)[::-1][:top_k]
        top = [{"digit": self._label(classes[i]), "probability": float(probs[i])} for i in order]
        return {"digit": top[0]["digit"], "confidence": top[0]["probability"], "top": top}

    def info(self):
        s = self.spec
        return {
            "key": self.key,
            "name": s["name"],
            "notebook": s["notebook"],
            "features": s["features"],
            "accuracy": s["accuracy"],
            "gives_probabilities": s["kind"] == "keras"
            or (self.ready and hasattr(self.model, "predict_proba")),
            "ready": self.ready,
            "error": self.error,
        }


def load_all():
    return {key: LoadedModel(key, spec).load() for key, spec in MODELS.items()}
