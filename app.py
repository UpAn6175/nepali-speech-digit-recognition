"""Flask web app for the speech digit recognizer.

Run with:  python app.py   then open http://127.0.0.1:5000
"""
import logging
import os
import tempfile

import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

import config
from inference.predictor import ModelUnavailable, load_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

# Loaded once at startup, never retrained per request.
MODELS = load_all()


class AudioError(ValueError):
    pass


def default_model_key():
    if config.DEFAULT_MODEL in MODELS and MODELS[config.DEFAULT_MODEL].ready:
        return config.DEFAULT_MODEL
    ready = [k for k, m in MODELS.items() if m.ready]
    return ready[0] if ready else config.DEFAULT_MODEL


def check_audio(path):
    """Reject files librosa cannot decode, or that are empty, silent or too long."""
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
    except Exception:
        raise AudioError("This file could not be read as audio. Please upload a valid WAV file.")
    duration = len(y) / sr if sr else 0
    if duration < 0.1:
        raise AudioError("The audio is too short. Record at least a fraction of a second of speech.")
    if duration > config.MAX_AUDIO_SECONDS:
        raise AudioError(
            f"The audio is {duration:.1f} s long. Please upload a clip of one spoken number "
            f"(under {config.MAX_AUDIO_SECONDS} s)."
        )
    if not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-4:
        raise AudioError("The audio appears to be silent.")
    return {"duration": round(duration, 2), "sample_rate": int(sr)}


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=[m.info() for m in MODELS.values()],
        default_model=default_model_key(),
        allowed=sorted(config.ALLOWED_EXTENSIONS),
        max_mb=config.MAX_UPLOAD_MB,
    )


@app.get("/api/models")
def list_models():
    return jsonify(default=default_model_key(), models=[m.info() for m in MODELS.values()])


@app.post("/api/predict")
def predict():
    file = request.files.get("audio")
    if file is None or not file.filename:
        return jsonify(error="No audio file was uploaded."), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(f".{e}" for e in sorted(config.ALLOWED_EXTENSIONS))
        return jsonify(error=f"Unsupported file type. Please upload one of: {allowed}."), 400

    key = request.form.get("model") or default_model_key()
    model = MODELS.get(key)
    if model is None:
        return jsonify(error=f"Unknown model '{key}'."), 400
    if not model.ready:
        return jsonify(error=f"The {model.spec['name']} model is not available: {model.error}"), 503

    fd, tmp_path = tempfile.mkstemp(suffix="." + ext)
    try:
        with os.fdopen(fd, "wb") as tmp:
            file.save(tmp)
        audio = check_audio(tmp_path)
        result = model.predict(tmp_path)
    except AudioError as e:
        return jsonify(error=str(e)), 400
    except ModelUnavailable as e:
        return jsonify(error=str(e)), 503
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="Something went wrong while processing the audio."), 500
    finally:
        os.remove(tmp_path)

    return jsonify(model=model.info(), audio=audio, **result)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(error=f"File is too large. The limit is {config.MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
