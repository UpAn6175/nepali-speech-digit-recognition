# Nepali Speech Digit Recognition

A Flask website that takes an uploaded recording of one spoken number and predicts which number
was said, using the models and feature-extraction code from the project's training notebooks.

## 1. Project overview

The original project trains classifiers that recognise spoken numbers from short WAV clips.
This repository wraps the trained models in a website:

1. The user opens the page and either drags in / selects an audio file, or records one with the
   microphone (**Record** tab). They can play it back.
2. They press **Predict Digit**. The browser sends the file to the Flask backend.
3. The backend runs the **same** librosa feature extraction as the notebook that trained the
   selected model, applies the saved scaler if there is one, and runs the saved model.
4. The page shows the predicted number, a confidence bar and the top 5 candidates
   (for models that output probabilities).

## 2. ML approach

Speech → MFCC features → classifier. There are no delta or delta-delta features, and no extra
preprocessing such as trimming, pre-emphasis or normalisation, beyond what each notebook does.
All MFCCs use librosa defaults: `n_fft=2048`, `hop_length=512`, `win_length=2048`, Hann window, 128 mel bands.

## 3. Dataset

From the notebook code and outputs (the audio itself is not part of this repository):

- **Nepali numbers** (`TrainingCodes/*`): 6 speakers × numbers 1–100 × 3 repetitions = 1,800 WAV clips
  (files like `NEP_S2_D(25)_r1.wav`, recorded at 44.1 or 48 kHz, mostly stereo, 0.4–2.5 s long).
  The raw data is stored per speaker (`1_Organized(22)/` … `6_Organized()/`); the notebooks read it
  re-organised into one folder per number (`All Files/1` … `All Files/100`), which
  [`tools/organize_dataset.py`](tools/organize_dataset.py) rebuilds from the file names.
  The notebooks report 1,797 clips; the current folder has 1,800. Split 80 / 20 with `random_state=42` (360 test clips).
  `samples/` holds 6 clips from the LSTM's **test** split for demos (never seen in training).

The separate Tiwa experiment (`svm_updated.ipynb`, 46 classes) is not part of the website.

## 4. Feature extraction

Implemented in [`preprocessing/features.py`](preprocessing/features.py), copied from each notebook:

| Model | Loading | Features | Shape |
|---|---|---|---|
| LSTM | `librosa.load(path, duration=1.0)` (22,050 Hz, first 1 s) | 40 MFCCs per frame, post zero-padded to 44 frames | (44, 40) |
| SVM (Nepali) | `librosa.load(path, sr=None)` (native rate) | mean of 40 MFCCs | 40 |
| Random Forest | `librosa.load(path, sr=22050)` | mean of 42 MFCCs + mean ZCR + mean spectral centroid | 44 |

## 5. Models used

| Model | Notebook | Hyperparameters | Notebook test acc. | Retrained test acc. (files in `models/`) | Probabilities |
|---|---|---|---|---|---|
| **LSTM (default)** | `TrainingCodes/LSTM(1).ipynb` | LSTM(64, seq) → Dropout .3 → LSTM(32) → Dropout .3 → Dense(100, softmax) | 68.61 % | **78.33 %** | yes (softmax) |
| SVM (Nepali) | `TrainingCodes/SVM(1).ipynb` | SVC rbf, C=10, γ=0.1, StandardScaler | 66.39 % | 66.39 % | no |
| Random Forest | `TrainingCodes/RandomForest(3).ipynb` | 150 trees, max_depth 20, min_samples_leaf 3, balanced | 49.17 % | 46.39 % | yes |

The saved models from the notebooks were not available, so the models in `models/` were retrained with
`train_models.py` (the notebooks' training code, unchanged) on the same dataset. The deterministic SVM
reproduces the notebook's 66.39 % exactly, which confirms the same data, split and features. The LSTM and
RF differ because neural-network training is not bit-for-bit reproducible across TensorFlow versions,
and the RF sees 1,800 instead of 1,797 clips.

**Why the LSTM is the default:** it is the most accurate of the three models trained on the same
100-class dataset, and its softmax output gives a real confidence (the SVM was trained without
`probability=True`). Any model can be selected in the page's dropdown, or made the default with `DIGIT_MODEL`.

## 6. How the models were trained

- **LSTM:** MFCC sequences padded with `pad_sequences(padding='post')`, labels `int(folder) - 1`
  one-hot encoded, Adam + categorical cross-entropy, 200 epochs, batch 32. Saved as `NEPALI_LSTM_model.keras`.
- **SVM (Nepali):** a GridSearchCV over C and γ (best: C=5, γ=0.05, CV 66.9 %), then the saved model was fit
  with C=10, γ=0.1. Labels are indices into the string-sorted folder list.
- **Random Forest:** StandardScaler fit on all features, stratified split, then fit. Labels are the same sorted folder list.

See [`docs/PIPELINE_ANALYSIS.md`](docs/PIPELINE_ANALYSIS.md) for the full analysis.

## 7. How the website works

```
browser ──POST /api/predict (multipart: audio, model)──▶ Flask (app.py)
   validate extension / size (≤10 MB) → save to a temp file → check it decodes, isn't silent, ≤30 s
   → preprocessing/features.py   (same extractor as the notebook)
   → saved scaler (if any)       (loaded once at startup)
   → saved model                 (loaded once at startup)
   → class index → number        (models/labels/*.json)
◀── JSON {digit, confidence, top[5], model, audio}   (temp file deleted)
```

Models are loaded **once** when the app starts ([`inference/predictor.py`](inference/predictor.py)),
never retrained per request. A model whose files are missing is shown as "Not available" and the
rest of the app keeps working.

### Changes compared with the notebook prediction code

All of these are kept as small as possible. No training code or notebook was modified.

1. **LSTM padding length:** the notebook's `predict_digit` called `pad_sequences([mfcc])` with no
   `maxlen`, so a clip shorter than 1 s produced fewer than the 44 time steps used in training.
   The app passes `maxlen` = the model's own input length (read from the loaded model), with the same
   post zero-padding. For clips ≥ 1 s the result is identical to the notebook.
2. **Paths:** absolute `/Users/...` and `/content/drive/...` paths are replaced by the project-relative `models/` folder.
3. **Label lists:** the SVM notebook rebuilt the class list from the dataset folder at prediction
   time. The website has no dataset, so the same sorted list is stored in `models/labels/`.
4. **Confidence:** the SVMs give no probabilities without retraining, so only the label is shown for them.

### Microphone recording

The **Record** tab captures raw audio in the browser (Web Audio API, echo cancellation / noise suppression /
auto-gain turned off), cuts the silence before and after the voice (keeping 0.15 s before and 0.25 s after),
and saves it as a 16-bit mono WAV at the microphone's sample rate (usually 44.1 or 48 kHz, the same rates as the
dataset). That WAV is uploaded exactly like a chosen file, so it goes through the same server-side preprocessing
and model. The silence trimming matters because the training clips are tightly cut and the LSTM only looks at
the first second. Microphone access needs `http://127.0.0.1:5000` or `localhost` (or HTTPS).

### API

- `GET /api/models`: lists the models and whether each one is loaded.
- `POST /api/predict`: form fields `audio` (file) and optional `model` (`nepali_lstm`, `nepali_svm`, `nepali_rf`).

## 8. Project structure

```
nepali-speech-digit-recognition/
├── TrainingCodes/            the original training notebooks (LSTM, SVM, Random Forest)
├── app.py                    Flask app: routes, validation, error handling
├── config.py                 settings + model registry (switch models here or via DIGIT_MODEL)
├── preprocessing/features.py feature extractors copied from the notebooks
├── inference/predictor.py    loads models once, runs predictions
├── models/                   the trained .keras / .joblib files the site uses
│   └── labels/               class-index → number mappings
├── templates/index.html      the page
├── static/css, js, assets    styling, upload / playback / result logic, icon
├── tests/test_app.py         checks the app predicts exactly what the notebook code predicts
├── samples/                  6 held-out test clips for demos
├── train_models.py           recreates the model files with the notebook code
├── tools/organize_dataset.py per-speaker folders → one folder per number
├── tools/make_test_models.py builds placeholder models on synthetic tones (testing only)
├── docs/PIPELINE_ANALYSIS.md analysis of the original notebooks
├── requirements.txt
└── requirements-dev.txt      + pytest
```

## 9. Installation

Requires Python 3.11–3.13. Use a fresh virtual environment: `requirements.txt` pins the exact library
versions the files in `models/` were saved with, which are newer than the ones in your Anaconda base environment.

```bash
git clone https://github.com/UpAn6175/nepali-speech-digit-recognition.git
cd nepali-speech-digit-recognition
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Getting the trained model files

The trained model files are already included in `models/`, so the site works right after installing.
To recreate them, use
[`train_models.py`](train_models.py), which runs each notebook's training cells **unchanged**
(same features, split, `random_state`, hyperparameters and 200 LSTM epochs) and saves the files into `models/`:

```bash
python tools/organize_dataset.py "/path/to/Digit Recognition" "/path/to/All Files"  # once
python train_models.py --dataset "/path/to/All Files"                                # LSTM, SVM, RF
```

It also writes `models/metrics.json` with the new test accuracy, which the page shows instead of the
notebook figure. The only code difference is `int(np.max(y_train)) + 1` for the LSTM class count,
because newer Keras rejects a NumPy integer there; the value is the same. Accuracy can differ slightly
from the notebook printouts, because TensorFlow training is not bit-for-bit reproducible and file order can differ between machines.

If you already have saved files instead, copy them into `models/` (see [`models/README.md`](models/README.md)):

```
models/NEPALI_LSTM_model.keras
models/NEPALI_SVM_model.joblib   models/NEPALI_SVM_scaler.joblib    (optional)
models/NEP_RF_model.joblib       models/NEP_RF_scaler.joblib        (optional)
```

`.joblib` files must be loaded with the **same scikit-learn version** that saved them. If you retrain with
different library versions, update the pins in `requirements.txt` to match.

## 10. Running the website

```bash
python app.py
# open http://127.0.0.1:5000
```

Options (environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `DIGIT_MODEL` | `nepali_lstm` | model selected by default |
| `DIGIT_MODELS_DIR` | `./models` | folder containing the trained files |
| `PORT` | `5000` | port |

## 11. Making predictions

1. Drag a WAV file onto the upload area, or click **choose a file**. Or open the **Record** tab, click the red
   button, say one number, and click again (recording stops by itself after 3 s).
2. Listen to it with the audio player to check it's the right clip.
3. Optionally pick a different model.
4. Press **Predict Digit**. The large number is the prediction, and the bar shows the model's confidence.
5. Press **Clear** to start again.

From the command line:

```bash
curl -F "audio=@NEP_S2_D(25)_r1.wav" http://127.0.0.1:5000/api/predict
```

## 12. Dependencies

Flask (web server), librosa + soundfile (audio loading and MFCCs), NumPy, TensorFlow/Keras (LSTM),
scikit-learn + joblib (SVM, Random Forest, scalers). pytest is only needed for the tests.
Notebook-only packages (pandas, matplotlib, seaborn) are not needed by the website.

## 13. Testing

```bash
pip install -r requirements-dev.txt
# with the real models in models/ and a few WAV files from the dataset:
TEST_WAV_DIR=/path/to/some/wavs python -m pytest tests -q
# or, before the real models are available, with placeholder models:
python tools/make_test_models.py /tmp/digit_test_models
DIGIT_MODELS_DIR=/tmp/digit_test_models python -m pytest tests -q
```

The tests run each notebook's own prediction code next to the web API and assert that both
give the same answer (and, for the LSTM, the same probability).

## 14. Limitations

- Accuracy is that of the notebook models: 78.3 % on the 360 held-out Nepali test clips for the retrained LSTM (6 speakers only).
  Expect lower accuracy on speakers, microphones or noise levels not in the dataset.
- The LSTM only listens to the **first second** of the clip, as in training. Recordings are trimmed automatically; trim leading silence from uploaded files yourself.
- The model was trained on 6 speakers, so new voices and microphones (including live recordings) are less accurate than the test-set figure.
- The Nepali SVM uses the file's native sample rate, as in training, so recordings at a rate
  different from the dataset's produce shifted features.
- SVM models show no confidence, because they were trained without `probability=True`.
- The LSTM's reported accuracy is measured on the same split used as `validation_data` during training,
  so it is slightly optimistic. The RF scaler was fit before the train/test split.
- WAV is the format used in training. FLAC/OGG/MP3 are decoded by librosa/soundfile and resampled the same way, but were not part of training.
- Flask's built-in server is for local use. For a public deployment use a WSGI server such as `gunicorn app:app`.

## 15. Possible future improvements

- Retrain the SVM with `probability=True` (and the grid-search best parameters) to get confidences.
- Fit the RF scaler on the training split only, and use a separate validation split for the LSTM.
- Add delta / delta-delta features or data augmentation (noise, speed, pitch) to improve robustness.
- Voice-activity detection to trim silence before the 1-second window.
- Docker image for one-command deployment.
