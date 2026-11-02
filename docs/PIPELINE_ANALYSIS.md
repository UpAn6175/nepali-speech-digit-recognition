# Pipeline analysis of the existing notebooks

Written before any web code, from the six notebooks in the project. Nothing in the
original notebooks has been modified.

## 1. What the project does

Spoken-number recognition: a short WAV clip of one spoken number goes in, a class label comes out.
Every notebook follows the same shape: `librosa` MFCC features, then a classifier.

The six notebooks cover **three different datasets**:

| Notebook | Dataset (from paths / outputs) | Classes | Samples | Model | Held-out accuracy |
|---|---|---|---|---|---|
| `TrainingCodes/LSTM(1).ipynb` | Nepali, `Organized Data/All Files`, folders `1`–`100` | 100 | 1,797 | 2-layer LSTM | **68.61 %** (360 test) |
| `TrainingCodes/SVM(1).ipynb` | same Nepali set | 100 | 1,797 | SVC (RBF, C=10, γ=0.1) | 66.39 % (360 test) |
| `TrainingCodes/RandomForest(3).ipynb` | same Nepali set | 100 | 1,797 | Random Forest | 49.17 % (360 test) |
| `svm_updated.ipynb` | "Tiwa_Digit", Google Drive, `dataset.csv` | 46 (1–30, 40…100, 200…1000) | 1,040 | SVC (grid search, best C=10, γ=scale, RBF) | 76.28 % (156 test) |
| `randomforest.ipynb` | `./Digitdataset` (same 46 Tiwa classes) | 46 | 1,040 | Random Forest | 50.96 % (208 test) |
| `lstm.ipynb` | `./digitdataset` (likely Tiwa, earlier version) | 46 | ~790 | 1-layer LSTM | 33.96 % |

The `TrainingCodes` folder is the most complete set: it has all three models on one dataset,
each one saves its model (`.keras` / `.joblib`), and each has a working single-file prediction cell.
The root `lstm.ipynb` / `randomforest.ipynb` use the same 46 Tiwa classes as `svm_updated`, ran on another machine, and save no model.

## 2. Training pipelines (exact parameters)

All `librosa.feature.mfcc` calls use librosa defaults: `n_fft=2048`, `hop_length=512`,
`win_length=2048`, Hann window, `n_mels=128`, mono. No delta or delta-delta features are used anywhere.

**Nepali LSTM** (`LSTM(1)`, cells 14–19, the run that produced 68.61 %)
1. `librosa.load(path, duration=1.0)`: resampled to 22,050 Hz, first 1 s only
2. `mfcc(n_mfcc=40).T`: shape (frames, 40), ≤ 44 frames for 1 s
3. `pad_sequences(padding='post', dtype='float32')`: zero-padded to the longest clip, giving (44, 40)
4. No scaling
5. Labels: `int(folder) - 1`, so class index 0…99 means number 1…100
6. Split 80/20, stratified, `random_state=42`
7. Model: LSTM(64, return_sequences) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(100, softmax); Adam, categorical cross-entropy, 200 epochs, batch 32
8. Saved as `NEPALI_LSTM_model.keras`

**Nepali SVM** (`SVM(1)`)
1. `librosa.load(path, sr=None)`: **native sample rate**, full clip
2. `mfcc(n_mfcc=40)`, mean over time, giving a 40-dim vector
3. Split 80/20, **not stratified**, `random_state=42`
4. `StandardScaler` fit on train
5. GridSearchCV found C=5, γ=0.05 (CV 66.9 %), but the saved model is `SVC(C=10, gamma=0.1)` from the next cell
6. Labels: index into **lexicographically sorted** folder names (`'1','10','100','11',…`)
7. Saved as `NEPALI_SVM_model.joblib` + `NEPALI_SVM_scaler.joblib`

**Nepali Random Forest** (`RandomForest(3)`)
1. `librosa.load(path, sr=22050)`, full clip
2. 42 MFCC means + mean ZCR + mean spectral centroid, giving 44 dims
3. `StandardScaler` fit on **all** data, then split 80/20 stratified
4. RF(n_estimators=150, max_depth=20, min_samples_leaf=3, class_weight='balanced', random_state=42)
5. Labels: sorted folder names (same order as SVM; the notebook hard-codes the map)
6. Saved as `NEP_RF_model.joblib` + `NEP_RF_scaler.joblib`

**Tiwa SVM** (`svm_updated`)
1. `librosa.load(path, sr=16000)`, full clip
2. 13 MFCC mean + 13 MFCC std, giving 26 dims
3. 70/15/15 stratified split from `dataset.csv`, scaler fit on train
4. GridSearchCV (C, gamma, kernel) gave C=10, gamma='scale', rbf
5. Labels: the integer `label` column itself (prediction *is* the number)
6. Saved as `Tiwa_SVM_model.joblib` + `Tiwa_SVM_scaler.joblib`

## 3. Testing pipelines

Each notebook evaluates with `accuracy_score` + `classification_report` on the held-out split, and
`svm_updated` adds a confusion matrix and a grid-search plot. Each also has a "predict one file" cell
that reloads the saved model and repeats the training-time feature extraction. Those cells are the
reference for the web backend.

## 4. Recommended model for the website

**Nepali LSTM** (`NEPALI_LSTM_model.keras`), because:
- It has the highest held-out accuracy of the three models trained on the same 100-class dataset (68.6 % vs 66.4 % SVM vs 49.2 % RF).
- Its softmax output gives real class probabilities, so the site can show confidence and a top-5 chart.
  The SVMs were trained without `probability=True`, so they only give a label.
- It needs no scaler file, so there's one less thing that can go missing.

The Tiwa SVM has the best raw number (76.3 %), but it's a different, smaller dataset and label set
and gives no probabilities. The backend supports all four saved models through a registry, so
switching models only changes one setting.

## 5. What has to change for deployment (smallest changes only)

1. **LSTM padding length.** In training every clip was padded to 44 frames, but the notebook's
   `predict_digit` pads each file only to its own length. A clip shorter than 1 s then gives
   fewer than 44 time steps, which is a different input shape from the one the model was trained on.
   Fix: pad (post, zeros, as in training) to the model's own `input_shape[1]`, read from the loaded model.
2. **Hard-coded `/Users/...` and `/content/drive/...` paths.** Replaced with a project-relative `models/` folder.
3. **Label lists that are rebuilt from the dataset folder at prediction time** (SVM). The website
   has no dataset, so the sorted folder list printed in the notebook output is saved as a small JSON file next to the model.
4. **Confidence for SVMs.** None is available without retraining, so the site shows only the label for those models. The model is not retrained.
5. **Version pinning.** `.joblib` files must load with the same scikit-learn version that saved them,
   and `.keras` files with Keras 3 / TF 2.19. The LSTM notebook also downgrades NumPy to 1.24.4, which
   TF 2.19 itself rejects (it needs NumPy ≥ 1.26), so `requirements.txt` uses a compatible NumPy.

Things worth knowing but **not changed**, because changing them would mean retraining:
SVM split not stratified, and its saved model isn't the grid-search winner. RF scaler is fit before the
split (slight leakage). The LSTM uses the test set as `validation_data`, so 68.6 % is also the
number training was watched on. The Nepali SVM uses the file's native sample rate, so uploads
recorded at a different rate than the dataset will produce shifted features.

## 6. Proposed architecture

**Flask**: the existing code is plain synchronous Python, prediction takes milliseconds, and
Flask's `templates/` + `static/` layout matches the requested structure.

```
speech-digit-webapp/
├── app.py                 Flask app: routes, upload validation, error handling
├── config.py              model registry + active model setting (env var DIGIT_MODEL)
├── preprocessing/
│   └── features.py        feature extractors copied 1:1 from the notebooks
├── inference/
│   └── predictor.py       loads the model + scaler + labels ONCE at startup, predicts
├── models/                put the saved .keras / .joblib files here (+ label JSONs)
├── templates/index.html
├── static/css, static/js
├── tests/                 pipeline-equivalence and API tests
├── requirements.txt
└── README.md
```

Request flow: browser uploads file → `POST /api/predict` → validate extension and size → save to a temp
file → the selected model's extractor (identical to its notebook) → scaler, if the model has one → model →
JSON `{digit, confidence, top_k}` → temp file deleted → UI shows result.
