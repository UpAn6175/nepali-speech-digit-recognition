"""Train the models with the notebooks' own code and save them into models/.

The original notebooks saved their models to files that are no longer
available. This script re-runs the training cells of each notebook unchanged
(same feature extraction, split, random_state, hyperparameters, epochs) and
writes the files the website loads. It does not change the methodology.

Usage (run on the computer that has the dataset):

    # Nepali LSTM, SVM and Random Forest from the "All Files" folder (subfolders 1..100)
    python train_models.py --dataset "/Users/ankitaupadhyaya/Documents/Audio Files/Organized Data/All Files"

    # only some models
    python train_models.py --dataset "..." --models lstm svm

Accuracy can differ slightly from the notebook printouts: TensorFlow training is
not bit-for-bit reproducible, and os.listdir() file order can differ between
machines, which changes which clips land in the test split.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

OUT_DIR = Path(os.environ.get("DIGIT_MODELS_DIR", Path(__file__).resolve().parent / "models"))
METRICS_FILE = "metrics.json"


def save_metric(key, accuracy):
    path = OUT_DIR / METRICS_FILE
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics[key] = round(float(accuracy), 4)
    path.write_text(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# TrainingCodes/LSTM(1).ipynb, cells 14-19
# ---------------------------------------------------------------------------
def train_lstm(folder_path, epochs=200):
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical

    # 1. Extract MFCC features
    def extract_mfcc_sequences(folder_path, n_mfcc=40, duration=1.0):
        data = []
        labels = []

        for digit_folder in sorted(os.listdir(folder_path)):
            digit_path = os.path.join(folder_path, digit_folder)

            if not os.path.isdir(digit_path):
                continue

            label = int(digit_folder) - 1  # Convert folder name (e.g., '99') to int and subtract 1 for 0-based labels

            for file in os.listdir(digit_path):
                if not file.lower().endswith(".wav"):
                    continue

                file_path = os.path.join(digit_path, file)
                try:
                    y, sr = librosa.load(file_path, duration=duration)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
                    data.append(mfcc)
                    labels.append(label)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        return data, labels

    X_sequences, y = extract_mfcc_sequences(folder_path)

    # 2. Pad sequences
    X_sequences_padded = pad_sequences(X_sequences, padding='post', dtype='float32')

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences_padded, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # int(): newer Keras rejects a NumPy integer for Dense units; value is unchanged
    num_classes = int(np.max(y_train)) + 1  # Automatically detect number of classes
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Cell 18: evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:\n", classification_report(y_true, y_pred, zero_division=0))

    # Cell 19
    model.save(OUT_DIR / "NEPALI_LSTM_model.keras")
    save_metric("nepali_lstm", acc)


# ---------------------------------------------------------------------------
# TrainingCodes/SVM(1).ipynb, cells 1-10
# ---------------------------------------------------------------------------
def train_svm(dataset_path, grid_search=False):
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler

    def extract_features(file_path):
        """Extract MFCC features from a WAV file"""
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)

    def load_dataset(dataset_path):
        """Load dataset from folder structure"""
        class_folders = sorted([f for f in os.listdir(dataset_path)
                                if os.path.isdir(os.path.join(dataset_path, f))])

        features = []
        labels = []

        for label, folder in enumerate(class_folders):
            folder_path = os.path.join(dataset_path, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        feature = extract_features(file_path)
                        features.append(feature)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        return np.array(features), np.array(labels), class_folders

    print("Loading dataset...")
    X, y, class_names = load_dataset(dataset_path)
    print(f"Loaded {len(X)} samples with {len(class_names)} classes")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if grid_search:  # cell 5: its result was printed but not used for the saved model
        param_grid = {'C': [1, 5, 10, 20], 'gamma': [0.01, 0.05, 0.1, 0.5], 'kernel': ['rbf']}
        grid = GridSearchCV(svm.SVC(), param_grid, cv=5)
        grid.fit(X_train, y_train)
        print("Best parameters:", grid.best_params_)
        print("Best cross-validation accuracy:", grid.best_score_)

    # Cell 6: the model that was saved
    print("\nTraining SVM classifier...")
    clf = svm.SVC(kernel='rbf', C=10, gamma=0.1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred, labels=range(len(class_names)),
                                target_names=class_names, zero_division=0))
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    check_label_order(class_names)
    joblib.dump(clf, OUT_DIR / 'NEPALI_SVM_model.joblib')
    joblib.dump(scaler, OUT_DIR / 'NEPALI_SVM_scaler.joblib')
    save_metric("nepali_svm", accuracy)


# ---------------------------------------------------------------------------
# TrainingCodes/RandomForest(3).ipynb, cells 1-8
# ---------------------------------------------------------------------------
def train_rf(DATASET_PATH):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SAMPLE_RATE = 22050
    N_MFCC = 42

    def extract_features(file_path):
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # ZCR
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T)

        # Spectral centroid
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T)

        # Combine
        features = np.concatenate([mfcc_mean, [zcr, centroid]])
        return features

    X, y = [], []
    class_labels = sorted([
        f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))
    ])
    label_map = {label: idx for idx, label in enumerate(class_labels)}

    for label in class_labels:
        folder_path = os.path.join(DATASET_PATH, label)
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                try:
                    feat = extract_features(file_path)
                    X.append(feat)
                    y.append(label_map[label])
                except Exception as e:
                    print(f"Failed {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # RF Model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification Report:\n", classification_report(
        y_test, y_pred, labels=range(len(class_labels)), target_names=class_labels, zero_division=0))

    check_label_order(class_labels)
    joblib.dump(model, OUT_DIR / 'NEP_RF_model.joblib')
    joblib.dump(scaler, OUT_DIR / 'NEP_RF_scaler.joblib')
    save_metric("nepali_rf", acc)


def check_label_order(class_names):
    """The website maps class index -> number with the same sorted folder list."""
    expected = json.loads((Path(__file__).resolve().parent / "models/labels/nepali_sorted_folder_labels.json").read_text())
    if list(class_names) != expected:
        sys.exit(f"Dataset folders {class_names[:5]}... do not match models/labels/nepali_sorted_folder_labels.json; "
                 "update that file to the printed folder list before using this model.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="Nepali dataset folder with one subfolder per number (1..100)")
    p.add_argument("--models", nargs="+", choices=["lstm", "svm", "rf"],
                   help="which models to train (default: all three)")
    p.add_argument("--epochs", type=int, default=200, help="LSTM epochs (notebook: 200)")
    p.add_argument("--svm-grid-search", action="store_true",
                   help="also re-run the SVM notebook's grid search (printed only, as in the notebook)")
    args = p.parse_args()

    models = args.models or ["lstm", "svm", "rf"]
    if not args.dataset:
        p.error("--dataset is required")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for m in models:
        print(f"\n===== Training {m} =====")
        if m == "lstm":
            train_lstm(args.dataset, epochs=args.epochs)
        elif m == "svm":
            train_svm(args.dataset, grid_search=args.svm_grid_search)
        elif m == "rf":
            train_rf(args.dataset)
    print(f"\nSaved to {OUT_DIR}. Start the website with: python app.py")


if __name__ == "__main__":
    main()
