# Trained model files

This folder holds the trained model files (recreated with `train_models.py`). The app loads every model whose files it finds
at startup and shows the rest as "Not available".

| Model | Files | Saved by |
|---|---|---|
| LSTM (default) | `NEPALI_LSTM_model.keras` | `TrainingCodes/LSTM(1).ipynb`, cell 19 |
| SVM (Nepali) | `NEPALI_SVM_model.joblib`, `NEPALI_SVM_scaler.joblib` | `TrainingCodes/SVM(1).ipynb`, cell 10 |
| Random Forest | `NEP_RF_model.joblib`, `NEP_RF_scaler.joblib` | `TrainingCodes/RandomForest(3).ipynb`, cell 8 |

`labels/` holds the class-index to number mappings taken from the notebooks:

- `nepali_lstm_labels.json`: index `i` is number `i + 1` (`label = int(folder) - 1` in training).
- `nepali_sorted_folder_labels.json`: folder names sorted as strings (`"1", "10", "100", "11", …`),
  the order the SVM and Random Forest notebooks used.
