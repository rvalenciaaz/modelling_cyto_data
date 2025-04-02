import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import os
import joblib

# --- Ensure custom classes are available for pickle ---
from train_and_save import PyTorchNNClassifierWithVal, ConfigurableNN
import __main__
setattr(__main__, "PyTorchNNClassifierWithVal", PyTorchNNClassifierWithVal)
setattr(__main__, "ConfigurableNN", ConfigurableNN)

# --- Define a simple FrozenEstimator (if not available) ---
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    class FrozenEstimator:
        def __init__(self, estimator):
            self.estimator = estimator
        def predict(self, X):
            return self.estimator.predict(X)
        def predict_proba(self, X):
            return self.estimator.predict_proba(X)

def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] {message}")

# ---------------------------------------------------------
# LOAD THE SAVED ARTIFACTS
# ---------------------------------------------------------
scaler_path = "scaler.pkl"
model_path = "best_nn_model.pkl"
calib_data_path = "data_for_calibration.npz"

for path in [scaler_path, model_path, calib_data_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file '{path}' not found. Please ensure it was saved during training.")

# Load the scaler
scaler = joblib.load(scaler_path)
log_message(f"Loaded scaler from '{scaler_path}'.")

# Load the trained classifier (e.g. an instance of PyTorchNNClassifierWithVal)
classifier = joblib.load(model_path)
log_message(f"Loaded classifier from '{model_path}'.")

# Load calibration data
data = np.load(calib_data_path)
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]
log_message(f"Loaded calibration data from '{calib_data_path}'.")

# ---------------------------------------------------------
# APPLY PREPROCESSING
# ---------------------------------------------------------
# Use the scaler to transform the data
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------------------------
# WRAP THE CLASSIFIER FOR CALIBRATION
# ---------------------------------------------------------
# Wrap the (already trained) classifier so that it can be used with cv="prefit"
calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(classifier), method='sigmoid', cv='prefit')
calibrator.fit(X_train_scaled, y_train)
log_message("Calibration complete using CalibratedClassifierCV.")

# ---------------------------------------------------------
# EVALUATE THE CALIBRATED MODEL
# ---------------------------------------------------------
y_calibrated_proba = calibrator.predict_proba(X_test_scaled)
y_calibrated_pred = np.argmax(y_calibrated_proba, axis=1)

calibrated_accuracy = accuracy_score(y_test, y_calibrated_pred)
calibrated_f1_macro = f1_score(y_test, y_calibrated_pred, average='macro')
calibrated_f1_weighted = f1_score(y_test, y_calibrated_pred, average='weighted')

log_message(f"Calibrated Model Accuracy on Test Set: {calibrated_accuracy:.4f}")
log_message(f"Calibrated Model F1-macro on Test Set: {calibrated_f1_macro:.4f}")
log_message(f"Calibrated Model F1-weighted on Test Set: {calibrated_f1_weighted:.4f}")

class_report_calibrated = classification_report(y_test, y_calibrated_pred)
log_message("\nClassification Report (Calibrated):")
log_message(class_report_calibrated)

# ---------------------------------------------------------
# SAVE THE CALIBRATED MODEL
# ---------------------------------------------------------
with open("calibrator.pkl", "wb") as f:
    pickle.dump(calibrator, f)
log_message("Saved calibrator to 'calibrator.pkl'.")

