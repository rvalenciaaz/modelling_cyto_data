import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Local imports
from src.data_utils import read_csv_files, log_message
from src.mad_filter import filter_by_mad
from src.train_utils import run_optuna_tuning
from src.model_utils import PyTorchNNClassifierWithVal

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    log_steps = []

    # 1) Read data
    combined_df = read_csv_files(pattern="species*.csv", label_prefix="species", log_steps=log_steps)

    # 2) MAD filter
    filtered_df, features_to_keep = filter_by_mad(combined_df, label_col="Label", mad_threshold=5.0)
    log_message(f"Number of features kept after MAD filter: {len(features_to_keep)}", log_steps=log_steps)

    # 3) Prepare X, y
    X = filtered_df.drop("Label").to_numpy()
    y = filtered_df["Label"].to_numpy()

    # Label Encode y
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    log_message(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}", log_steps=log_steps)

    # 4) Run Optuna
    log_message("=== Starting Optuna hyperparameter optimization ===", log_steps=log_steps)
    best_params, best_score = run_optuna_tuning(X_train, y_train, n_trials=30, n_splits=3)
    log_message(f"Optuna best params: {best_params}", log_steps=log_steps)
    log_message(f"Optuna best CV accuracy: {best_score:.4f}", log_steps=log_steps)

    # 5) Final training (refit with best hyperparams)
    log_message("Re-fitting final model on entire training set...", log_steps=log_steps)
    final_params = best_params.copy()
    final_params["epochs"] = 50  # more epochs than used in the quick search
    final_params["verbose"] = True

    # Scale the entire training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Save scaler & label encoder
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    le_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
    joblib.dump(label_encoder, le_path)

    # Save feature list
    features_path = os.path.join(OUTPUT_DIR, "features_used.json")
    with open(features_path, "w") as f:
        json.dump(features_to_keep, f, indent=2)

    # Train final classifier
    clf = PyTorchNNClassifierWithVal(**final_params)
    clf.fit(X_train_scaled, y_train)

    # Evaluate on test
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    log_message(f"Final Test Accuracy: {test_acc:.4f}", log_steps=log_steps)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Final NN Model)")
    plt.tight_layout()
    cm_fig_path = os.path.join(OUTPUT_DIR, "confusion_matrix_nn.png")
    plt.savefig(cm_fig_path, bbox_inches='tight')
    plt.close()

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    log_message("\nClassification Report:\n" + class_report, log_steps=log_steps)

    # Save metrics
    metrics_dict = {
        "test_accuracy": float(test_acc),
        "classification_report": class_report,
        "optuna_best_params": best_params,
        "optuna_best_cv_score": float(best_score)
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save final model weights
    model_path = os.path.join(OUTPUT_DIR, "best_model_state.pth")
    import torch
    torch.save(clf.model_.state_dict(), model_path)

    log_message(f"Saved final model state to '{model_path}'.", log_steps=log_steps)
    log_message(f"Saved metrics to '{metrics_path}'", log_steps=log_steps)

    # Save a log of steps
    log_path = os.path.join(OUTPUT_DIR, "log_steps.json")
    with open(log_path, "w") as f:
        json.dump(log_steps, f, indent=2)
    log_message(f"Saved detailed log with timestamps to '{log_path}'", log_steps=log_steps)

    log_message("All done!", log_steps=log_steps)

if __name__ == "__main__":
    main()
