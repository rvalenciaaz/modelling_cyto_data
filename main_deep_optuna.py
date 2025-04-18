# main_deep_optuna.py
import os
import json
import datetime
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch

# Local imports
from src.data_utils import read_and_combine_csv
from src.mad_filter import mad_feature_filter
from src.train_utils import run_nn_optuna  # now has no scaling inside
from src.model_utils import PyTorchNNClassifierWithVal

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_steps = []
def log_message(message: str):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

def main():
    log_steps = []

    # 1) Read data
    combined_df = read_and_combine_csv(pattern="species*.csv", label_prefix="species")

    # 2) MAD filter
    filtered_df, features_to_keep = mad_feature_filter(combined_df, label_col="Label", mad_threshold=5.0)
    log_message(f"Number of features kept after MAD filter: {len(features_to_keep)}")

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

    np.savez("data_for_calibration.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    log_message(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # -----------------------------------------------------------------------
    # Scale the data right after the split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    # -----------------------------------------------------------------------

    # 4) Run Optuna Tuning on scaled data
    log_message("=== Starting Optuna hyperparameter optimization ===")
    best_params, best_score = run_nn_optuna(X_train_scaled, y_train, n_trials=30)
    log_message(f"Optuna best params: {best_params}")
    log_message(f"Optuna best CV accuracy: {best_score:.4f}")

    # (NEW) Save best_params to JSON
    best_params_path = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log_message(f"Saved best_params to '{best_params_path}'")

    # 5) Final training (refit with best hyperparams)
    log_message("Re-fitting final model on entire training set...")
    final_params = best_params.copy()
    final_params["epochs"] = 50  # more epochs
    final_params["verbose"] = True

    # Save scaler & label encoder
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    log_message(f"Saved StandardScaler to '{scaler_path}'")

    le_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
    joblib.dump(label_encoder, le_path)
    log_message(f"Saved LabelEncoder to '{le_path}'")

    # Save feature list
    features_path = os.path.join(OUTPUT_DIR, "features_used.json")
    with open(features_path, "w") as f:
        json.dump(features_to_keep, f, indent=2)

    # Train final classifier on scaled data
    clf = PyTorchNNClassifierWithVal(**final_params)
    clf.fit(X_train_scaled, y_train)

    # If your PyTorchNNClassifierWithVal stores per-epoch losses:
    train_loss_history = getattr(clf, "train_loss_history_", [])
    val_loss_history   = getattr(clf, "val_loss_history_", [])

    # Evaluate on test set
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    log_message(f"Final Test Accuracy: {test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # (NEW) Save CM plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Final NN Model)")
    plt.tight_layout()
    cm_fig_path = os.path.join(OUTPUT_DIR, "confusion_matrix_nn.png")
    plt.savefig(cm_fig_path, bbox_inches='tight')
    plt.close()
    log_message(f"Saved confusion matrix plot to '{cm_fig_path}'")

    # (NEW) Save the confusion matrix data & label classes for replication
    # best done in the same metrics.json or in a separate file
    cm_data_path = os.path.join(OUTPUT_DIR, "cm_data.json")
    cm_dict = {
        "confusion_matrix": cm.tolist(),
        "classes": label_encoder.classes_.tolist(),  # so we know row/column labels
    }
    with open(cm_data_path, "w") as f:
        json.dump(cm_dict, f, indent=2)
    log_message(f"Saved raw confusion matrix data to '{cm_data_path}'")

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    log_message("\nClassification Report:\n" + class_report)

    # Save a plot of training/validation loss vs. epoch (if recorded)
    if train_loss_history and val_loss_history:
        loss_fig_path = os.path.join(OUTPUT_DIR, "loss_plot.png")
        plt.figure()
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_fig_path, bbox_inches='tight')
        plt.close()
        log_message(f"Saved loss plot to '{loss_fig_path}'")

        # Also save the loss history to JSON for replication
        loss_history_path = os.path.join(OUTPUT_DIR, "loss_history.json")
        with open(loss_history_path, "w") as f:
            json.dump(
                {
                    "train_loss": train_loss_history,
                    "val_loss": val_loss_history
                },
                f,
                indent=2
            )
        log_message(f"Saved loss history to '{loss_history_path}'")

    # Save high-level metrics
    metrics_dict = {
        "test_accuracy": float(test_acc),
        "classification_report": class_report,
        "optuna_best_params": best_params,
        "optuna_best_cv_score": float(best_score),
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    log_message(f"Saved metrics to '{metrics_path}'")

    # Save final model
    model_path = os.path.join(OUTPUT_DIR, "best_model_state.pth")
    torch.save(clf.model_.state_dict(), model_path)
    log_message(f"Saved final model state to '{model_path}'")

    # Save a log of steps
    log_path = os.path.join(OUTPUT_DIR, "log_steps.json")
    with open(log_path, "w") as f:
        json.dump(log_steps, f, indent=2)
    log_message(f"Saved detailed log with timestamps to '{log_path}'")

    log_message("All done!")

if __name__ == "__main__":
    main()
