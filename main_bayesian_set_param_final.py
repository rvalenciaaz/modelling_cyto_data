# main_bay.py
import os
import json
import datetime
import pickle
import numpy as np
import polars as pl

import torch
import pyro

# scikit-learn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Plotting
import matplotlib.pyplot as plt

# Local project imports
from src.data_utils import read_and_combine_csv  # used by everyone
from src.mad_filter import mad_feature_filter    # used by everyone
from src.train_utils import (
    train_pyro_model,          # Bayesian
    train_pyro_model_with_val  # Bayesian (if needed later)
)
from src.prediction_utils import predict_pyro_model  # Bayesian

# Fix random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
pyro.set_rng_seed(42)

# ---------------------------------------------------------
# OUTPUT FOLDER & LOGGING SETUP
# ---------------------------------------------------------
OUTPUT_FOLDER = "replication_files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, "main_bay.log")

log_steps = []
def log_message(message: str):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now_str}] {message}"
    print(message)
    log_steps.append(line)
    # write immediately to log file
    with open(LOG_FILE_PATH, "a") as f:
        f.write(line + "\n")


def main():
    # ---------------------------------------------------------
    # 1. READ & COMBINE CSV FILES
    # ---------------------------------------------------------
    log_message("Reading CSV files (pattern='species*.csv')...")
    combined_df = read_and_combine_csv(label_prefix="species", pattern="species*.csv")
    log_message(f"Combined dataset shape: {combined_df.shape}")

    # ---------------------------------------------------------
    # 2. MAD-BASED FEATURE FILTER
    # ---------------------------------------------------------
    log_message("Applying MAD filter (threshold=5)...")
    final_df, features_to_keep = mad_feature_filter(
        data=combined_df, label_col="Label", mad_threshold=5
    )
    log_message(f"Kept {len(features_to_keep)} features after MAD filter.")

    # ---------------------------------------------------------
    # 3. TRAIN/TEST SPLIT + SCALING
    # ---------------------------------------------------------
    X = final_df.drop("Label").to_numpy()
    y = final_df.select("Label").to_numpy().ravel()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.int32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.int32)

    # ---------------------------------------------------------
    # 4. USER-DEFINED HYPERPARAMETERS
    # ---------------------------------------------------------
    hidden_size   = 96
    num_layers    = 3
    learning_rate = 0.008958470263677291

    log_message("Using user-defined hyperparameters:")
    log_message(f"  hidden_size   = {hidden_size}")
    log_message(f"  num_layers    = {num_layers}")
    log_message(f"  learning_rate = {learning_rate}")
    # Persist these hyper-parameters to disk so other scripts
    # (or a later run) can pick them up.
    best_params = {
        "hidden_size":   hidden_size,
        "num_layers":    num_layers,
        "learning_rate": learning_rate
    }
    best_params_file = os.path.join(OUTPUT_FOLDER, "best_params.json")
    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=2)
    log_message(f"Saved best hyperparams => {best_params_file}")
    # ---------------------------------------------------------
    # 5. FINAL TRAINING ON FULL TRAIN SET + TEST EVAL
    # ---------------------------------------------------------
    log_message("Final training on full training data...")
    num_epochs_final = 20000
    final_guide, final_train_losses = train_pyro_model(
        X_train_t, y_train_t,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=len(np.unique(y_train)),
        learning_rate=learning_rate,
        num_epochs=num_epochs_final,
        verbose=True
    )

    # Save final training losses
    final_losses_file = os.path.join(OUTPUT_FOLDER, "final_losses.pkl")
    with open(final_losses_file, "wb") as f:
        pickle.dump(final_train_losses, f)
    log_message(f"Saved final training losses => {final_losses_file}")

    # Plot final training loss
    epochs_fin = range(1, len(final_train_losses)+1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_fin, final_train_losses, label='Final Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final Model Training Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    final_plot_path = os.path.join(OUTPUT_FOLDER, "final_training_loss.png")
    plt.savefig(final_plot_path)
    plt.close()
    log_message(f"Saved final training loss plot => {final_plot_path}")

    # ---------------------------------------------------------
    # 6. TEST SET EVALUATION
    # ---------------------------------------------------------
    log_message("Predicting on test set with final model...")
    test_preds = predict_pyro_model(
        X_test_t,
        final_guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=len(np.unique(y_train)),
        num_samples=1000
    )
    test_acc = accuracy_score(y_test_t, test_preds)
    class_rep = classification_report(y_test_t, test_preds, target_names=label_encoder.classes_)

    log_message(f"Test Accuracy = {test_acc:.4f}")
    log_message("\nClassification Report:\n" + class_rep)

    # Save & plot confusion matrix
    cm = confusion_matrix(y_test_t, test_preds)
    cm_path = os.path.join(OUTPUT_FOLDER, "confusion_matrix.npy")
    np.save(cm_path, cm)
    log_message(f"Saved confusion matrix array => {cm_path}")

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Test Set Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    cm_plot_path = os.path.join(OUTPUT_FOLDER, "test_confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()
    log_message(f"Saved confusion matrix plot => {cm_plot_path}")

    # ---------------------------------------------------------
    # 7. SAVE ARTIFACTS
    # ---------------------------------------------------------
    metrics_dict = {
        "user_defined_hyperparams": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate
        },
        # Ensure cv_mean_accuracy and cv_std_accuracy are defined if using cross-validation
        # "cv_mean_accuracy": float(cv_mean_accuracy),
        # "cv_std_accuracy": float(cv_std_accuracy),
        "test_accuracy": float(test_acc),
        "classification_report": class_rep
    }
    metrics_path = os.path.join(OUTPUT_FOLDER, "metrics_pyro.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    log_message(f"Saved metrics => {metrics_path}")

    user_params_file = os.path.join(OUTPUT_FOLDER, "user_params.json")
    with open(user_params_file, "w") as f:
        json.dump({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate
        }, f, indent=2)
    log_message(f"Saved user hyperparams => {user_params_file}")

    guide_state = final_guide.state_dict()
    guide_params_path = os.path.join(OUTPUT_FOLDER, "bayesian_nn_pyro_params.pkl")
    with open(guide_params_path, "wb") as f:
        pickle.dump(guide_state, f)
    log_message(f"Saved final guide params => {guide_params_path}")

    scaler_path = os.path.join(OUTPUT_FOLDER, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log_message(f"Saved scaler => {scaler_path}")

    encoder_path = os.path.join(OUTPUT_FOLDER, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    log_message(f"Saved label encoder => {encoder_path}")

    features_path = os.path.join(OUTPUT_FOLDER, "features_to_keep.json")
    with open(features_path, "w") as f:
        json.dump(features_to_keep, f, indent=2)
    log_message(f"Saved features to keep => {features_path}")

    # Save timestamped log steps
    log_path = os.path.join(OUTPUT_FOLDER, "log_steps_pyro.json")
    with open(log_path, "w") as f:
        json.dump(log_steps, f, indent=2)
    log_message(f"Saved log => {log_path}")

    log_message("All done!")


if __name__ == "__main__":
    main()
