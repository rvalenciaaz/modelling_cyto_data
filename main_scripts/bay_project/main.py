import os
import json
import datetime
import pickle
import numpy as np
import torch
import pyro

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

# Local imports from src/
from src.data_utils import read_and_combine_csv
from src.mad_filter import mad_feature_filter
from src.train_utils import (
    train_pyro_model, train_pyro_model_with_val,
    predict_pyro_model, tune_hyperparameters
)
from src.prediction_utils import predict_pyro_model as predict_pyro_model_fn
from src.prediction_utils import predict_pyro_probabilities

# Fix seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
pyro.set_rng_seed(42)

# 0. Logging utility & output folder
log_steps = []
def log_message(message: str):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

OUTPUT_FOLDER = "replication_files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def main():
    # ---------------------------------------------------------
    # 1. READ CSV FILES & COMBINE
    # ---------------------------------------------------------
    log_message("Reading and combining CSV files...")
    combined_df = read_and_combine_csv(label_prefix="species", pattern="species*.csv")
    log_message(f"Combined dataset shape: {combined_df.shape}")

    # ---------------------------------------------------------
    # 2. MAD-BASED FEATURE FILTER
    # ---------------------------------------------------------
    log_message("Filtering features by MAD...")
    final_df, features_to_keep = mad_feature_filter(combined_df, label_col="Label", mad_threshold=5)
    log_message(f"Kept {len(features_to_keep)} features after MAD filter.")

    # ---------------------------------------------------------
    # 3. TRAIN/TEST SPLIT & SCALING
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
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    # ---------------------------------------------------------
    # 4. OPTUNA HYPERPARAMETER TUNING
    # ---------------------------------------------------------
    log_message("Starting hyperparameter tuning with Optuna...")
    best_params = tune_hyperparameters(X_train_t, y_train_t, n_trials=40)
    hidden_size   = best_params["hidden_size"]
    num_layers    = best_params["num_layers"]
    learning_rate = best_params["learning_rate"]
    log_message(f"Best hyperparameters: {best_params}")

    # ---------------------------------------------------------
    # 5. 5-FOLD CROSS VALIDATION
    # ---------------------------------------------------------
    log_message("Starting 5-fold CV with best hyperparams...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    num_epochs_cv = 20000
    fold_train_losses = []
    fold_val_losses   = []
    fold_accuracies  = []

    from src.train_utils import train_pyro_model_with_val
    from src.prediction_utils import predict_pyro_model

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[tr_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[tr_idx], y_train[val_idx]

        X_fold_train_t = torch.tensor(X_fold_train, dtype=torch.float32)
        y_fold_train_t = torch.tensor(y_fold_train, dtype=torch.long)
        X_fold_val_t   = torch.tensor(X_fold_val,   dtype=torch.float32)
        y_fold_val_t   = torch.tensor(y_fold_val,   dtype=torch.long)

        guide_cv, train_losses_cv, val_losses_cv = train_pyro_model_with_val(
            X_fold_train_t, y_fold_train_t,
            X_fold_val_t,   y_fold_val_t,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=len(np.unique(y_train)),
            learning_rate=learning_rate,
            num_epochs=num_epochs_cv,
            verbose=False
        )

        fold_train_losses.append(train_losses_cv)
        fold_val_losses.append(val_losses_cv)

        val_preds_fold = predict_pyro_model(
            X_fold_val_t, guide_cv,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=len(np.unique(y_train)),
            num_samples=300
        )
        fold_acc = accuracy_score(y_fold_val_t, val_preds_fold)
        fold_accuracies.append(fold_acc)
        log_message(f"[Fold {fold_idx+1}] Accuracy = {fold_acc:.4f}")

    cv_mean_accuracy = np.mean(fold_accuracies)
    cv_std_accuracy  = np.std(fold_accuracies)
    log_message(f"CV Accuracy: {cv_mean_accuracy:.4f} ± {cv_std_accuracy:.4f}")

    # Save fold losses
    cv_fold_losses_path = os.path.join(OUTPUT_FOLDER, "cv_fold_losses.pkl")
    with open(cv_fold_losses_path, "wb") as f:
        pickle.dump({"train_losses": fold_train_losses, "val_losses": fold_val_losses}, f)

    # Plot all-folds loss
    plt.figure(figsize=(10, 6))
    for i in range(len(fold_train_losses)):
        epochs = range(1, len(fold_train_losses[i]) + 1)
        plt.plot(epochs, fold_train_losses[i], label=f"Train Fold {i+1}")
        plt.plot(epochs, fold_val_losses[i],   label=f"Val Fold {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("All Folds Train & Val Loss Curves")
    plt.legend(loc="upper right", ncol=2, fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    all_folds_plot = os.path.join(OUTPUT_FOLDER, "cv_folds_loss_trends.png")
    plt.savefig(all_folds_plot)
    plt.close()

    # Plot aggregated mean ± std
    all_train_arr = np.array(fold_train_losses)
    all_val_arr   = np.array(fold_val_losses)
    mean_train = all_train_arr.mean(axis=0)
    std_train  = all_train_arr.std(axis=0)
    mean_val   = all_val_arr.mean(axis=0)
    std_val    = all_val_arr.std(axis=0)

    epochs = range(1, all_train_arr.shape[1] + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_train, label="Train Loss (mean)")
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.3)
    plt.plot(epochs, mean_val, label="Val Loss (mean)")
    plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Aggregated CV Loss (Mean ± Std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    agg_folds_plot = os.path.join(OUTPUT_FOLDER, "cv_aggregated_loss.png")
    plt.savefig(agg_folds_plot)
    plt.close()

    # ---------------------------------------------------------
    # 6. FINAL TRAINING ON FULL TRAIN SET + TEST EVAL
    # ---------------------------------------------------------
    log_message("Final training on full training data...")
    num_epochs_final = 20000
    from src.train_utils import train_pyro_model
    final_guide, final_train_losses = train_pyro_model(
        X_train_t, y_train_t,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=len(np.unique(y_train)),
        learning_rate=learning_rate,
        num_epochs=num_epochs_final,
        verbose=True
    )

    # Save final train loss
    final_losses_file = os.path.join(OUTPUT_FOLDER, "final_losses.pkl")
    with open(final_losses_file, "wb") as f:
        pickle.dump(final_train_losses, f)

    # Plot final train loss
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

    # TEST PREDICTION
    log_message("Predicting on test set with final model...")
    from src.prediction_utils import predict_pyro_model
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

    # ---------------------------------------------------------
    # 7. SAVE ARTIFACTS (metrics, model, scaler, encoder, features, logs)
    # ---------------------------------------------------------
    metrics_dict = {
        "best_hyperparams": best_params,
        "cv_mean_accuracy": float(cv_mean_accuracy),
        "cv_std_accuracy": float(cv_std_accuracy),
        "test_accuracy": float(test_acc),
        "classification_report": class_rep
    }
    metrics_path = os.path.join(OUTPUT_FOLDER, "metrics_pyro.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    guide_state = final_guide.state_dict()
    guide_params_path = os.path.join(OUTPUT_FOLDER, "bayesian_nn_pyro_params.pkl")
    with open(guide_params_path, "wb") as f:
        pickle.dump(guide_state, f)

    scaler_path = os.path.join(OUTPUT_FOLDER, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    encoder_path = os.path.join(OUTPUT_FOLDER, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    features_path = os.path.join(OUTPUT_FOLDER, "features_to_keep.json")
    with open(features_path, "w") as f:
        json.dump(features_to_keep, f, indent=2)

    # Save log steps
    log_path = os.path.join(OUTPUT_FOLDER, "log_steps_pyro.json")
    with open(log_path, "w") as f:
        json.dump(log_steps, f, indent=2)

    log_message("All done!")

if __name__ == "__main__":
    main()
