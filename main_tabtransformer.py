# main.py
import os
import json
import datetime
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import torch

# Local project imports
from src.data_utils import read_and_combine_csv
from src.mad_filter import mad_feature_filter
from train_utils import objective
from model_utils import TabTransformerClassifierWithVal

# ---------------------------
# 0. LOGGING UTILITY
# ---------------------------
log_steps = []
def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# ---------------------------
# 1. READ CSV FILES
# ---------------------------
log_message("Reading CSV files (pattern='species*.csv') via src.data_utils...")
combined_pl = read_and_combine_csv(label_prefix="species", pattern="species*.csv")
log_message(f"Combined dataset shape (Polars): {combined_pl.shape}")

# ---------------------------
# 2. APPLY MAD FILTER
# ---------------------------
log_message("Filtering numeric features based on MAD (threshold=5) via src.mad_filter...")
final_pl, features_to_keep = mad_feature_filter(
    data=combined_pl, 
    label_col="Label", 
    mad_threshold=5
)
log_message(f"Number of numeric features kept after MAD filtering: {len(features_to_keep)}")

# Convert Polars to pandas
final_df = final_pl.to_pandas()

# Strip "species" from the label string if needed
final_df["Label"] = final_df["Label"].str.replace("species", "", regex=False)

# ---------------------------
# 3. PREPARE FEATURES & LABELS (All Numeric)
# ---------------------------
# Drop the label column to get X, and keep the label in y
X = final_df.drop(columns=["Label"]).values
y = final_df["Label"].values

# Encode the labels (but not the features!)
main_label_encoder = LabelEncoder()
y_encoded = main_label_encoder.fit_transform(y)

# ---------------------------
# 3.1 TRAIN/TEST SPLIT
# ---------------------------
log_message("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Optionally save data (e.g., for calibration)
np.savez("data_for_calibration.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

log_message(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# ---------------------------
# 3.2 OPTIONAL SCALING FOR NUMERIC FEATURES
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Because the TabTransformerClassifierWithVal expects two separate inputs
# (X_categ, X_num), and we know everything is numeric, we pass an empty array
# for X_categ, and our scaled numeric data for X_num.
X_categ_train = np.empty((X_train.shape[0], 0))  # shape=(n, 0)
X_num_train   = X_train
X_categ_test  = np.empty((X_test.shape[0], 0))
X_num_test    = X_test

# ---------------------------
# 4. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ---------------------------
log_message("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, X_categ_train, X_num_train, y_train), n_trials=20)
best_params = study.best_params
log_message(f"Best hyperparameters found: {best_params}")

# ---------------------------
# 5. FINAL TRAINING WITH BEST HYPERPARAMETERS
# ---------------------------
final_clf = TabTransformerClassifierWithVal(
    transformer_dim=best_params["transformer_dim"],
    depth=best_params["depth"],
    heads=best_params["heads"],
    dim_forward=best_params["dim_forward"],
    dropout=best_params["dropout"],
    mlp_hidden_dims=[best_params["mlp_hidden_dim1"], best_params["mlp_hidden_dim2"]],
    learning_rate=best_params["learning_rate"],
    batch_size=best_params["batch_size"],
    epochs=100,  # Extended training
    verbose=True
)
log_message("Fitting final TabTransformer model with optimized hyperparameters (100 epochs)...")
final_clf.fit(X_categ_train, X_num_train, y_train)

# ---------------------------
# 6. 5-FOLD CROSS-VALIDATION FOR UNCERTAINTY ESTIMATES
# ---------------------------
log_message("Performing 5-fold CV with optimized hyperparameters for uncertainty estimates...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_train_losses = []
fold_val_losses = []
fold_val_accuracies = []

fold_idx = 1
for train_index, val_index in kf.split(X_categ_train, y_train):
    X_cat_tr_fold, X_cat_val_fold = X_categ_train[train_index], X_categ_train[val_index]
    X_num_tr_fold, X_num_val_fold = X_num_train[train_index], X_num_train[val_index]
    y_tr_fold, y_val_fold = y_train[train_index], y_train[val_index]

    clf_fold = TabTransformerClassifierWithVal(
        transformer_dim=best_params["transformer_dim"],
        depth=best_params["depth"],
        heads=best_params["heads"],
        dim_forward=best_params["dim_forward"],
        dropout=best_params["dropout"],
        mlp_hidden_dims=[best_params["mlp_hidden_dim1"], best_params["mlp_hidden_dim2"]],
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        epochs=100,
        verbose=False
    )
    clf_fold.fit(
        X_cat_tr_fold, X_num_tr_fold, y_tr_fold,
        X_cat_val_fold, X_num_val_fold, y_val_fold
    )

    tr_losses, val_losses = clf_fold.get_train_val_losses()
    fold_train_losses.append(tr_losses)
    fold_val_losses.append(val_losses)

    y_val_pred = clf_fold.predict(X_cat_val_fold, X_num_val_fold)
    val_acc = accuracy_score(y_val_fold, y_val_pred)
    fold_val_accuracies.append(val_acc)
    log_message(f"Fold {fold_idx} complete. Validation Accuracy: {val_acc:.4f}")
    fold_idx += 1

mean_cv_val_acc = np.mean(fold_val_accuracies)
std_cv_val_acc  = np.std(fold_val_accuracies)
log_message(f"5-Fold CV Validation Accuracy: Mean = {mean_cv_val_acc:.4f}, Std = {std_cv_val_acc:.4f}")

# ---------------------------
# 6.1 PLOT TRAIN & VAL LOSSES PER FOLD
# ---------------------------
epochs_range = np.arange(1, final_clf.epochs + 1)
fold_train_losses_arr = np.array(fold_train_losses)
fold_val_losses_arr   = np.array(fold_val_losses)

plt.figure(figsize=(10, 6))
for i in range(len(fold_train_losses)):
    plt.plot(epochs_range, fold_train_losses[i],
             label=f"Train Loss (Fold {i+1})", alpha=0.6)
    plt.plot(epochs_range, fold_val_losses[i],
             label=f"Val Loss (Fold {i+1})", alpha=0.6, linestyle='--')
plt.title("Train & Validation Loss Curves (All Folds)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("tabtransformer_cv_all_folds_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved per-fold train/val loss plot to 'tabtransformer_cv_all_folds_loss.png'.")

mean_train = fold_train_losses_arr.mean(axis=0)
std_train  = fold_train_losses_arr.std(axis=0)
mean_val   = fold_val_losses_arr.mean(axis=0)
std_val    = fold_val_losses_arr.std(axis=0)

plt.figure(figsize=(8, 5))
plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train,
                 alpha=0.2, label="Train ±1 Std")
plt.plot(epochs_range, mean_train, label="Mean Train Loss")
plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val,
                 alpha=0.2, label="Val ±1 Std")
plt.plot(epochs_range, mean_val, label="Mean Val Loss")
plt.title("Aggregated 5-Fold CV Loss (Mean ± Std)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("tabtransformer_cv_mean_confidence_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved mean ± std train/val loss plot to 'tabtransformer_cv_mean_confidence_loss.png'.")

# ---------------------------
# 7. FINAL EVALUATION ON HELD-OUT TEST SET
# ---------------------------
log_message("Evaluating final model on the test set...")
y_test_pred = final_clf.predict(X_categ_test, X_num_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
log_message(f"Final Model Accuracy on Test Set: {test_accuracy:.4f}")

class_report = classification_report(y_test, y_test_pred, target_names=main_label_encoder.classes_)
log_message("\nClassification Report:")
log_message(class_report)

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=main_label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (TabTransformer)")
plt.tight_layout()
plt.savefig("confusion_matrix_tabtransformer.png", bbox_inches='tight')
plt.close()
log_message("Saved confusion matrix to 'confusion_matrix_tabtransformer.png'.")

# ---------------------------
# 8. SAVE RESULTS & LOGS
# ---------------------------
metrics_dict = {
    "test_accuracy": float(test_accuracy),
    "classification_report": class_report,
    "cv_val_accuracies": [float(acc) for acc in fold_val_accuracies],
    "cv_val_accuracy_mean": float(mean_cv_val_acc),
    "cv_val_accuracy_std": float(std_cv_val_acc)
}
with open("metrics_tabtransformer.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
log_message("Saved metrics to 'metrics_tabtransformer.json'.")

np.save("confusion_matrix_tabtransformer.npy", cm)
torch.save(final_clf.model_.state_dict(), "tabtransformer_model_state.pth")
log_message("Saved TabTransformer model's state_dict to 'tabtransformer_model_state.pth'.")

# Example of saving label encoder (for y) if needed:
import pickle
with open("main_label_encoder.pkl", "wb") as f:
    pickle.dump(main_label_encoder, f)

with open("log_steps_tabtransformer.json", "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log to 'log_steps_tabtransformer.json'.")

# ---------------------------
# 9. SAVE ADDITIONAL ARTIFACTS FOR PLOT REPLICATION
# ---------------------------
np.savez("plot_data_artifacts.npz",
         epochs_range=epochs_range,
         fold_train_losses_arr=fold_train_losses_arr,
         fold_val_losses_arr=fold_val_losses_arr,
         mean_train=mean_train,
         std_train=std_train,
         mean_val=mean_val,
         std_val=std_val)
log_message("Saved additional plot data to 'plot_data_artifacts.npz'.")

log_message("All done!")
