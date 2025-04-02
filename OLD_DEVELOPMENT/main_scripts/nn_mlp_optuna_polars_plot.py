# replicate_plots.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def replicate_plots():
    # -----------------------------
    # 1. REPLICATE CV LOSS CURVES
    # -----------------------------
    data = np.load("cv_plot_data.npz")
    epochs_range = data["epochs_range"]
    fold_train_losses = data["fold_train_losses"]
    fold_val_losses = data["fold_val_losses"]
    mean_train = data["mean_train"]
    std_train = data["std_train"]
    mean_val = data["mean_val"]
    std_val = data["std_val"]

    # (A) Plot training/validation loss for each fold
    plt.figure(figsize=(10, 6))
    for i in range(len(fold_train_losses)):
        plt.plot(epochs_range, fold_train_losses[i], alpha=0.6, label=f"Train Loss (Fold {i+1})")
        plt.plot(epochs_range, fold_val_losses[i], alpha=0.6, linestyle="--", label=f"Val Loss (Fold {i+1})")
    plt.title("Train & Validation Loss Curves (All Folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("cv_all_folds_loss_replicated.png", bbox_inches='tight')
    plt.close()

    # (B) Plot mean ± std for train/val loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, mean_train, label="Mean Train Loss")
    plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train, alpha=0.2)
    plt.plot(epochs_range, mean_val, label="Mean Val Loss")
    plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val, alpha=0.2)
    plt.title("Mean ± 1 Std Train/Val Loss (5-fold CV)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cv_mean_confidence_loss_replicated.png", bbox_inches='tight')
    plt.close()

    # -----------------------------
    # 2. REPLICATE CONFUSION MATRIX
    # -----------------------------
    cm = np.load("confusion_matrix.npy")

    # If you don't remember which labels you had, just do a generic plot:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Replicated)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_replicated.png", bbox_inches='tight')
    plt.close()

    print("Plots replicated and saved!")

if __name__ == "__main__":
    replicate_plots()
