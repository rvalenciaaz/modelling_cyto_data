import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import json

# ---------------------------
# 1. Load CV Plot Data
# ---------------------------
data = np.load("cv_plot_data.npz", allow_pickle=True)
epochs_range    = data["epochs_range"]
fold_train_losses = data["fold_train_losses"]
fold_val_losses   = data["fold_val_losses"]
mean_train      = data["mean_train"]
std_train       = data["std_train"]
mean_val        = data["mean_val"]
std_val         = data["std_val"]
fold_val_accuracies = data["fold_val_accuracies"]

# ---------------------------
# 2. Plot Train & Validation Loss Curves (All Folds)
# ---------------------------
plt.figure(figsize=(10, 6))
num_folds = len(fold_train_losses)
for i in range(num_folds):
    plt.plot(epochs_range, fold_train_losses[i],
             label=f"Train Loss (Fold {i+1})", alpha=0.6)
    plt.plot(epochs_range, fold_val_losses[i],
             label=f"Val Loss (Fold {i+1})", alpha=0.6, linestyle='--')
plt.title("Train & Validation Loss Curves (All Folds)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("reproduced_cv_all_folds_loss.png", bbox_inches='tight')
plt.show()

# ---------------------------
# 3. Plot Mean ± 1 Std Train/Val Loss (5-fold CV)
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, mean_train, label="Mean Train Loss", color='blue')
plt.fill_between(epochs_range,
                 mean_train - std_train,
                 mean_train + std_train,
                 alpha=0.2, color='blue')
plt.plot(epochs_range, mean_val, label="Mean Val Loss", color='orange')
plt.fill_between(epochs_range,
                 mean_val - std_val,
                 mean_val + std_val,
                 alpha=0.2, color='orange')
plt.title("Mean ± 1 Std Train/Val Loss (5-fold CV)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("reproduced_cv_mean_confidence_loss.png", bbox_inches='tight')
plt.show()

# ---------------------------
# 4. Reproduce the Confusion Matrix Plot
# ---------------------------
# Load the saved confusion matrix array
cm = np.load("confusion_matrix.npy")

# If you saved the label names somewhere (e.g., in metrics.json), load them.
# Here we attempt to load from metrics.json; otherwise, generic labels will be used.
try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    # If label names were saved, you could retrieve them. For this example, we assume they are stored as:
    # metrics["label_names"] = ["class0", "class1", ...]
    label_names = metrics.get("label_names", None)
except Exception as e:
    print("Could not load label names from metrics.json, using numeric labels instead.")
    label_names = None

plt.figure(figsize=(8, 6))
if label_names:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_names)
    disp.plot(cmap='Blues', values_format='d')
else:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Reproduced)")
plt.tight_layout()
plt.savefig("reproduced_confusion_matrix.png", bbox_inches='tight')
plt.show()
