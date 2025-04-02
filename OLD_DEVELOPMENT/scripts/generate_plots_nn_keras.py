import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from sklearn.metrics import ConfusionMatrixDisplay

# Use the same plotting style as before
plt.style.use(['science', 'nature'])

# -------------------------------
# Reproduce CV Loss Plots
# -------------------------------
# Load the cross-validation plot data saved in cv_plot_data.npz
data = np.load("cv_plot_data.npz")
epochs_range   = data['epochs_range']
fold_train_losses = data['fold_train_losses']
fold_val_losses   = data['fold_val_losses']
mean_train     = data['mean_train']
std_train      = data['std_train']
mean_val       = data['mean_val']
std_val        = data['std_val']

# Plot train & validation loss for each fold
plt.figure(figsize=(10, 6))
num_folds = fold_train_losses.shape[0]
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
plt.savefig("cv_all_folds_loss_reproduced.png", bbox_inches='tight')
plt.close()
print("Saved 'cv_all_folds_loss_reproduced.png'.")

# Plot mean ± 1 Std of train/validation loss across folds
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, mean_train, label="Mean Train Loss", color='blue')
plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train,
                 alpha=0.2, color='blue')
plt.plot(epochs_range, mean_val, label="Mean Val Loss", color='orange')
plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val,
                 alpha=0.2, color='orange')
plt.title("Mean ± 1 Std Train/Val Loss (5-fold CV)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("cv_mean_confidence_loss_reproduced.png", bbox_inches='tight')
plt.close()
print("Saved 'cv_mean_confidence_loss_reproduced.png'.")

# -------------------------------
# Reproduce Confusion Matrix Plot
# -------------------------------
# Load the confusion matrix saved earlier
cm = np.load("confusion_matrix.npy")

# Plot the confusion matrix using sklearn's ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Final Model)")
plt.tight_layout()
plt.savefig("confusion_matrix_reproduced.png", bbox_inches='tight')
plt.close()
print("Saved 'confusion_matrix_reproduced.png'.")
