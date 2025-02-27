import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Set a consistent style
sns.set_theme(style="whitegrid")

# ---------------------------
# Define folder paths
# ---------------------------
classical_dir = "output_classical_100k"
nn_dir = "output_nn_100k"

# ---------------------------
# PART 1: Classical Model Plots
# ---------------------------
print("Reproducing classical model plots...")

# 1. Combined Model Accuracy Plot
metrics_file = os.path.join(classical_dir, "classification_metrics_all_models.csv")
if os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=metrics_df, x="Model", y="Accuracy", palette="viridis")
    plt.title("Combined Model Accuracies")
    plt.ylim(0, 1)
    # Annotate each bar with its accuracy value
    for i, row in metrics_df.iterrows():
        ax.text(i, row['Accuracy'] + 0.02, f"{row['Accuracy']:.2f}", ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(classical_dir, "reproduced_combined_model_accuracies.png"))
    plt.close()
else:
    print(f"File not found: {metrics_file}")

# 2. Feature Importances: Random Forest
fi_rf_file = os.path.join(classical_dir, "feature_importances_rf.csv")
if os.path.exists(fi_rf_file):
    fi_rf = pd.read_csv(fi_rf_file)
    fi_rf.sort_values("Importance", ascending=False, inplace=True)

    # Top 10
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_rf.head(10), x="Importance", y="Feature", color='skyblue')
    plt.title("Random Forest Feature Importances (Top 10)")
    plt.tight_layout()
    plt.savefig(os.path.join(classical_dir, "reproduced_feature_importances_rf_top10.png"))
    plt.close()

    # Top 20
    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_rf.head(20), x="Importance", y="Feature", color='skyblue')
    plt.title("Random Forest Feature Importances (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(classical_dir, "reproduced_feature_importances_rf_top20.png"))
    plt.close()
else:
    print(f"File not found: {fi_rf_file}")

# 3. Feature Importances: XGBoost
fi_xgb_file = os.path.join(classical_dir, "feature_importances_xgb.csv")
if os.path.exists(fi_xgb_file):
    fi_xgb = pd.read_csv(fi_xgb_file)
    fi_xgb.sort_values("Importance", ascending=False, inplace=True)

    # Top 10
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_xgb.head(10), x="Importance", y="Feature", color='skyblue')
    plt.title("XGBoost Feature Importances (Top 10)")
    plt.tight_layout()
    plt.savefig(os.path.join(classical_dir, "reproduced_feature_importances_xgb_top10.png"))
    plt.close()

    # Top 20
    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_xgb.head(20), x="Importance", y="Feature", color='skyblue')
    plt.title("XGBoost Feature Importances (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(classical_dir, "reproduced_feature_importances_xgb_top20.png"))
    plt.close()
else:
    print(f"File not found: {fi_xgb_file}")

# 4. Optional: 5-Fold CV Results for RF, LR, XGBoost
cv_files = {
    "rf": "cv_results_rf.csv",
    "lr": "cv_results_lr.csv",
    "xgb": "cv_results_xgb.csv"
}
for model, filename in cv_files.items():
    file_path = os.path.join(classical_dir, filename)
    if os.path.exists(file_path):
        cv_df = pd.read_csv(file_path)
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=cv_df, x="Fold", y="Accuracy", palette="muted")
        plt.title(f"{model.upper()} 5-Fold CV Accuracies")
        plt.ylim(0, 1)
        for i, row in cv_df.iterrows():
            ax.text(i, row['Accuracy'] + 0.02, f"{row['Accuracy']:.2f}", ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(classical_dir, f"reproduced_cv_results_{model}.png"))
        plt.close()
    else:
        print(f"File not found: {file_path}")

# ---------------------------
# PART 2: PyTorch NN Plots
# ---------------------------
print("Reproducing PyTorch NN plots...")

# 1. CV Plot Data (Train/Val Loss Curves per Fold)
cv_plot_data_file = os.path.join(nn_dir, "cv_plot_data.npz")
if os.path.exists(cv_plot_data_file):
    data = np.load(cv_plot_data_file, allow_pickle=True)
    epochs_range = data["epochs_range"]
    fold_train_losses = data["fold_train_losses"]
    fold_val_losses = data["fold_val_losses"]

    plt.figure(figsize=(10, 6))
    num_folds = fold_train_losses.shape[0]
    for i in range(num_folds):
        plt.plot(epochs_range, fold_train_losses[i], label=f"Train Loss (Fold {i+1})", alpha=0.6)
        plt.plot(epochs_range, fold_val_losses[i], label=f"Val Loss (Fold {i+1})", alpha=0.6, linestyle='--')
    plt.title("Train & Validation Loss Curves (All Folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(nn_dir, "reproduced_cv_all_folds_loss.png"), bbox_inches='tight')
    plt.close()
else:
    print(f"File not found: {cv_plot_data_file}")

# 2. Mean ± 1 Std Train/Val Loss Plot (from CV)
if os.path.exists(cv_plot_data_file):
    mean_train = data["mean_train"]
    std_train  = data["std_train"]
    mean_val   = data["mean_val"]
    std_val    = data["std_val"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, mean_train, label="Mean Train Loss", color='blue')
    plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train,
                     alpha=0.2, color='blue')
    plt.plot(epochs_range, mean_val, label="Mean Val Loss", color='orange')
    plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val,
                     alpha=0.2, color='orange')
    plt.title("Mean ± 1 Std Train/Val Loss (5-Fold CV)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(nn_dir, "reproduced_cv_mean_confidence_loss.png"), bbox_inches='tight')
    plt.close()
else:
    print(f"File not found: {cv_plot_data_file}")

# 3. Confusion Matrix Plot for Final NN Model
cm_file = os.path.join(nn_dir, "confusion_matrix.npy")
if os.path.exists(cm_file):
    cm = np.load(cm_file)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Final Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(nn_dir, "reproduced_confusion_matrix_nn.png"), bbox_inches='tight')
    plt.close()
else:
    print(f"File not found: {cm_file}")

print("All available plots have been reproduced and saved as PNG files.")
