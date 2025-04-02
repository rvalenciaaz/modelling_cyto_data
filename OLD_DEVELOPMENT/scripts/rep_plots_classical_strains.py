import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science","nature"])
# Set the folder where the saved files reside
output_folder = "output_classical_100k"

# Dictionary for strain names
strains_dict = {
    "1": "Anaerostipes caccae",
    "2": "Bacteroides thetaiotaomicron",
    "3": "Blautia producta",
    "4": "Lactobacillus plantarum",
    "5": "Escherichia coli",
    "6": "Clostridium butyricum",
    "7": "Erysipelatoclostridium ramosum",
    "8": "Bifidobacterium longum",
    "9": "Akkermansia muciniphila"
}

def file_path(filename):
    return os.path.join(output_folder, filename)

def rename_labels(labels):
    return [f"{strains_dict.get(str(label), label)} ({label})" for label in labels]

def plot_confusion_matrix(csv_file, title, save_path, fmt="d",
                            xlabel="Predicted Label", ylabel="True Label"):
    """
    Load a confusion matrix from a CSV file and plot it using seaborn heatmap.
    """
    cm = pd.read_csv(csv_file, index_col=0)
    cm.index = rename_labels(cm.index)
    cm.columns = rename_labels(cm.columns)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=fmt,
                xticklabels=cm.columns, yticklabels=cm.index)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(csv_file, title_top10, title_top20,
                            save_path_top10, save_path_top20, save_path_all):
    """
    Load the full feature importance data from a CSV file and plot the top 10, top 20, and all features.
    """
    fi = pd.read_csv(csv_file)
    fi_sorted = fi.sort_values("Importance", ascending=False)

    # Plot top 10 features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_sorted.head(10), x="Importance", y="Feature", color="skyblue")
    plt.title(title_top10)
    plt.tight_layout()
    plt.savefig(save_path_top10)
    plt.close()

    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_sorted.head(20), x="Importance", y="Feature", color="skyblue")
    plt.title(title_top20)
    plt.tight_layout()
    plt.savefig(save_path_top20)
    plt.close()

    # Plot all features with smaller font size for better visibility
    plt.figure(figsize=(12, 10))
    sns.barplot(data=fi_sorted, x="Importance", y="Feature", color="skyblue")
    plt.title("Feature Importances (All Features)")
    plt.yticks(fontsize=8)  # Reduce label size for better visibility
    plt.tight_layout()
    plt.savefig(save_path_all)
    plt.close()

# --- Replicate Confusion Matrix Plots ---

# Random Forest Confusion Matrices
plot_confusion_matrix(file_path("confusion_matrix_rf_data.csv"),
                      "Confusion Matrix (Random Forest)",
                      file_path("replicated_confusion_matrix_rf.png"),
                      fmt="d")

plot_confusion_matrix(file_path("confusion_matrix_rf_normalized_data.csv"),
                      "Normalized Confusion Matrix (Random Forest)",
                      file_path("replicated_confusion_matrix_rf_normalized.png"),
                      fmt=".2f")

# Logistic Regression Confusion Matrices
plot_confusion_matrix(file_path("confusion_matrix_lr_data.csv"),
                      "Confusion Matrix (Logistic Regression)",
                      file_path("replicated_confusion_matrix_lr.png"),
                      fmt="d")

plot_confusion_matrix(file_path("confusion_matrix_lr_normalized_data.csv"),
                      "Normalized Confusion Matrix (Logistic Regression)",
                      file_path("replicated_confusion_matrix_lr_normalized.png"),
                      fmt=".2f")

# XGBoost Confusion Matrices
plot_confusion_matrix(file_path("confusion_matrix_xgb_data.csv"),
                      "Confusion Matrix (XGBoost)",
                      file_path("replicated_confusion_matrix_xgb.png"),
                      fmt="d")

plot_confusion_matrix(file_path("confusion_matrix_xgb_normalized_data.csv"),
                      "Normalized Confusion Matrix (XGBoost)",
                      file_path("replicated_confusion_matrix_xgb_normalized.png"),
                      fmt=".2f")

# --- Replicate Feature Importance Plots ---

# Random Forest Feature Importances
plot_feature_importance(file_path("feature_importances_rf_full.csv"),
                        "Random Forest Feature Importances (Top 10)",
                        "Random Forest Feature Importances (Top 20)",
                        file_path("replicated_feature_importances_rf_top10.png"),
                        file_path("replicated_feature_importances_rf_top20.png"),
                        file_path("replicated_feature_importances_rf_all.png"))

# XGBoost Feature Importances
plot_feature_importance(file_path("feature_importances_xgb_full.csv"),
                        "XGBoost Feature Importances (Top 10)",
                        "XGBoost Feature Importances (Top 20)",
                        file_path("replicated_feature_importances_xgb_top10.png"),
                        file_path("replicated_feature_importances_xgb_top20.png"),
                        file_path("replicated_feature_importances_xgb_all.png"))

print("All plots have been replicated and saved successfully in the output folder.")
