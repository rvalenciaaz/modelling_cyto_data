import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import median_abs_deviation
import scienceplots

plt.style.use(['science','ieee'])
# ------------------------------
# 1. LOAD SAVED ARTIFACTS
# ------------------------------
print("Loading saved models and preprocessing objects...")
best_rf = joblib.load("best_rf_model.pkl")
best_lr = joblib.load("best_lr_model.pkl")
best_xgb = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load list of features that passed MAD filtering
features_df = pd.read_csv("features_to_keep.csv")
features_to_keep = features_df["Feature"].tolist()

# ------------------------------
# 2. RECONSTRUCT THE DATASET & TEST SPLIT
# ------------------------------
print("Reconstructing dataset...")
csv_files = glob.glob("species*.csv")
df_list = []
for file_path in csv_files:
    temp_df = pd.read_csv(file_path)
    # Create a label from the filename (e.g. "species1.csv" -> "1")
    label = file_path.split('.')[0]
    temp_df['Label'] = label
    df_list.append(temp_df)
    
combined_df = pd.concat(df_list, ignore_index=True)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

# Keep only the features that passed filtering plus the target label
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()

# Separate features and target
X = final_df.drop(columns=["Label"])
y = final_df["Label"]

# Transform labels using the loaded label encoder
y_encoded = label_encoder.transform(y)

# Reproduce the same train/test split (test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Use the saved scaler to transform the test set
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 3. DEFINE HELPER FUNCTIONS FOR PLOTTING
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Plot standard confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.savefig(filename)
    plt.close()
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                fmt=".2f")
    plt.title(f"Normalized Confusion Matrix ({model_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    norm_filename = filename.replace(".png", "_normalized.png")
    plt.savefig(norm_filename)
    plt.close()

def plot_feature_importances(model, model_name, csv_filename, top_n=10, prefix="reproduced"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        fi_df.sort_values("Importance", ascending=False, inplace=True)
        
        # Save the computed importances
        fi_df.to_csv(csv_filename, index=False)
        
        # Plot top_n features
        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi_df.head(top_n), x="Importance", y="Feature", color='skyblue')
        plt.title(f"{model_name} Feature Importances (Top {top_n})")
        plt.tight_layout()
        plt.savefig(f"{prefix}_{model_name.lower()}_feature_importances_top{top_n}.png")
        plt.close()

def plot_cv_results(csv_file, model_name, filename):
    cv_df = pd.read_csv(csv_file)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=cv_df, x="Fold", y="Accuracy", palette="viridis")
    plt.title(f"{model_name} 5-Fold CV Accuracies")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ------------------------------
# 4. REPRODUCE CONFUSION MATRIX PLOTS
# ------------------------------
print("Reproducing confusion matrix plots...")

# Random Forest
y_pred_rf = best_rf.predict(X_test_scaled)
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest", "reproduced_confusion_matrix_rf.png")

# Logistic Regression
y_pred_lr = best_lr.predict(X_test_scaled)
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression", "reproduced_confusion_matrix_lr.png")

# XGBoost
y_pred_xgb = best_xgb.predict(X_test_scaled)
plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost", "reproduced_confusion_matrix_xgb.png")

# ------------------------------
# 5. REPRODUCE FEATURE IMPORTANCE PLOTS
# ------------------------------
print("Reproducing feature importance plots...")
plot_feature_importances(best_rf, "Random Forest", "reproduced_feature_importances_rf.csv", top_n=10)
plot_feature_importances(best_rf, "Random Forest", "reproduced_feature_importances_rf.csv", top_n=20)
plot_feature_importances(best_xgb, "XGBoost", "reproduced_feature_importances_xgb.csv", top_n=10)
plot_feature_importances(best_xgb, "XGBoost", "reproduced_feature_importances_xgb.csv", top_n=20)

# ------------------------------
# 6. REPRODUCE CROSS-VALIDATION FOLD ACCURACY PLOTS
# ------------------------------
print("Reproducing cross-validation fold accuracy plots...")
plot_cv_results("cv_results_rf.csv", "Random Forest", "reproduced_cv_results_rf.png")
plot_cv_results("cv_results_lr.csv", "Logistic Regression", "reproduced_cv_results_lr.png")
plot_cv_results("cv_results_xgb.csv", "XGBoost", "reproduced_cv_results_xgb.png")

# ------------------------------
# 7. REPRODUCE AGGREGATE METRICS PLOT
# ------------------------------
print("Reproducing aggregate model metrics plot...")
metrics_df = pd.read_csv("classification_metrics_all_models.csv")
plt.figure(figsize=(8, 6))
sns.barplot(data=metrics_df, x="Model", y="Accuracy", palette="magma")
plt.title("Aggregate Model Accuracies")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("reproduced_aggregate_model_metrics.png")
plt.close()

print("All plots have been reproduced and saved.")
