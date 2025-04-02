import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

# For modeling and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================
# 1. READING & SUBSAMPLING CSV FILES
# ====================================================
print("Reading CSV files and subsampling...")

csv_files = glob.glob("species*.csv")
df_list = []

for file_path in csv_files:
    temp_df = pd.read_csv(file_path)

    # Subsample up to 10,000 rows
    #temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)

    # Create a label from the filename, e.g. "species1.csv" -> "species1"
    label = file_path.split('.')[0]
    temp_df['Label'] = label

    df_list.append(temp_df)

combined_df = pd.concat(df_list, ignore_index=True)

# Clean up the 'Label' column (remove "species" prefix)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

print(f"Combined dataset shape: {combined_df.shape}")

# ====================================================
# 2. FILTERING FEATURES BASED ON MAD
# ====================================================
print("Filtering numeric features based on MAD...")

# Select numeric columns
numerical_data = combined_df.select_dtypes(include=[np.number])

# Calculate CV & MAD for each numeric column
cv_results = {}
for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val = numerical_data[col].std()

    # Coefficient of Variation (CV)
    cv = std_val / mean_val if mean_val != 0 else np.nan
    mad = median_abs_deviation(numerical_data[col].values, scale='normal')

    cv_results[col] = [col, cv, mad]

cv_df = pd.DataFrame(cv_results.values(), columns=['Feature', 'CV', 'MAD'])

# Threshold for MAD
MAD_THRESHOLD = 5
features_to_keep = cv_df.loc[cv_df["MAD"] >= MAD_THRESHOLD, "Feature"].tolist()

# Final DataFrame with columns passing MAD threshold plus the 'Label'
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()

print(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ====================================================
# 3. FEATURE SCALING & TRAIN/TEST SPLIT
# ====================================================
print("Splitting into train/test and scaling features...")

X = final_df.drop(columns=["Label"])
y = final_df["Label"]

# Convert labels to numeric if they are strings
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Data split complete.")
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ====================================================
# 4. HYPERPARAMETER GRIDS
# ====================================================
# Define parameter grids for each model
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

param_grid_lr = {
    "C": [0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]  # 'liblinear' supports l1 and l2
}

param_grid_xgb = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}

# ====================================================
# 5. RANDOM FOREST + GRID SEARCH
# ====================================================
print("\n=== Random Forest: Grid Search ===")
rf_clf = RandomForestClassifier(random_state=42)

grid_rf = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid_rf,
    cv=3,                # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,           # use all available cores
    verbose=1            # print intermediate steps
)

print("Starting GridSearchCV for Random Forest...")
grid_rf.fit(X_train_scaled, y_train)

print("Random Forest Grid Search complete.")
print(f"Best Random Forest Parameters: {grid_rf.best_params_}")
best_rf = grid_rf.best_estimator_

print("Re-fitting best Random Forest model on the entire training set...")
best_rf.fit(X_train_scaled, y_train)

# Evaluate
print("Evaluating Random Forest...")
y_pred_rf = best_rf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_rf:.4f}")

report_dict_rf = classification_report(
    y_test, 
    y_pred_rf, 
    target_names=label_encoder.classes_, 
    output_dict=True
)
report_df_rf = pd.DataFrame(report_dict_rf).transpose()
print("\nClassification Report (RF):")
print(report_df_rf)

report_df_rf.to_csv("classification_report_rf.csv", index=True)

# Save accuracy in a metrics CSV
metrics_df_rf = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy_rf]
})
metrics_df_rf.to_csv("classification_metrics_rf.csv", index=False)

# Confusion Matrix (Random Forest)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_rf_norm = confusion_matrix(y_test, y_pred_rf, normalize='true')

disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=label_encoder.classes_)
disp_rf.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Random Forest)")
plt.savefig("confusion_matrix_rf.png")
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(cm_rf_norm, annot=True, cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            fmt=".2f")
plt.title("Normalized Confusion Matrix (Random Forest)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix_rf_normalized.png")
plt.close()

# Feature Importances (Random Forest)
importances_rf = best_rf.feature_importances_
feature_names = X.columns
fi_df_rf = pd.DataFrame({"Feature": feature_names, "Importance": importances_rf})
fi_df_rf.sort_values("Importance", ascending=False, inplace=True)

print("\nFeature Importances (Random Forest) - top 10 shown:")
print(fi_df_rf.head(10))
fi_df_rf.to_csv("feature_importances_rf.csv", index=False)

# Plot top 10
plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df_rf.head(10), x="Importance", y="Feature", color='skyblue')
plt.title("Random Forest Feature Importances (Top 10)")
plt.tight_layout()
plt.savefig("feature_importances_rf_top10.png")
plt.close()

# NEW: Plot top 20 features
plt.figure(figsize=(10, 8))
sns.barplot(data=fi_df_rf.head(20), x="Importance", y="Feature", color='skyblue')
plt.title("Random Forest Feature Importances (Top 20)")
plt.tight_layout()
plt.savefig("feature_importances_rf_top20.png")
plt.close()

# ====================================================
# 6. LOGISTIC REGRESSION + GRID SEARCH
# ====================================================
print("\n=== Logistic Regression: Grid Search ===")
lr_clf = LogisticRegression(random_state=42, max_iter=1000)

grid_lr = GridSearchCV(
    estimator=lr_clf,
    param_grid=param_grid_lr,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV for Logistic Regression...")
grid_lr.fit(X_train_scaled, y_train)

print("Logistic Regression Grid Search complete.")
print(f"Best Logistic Regression Parameters: {grid_lr.best_params_}")
best_lr = grid_lr.best_estimator_

print("Re-fitting best Logistic Regression model on the entire training set...")
best_lr.fit(X_train_scaled, y_train)

# Evaluate
print("Evaluating Logistic Regression...")
y_pred_lr = best_lr.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("\n=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy_lr:.4f}")

report_dict_lr = classification_report(
    y_test, 
    y_pred_lr, 
    target_names=label_encoder.classes_, 
    output_dict=True
)
report_df_lr = pd.DataFrame(report_dict_lr).transpose()
print("\nClassification Report (Logistic Regression):")
print(report_df_lr)

report_df_lr.to_csv("classification_report_lr.csv", index=True)

# Save LR accuracy
metrics_df_lr = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "Accuracy": [accuracy_lr]
})
metrics_df_lr.to_csv("classification_metrics_lr.csv", index=False)

# Confusion Matrix (Logistic Regression)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_lr_norm = confusion_matrix(y_test, y_pred_lr, normalize='true')

disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=label_encoder.classes_)
disp_lr.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Logistic Regression)")
plt.savefig("confusion_matrix_lr.png")
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(cm_lr_norm, annot=True, cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            fmt=".2f")
plt.title("Normalized Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix_lr_normalized.png")
plt.close()

# ====================================================
# 7. XGBOOST + GRID SEARCH
# ====================================================
print("\n=== XGBoost: Grid Search ===")
xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

grid_xgb = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid_xgb,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV for XGBoost...")
grid_xgb.fit(X_train_scaled, y_train)

print("XGBoost Grid Search complete.")
print(f"Best XGBoost Parameters: {grid_xgb.best_params_}")
best_xgb = grid_xgb.best_estimator_

print("Re-fitting best XGBoost model on the entire training set...")
best_xgb.fit(X_train_scaled, y_train)

# Evaluate
print("Evaluating XGBoost...")
y_pred_xgb = best_xgb.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print("\n=== XGBoost Results ===")
print(f"Accuracy: {accuracy_xgb:.4f}")

report_dict_xgb = classification_report(
    y_test, 
    y_pred_xgb, 
    target_names=label_encoder.classes_, 
    output_dict=True
)
report_df_xgb = pd.DataFrame(report_dict_xgb).transpose()
print("\nClassification Report (XGBoost):")
print(report_df_xgb)

report_df_xgb.to_csv("classification_report_xgb.csv", index=True)

# Save XGB accuracy
metrics_df_xgb = pd.DataFrame({
    "Model": ["XGBoost"],
    "Accuracy": [accuracy_xgb]
})
metrics_df_xgb.to_csv("classification_metrics_xgb.csv", index=False)

# Confusion Matrix (XGBoost)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_xgb_norm = confusion_matrix(y_test, y_pred_xgb, normalize='true')

disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=label_encoder.classes_)
disp_xgb.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (XGBoost)")
plt.savefig("confusion_matrix_xgb.png")
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(cm_xgb_norm, annot=True, cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            fmt=".2f")
plt.title("Normalized Confusion Matrix (XGBoost)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix_xgb_normalized.png")
plt.close()

# Feature Importances (XGBoost)
importances_xgb = best_xgb.feature_importances_
fi_df_xgb = pd.DataFrame({"Feature": X.columns, "Importance": importances_xgb})
fi_df_xgb.sort_values("Importance", ascending=False, inplace=True)

print("\nFeature Importances (XGBoost) - top 10 shown:")
print(fi_df_xgb.head(10))
fi_df_xgb.to_csv("feature_importances_xgb.csv", index=False)

# Plot top 10
plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df_xgb.head(10), x="Importance", y="Feature", color='skyblue')
plt.title("XGBoost Feature Importances (Top 10)")
plt.tight_layout()
plt.savefig("feature_importances_xgb_top10.png")
plt.close()

# NEW: Plot top 20 features
plt.figure(figsize=(10, 8))
sns.barplot(data=fi_df_xgb.head(20), x="Importance", y="Feature", color='skyblue')
plt.title("XGBoost Feature Importances (Top 20)")
plt.tight_layout()
plt.savefig("feature_importances_xgb_top20.png")
plt.close()

# ====================================================
# 8. AGGREGATE METRICS FOR ALL MODELS
# ====================================================
print("\nGathering metrics for all models...")

metrics_combined = pd.concat([
    metrics_df_rf,
    metrics_df_lr,
    metrics_df_xgb
], ignore_index=True)

print("\n=== Combined Model Metrics ===")
print(metrics_combined)
metrics_combined.to_csv("classification_metrics_all_models.csv", index=False)

print("\nAll done! Check the CSV and PNG files for detailed results.")
