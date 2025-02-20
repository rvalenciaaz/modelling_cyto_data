import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

# For modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================
# 1. READING & SUBSAMPLING CSV FILES
# ====================================================
csv_files = glob.glob("species*.csv")
df_list = []

for file_path in csv_files:
    temp_df = pd.read_csv(file_path)

    # Subsample up to 10,000 rows
    temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)

    # Create a label from filename, e.g. "species1.csv" -> "species1"
    label = file_path.split('.')[0]
    temp_df['Label'] = label

    df_list.append(temp_df)

combined_df = pd.concat(df_list, ignore_index=True)

# Clean up the 'Label' column (remove "species" prefix)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

# ====================================================
# 2. FILTERING FEATURES BASED ON MAD
# ====================================================
# Select numeric columns
numerical_data = combined_df.select_dtypes(include=[np.number])

# Calculate CV & MAD for each numeric column
cv_results = {}
for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val = numerical_data[col].std()

    # Coefficient of Variation
    cv = std_val / mean_val if mean_val != 0 else np.nan
    mad = median_abs_deviation(numerical_data[col].values, scale='normal')

    cv_results[col] = [col, cv, mad]

cv_df = pd.DataFrame(cv_results.values(), columns=['Feature', 'CV', 'MAD'])

# Threshold for MAD
MAD_THRESHOLD = 5
features_to_keep = cv_df.loc[cv_df["MAD"] <= MAD_THRESHOLD, "Feature"].tolist()

# Final DataFrame with columns passing MAD threshold plus the 'Label'
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()

# ====================================================
# 3. FEATURE SCALING & TRAIN/TEST SPLIT
# ====================================================
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

# ====================================================
# 4. CLASSIFICATION
# ====================================================
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

# ====================================================
# 5. EVALUATION
# ====================================================

# ---------------------------
# 5A) Basic metrics & Report
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"MCC: {mcc:.4f}")

# Create a classification report as a dict
report_dict = classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_, 
    output_dict=True
)
# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()
print("\nClassification Report:")
print(report_df)

# Save the classification report to a CSV file
report_df.to_csv("classification_report_rf.csv", index=True)

# Optionally, save accuracy and MCC in a metrics CSV
metrics_df = pd.DataFrame({"Accuracy": [accuracy], "MCC": [mcc]})
metrics_df.to_csv("classification_metrics_rf.csv", index=False)

# ---------------------------
# 5B) Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')  # normalized version

# Method 1: Using sklearn's ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Random Forest)")
plt.savefig("confusion_matrix_rf.png")
plt.close()

# Method 2: Seaborn heatmap of normalized confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            fmt=".2f")
plt.title("Normalized Confusion Matrix (Random Forest)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix_rf_normalized.png")
plt.close()

# ---------------------------
# 5C) Feature Importances
# ---------------------------
importances = clf.feature_importances_
feature_names = X.columns  # column names in the same order as X

# Combine into a DataFrame
fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance descending
fi_df.sort_values("Importance", ascending=False, inplace=True)

# Print & save to CSV
print("\nFeature Importances (top 10 shown):")
print(fi_df.head(10))
fi_df.to_csv("feature_importances_rf.csv", index=False)

# Plot and save feature importances (bar chart)
plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x="Importance", y="Feature", color='skyblue')
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances_rf.png")
plt.close()
