import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import glob

# For scaling and classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ====================================================
# 1. READING & SUBSAMPLING CSV FILES
# ====================================================
csv_files = glob.glob("species*.csv")
df_list = []

for file_path in csv_files:
    temp_df = pd.read_csv(file_path)

    # Subsample up to 10,000 rows
    temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)

    # Add Label column from filename, e.g. "species1.csv" -> "species1"
    label = file_path.split('.')[0]
    temp_df['Label'] = label

    df_list.append(temp_df)

print(df_list)
combined_df = pd.concat(df_list, ignore_index=True)

# Clean up the 'Label' column (remove "species" prefix)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

# ====================================================
# 2. FILTERING FEATURES BASED ON MAD
# ====================================================
# Select numeric columns
numerical_data = combined_df.select_dtypes(include=[np.number])

# Calculate CV & MAD
cv_results = {}
for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val  = numerical_data[col].std()
    
    # Coefficient of Variation; avoid division by zero
    cv  = std_val / mean_val if mean_val != 0 else np.nan
    mad = median_abs_deviation(numerical_data[col].values, scale='normal')

    cv_results[col] = [col, cv, mad]

cv_df = pd.DataFrame(cv_results.values(), columns=['Feature', 'CV', 'MAD'])

# Example threshold for MAD
MAD_THRESHOLD = 5
features_to_keep = cv_df.loc[cv_df["MAD"] <= MAD_THRESHOLD, "Feature"].tolist()

# Final DataFrame: keep numeric columns passing MAD threshold plus the 'Label'
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()

# ====================================================
# 3. FEATURE SCALING & CLASSIFICATION
# ====================================================
# Separate features (X) and label (y)
X = final_df.drop(columns=["Label"])
y = final_df["Label"]

# If 'y' is categorical (string labels), convert to numeric with LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Scale the features (except the label) with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Create and train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_scaled)

# ====================================================
# 4. EVALUATION
# ====================================================
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("\nClassification Report:")
print(report)
