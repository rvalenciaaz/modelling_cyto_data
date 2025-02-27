import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
# Set plot style and increase font sizes for all plots
plt.style.use(["science", "nature"])
plt.rcParams.update({
    'axes.titlesize': 16,    # Title font size
    'axes.labelsize': 14,    # Axis labels font size
    'xtick.labelsize': 12,   # X tick labels font size
    'ytick.labelsize': 12,   # Y tick labels font size
    'legend.fontsize': 12,   # Legend font size
    'font.size': 14          # Default font size
})

# Create output folder for barplots if it doesn't exist
output_folder = "./barplots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ------------------------------------------------------------
# Define paths for preprocessing and model parameters
# ------------------------------------------------------------
preprocessing_folder = "./output_classical_100k/"
nn_model_folder = "./output_nn_100k/"

# ------------------------------------------------------------
# Load saved preprocessing objects and features list
# ------------------------------------------------------------
scaler = joblib.load(os.path.join(preprocessing_folder, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(preprocessing_folder, "label_encoder.pkl"))
features_df = pd.read_csv(os.path.join(preprocessing_folder, "features_to_keep.csv"))
features_to_keep = features_df["Feature"].tolist()

# ------------------------------------------------------------
# Load saved classical models
# ------------------------------------------------------------
best_rf = joblib.load(os.path.join(preprocessing_folder, "best_rf_model.pkl"))
best_lr = joblib.load(os.path.join(preprocessing_folder, "best_lr_model.pkl"))
best_xgb = joblib.load(os.path.join(preprocessing_folder, "best_xgb_model.pkl"))

# ------------------------------------------------------------
# Load saved NN model (best estimator)
# ------------------------------------------------------------
# The NN model was saved as a dict with keys: "state_dict" and "params"
saved_data = torch.load(os.path.join(nn_model_folder, "best_estimator.pth"), map_location=torch.device("cpu"))
state_dict = saved_data["state_dict"]
saved_params = saved_data["params"]

# ------------------------------------------------------------
# Define the NN architecture and classifier class
# ------------------------------------------------------------
class ConfigurableNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=1):
        super(ConfigurableNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class PyTorchNNClassifierWithVal:
    def __init__(self, hidden_size=32, learning_rate=1e-3, batch_size=32,
                 epochs=10, num_layers=1, verbose=True):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_layers = num_layers
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim_ = None
        self.output_dim_ = None
        self.model_ = None
        self.classes_ = None

    def predict(self, X):
        self.model_.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model_(X_torch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

# Reconstruct the NN classifier using saved parameters and load weights.
classifier = PyTorchNNClassifierWithVal(**saved_params)

# ------------------------------------------------------------
# Read and preprocess new dataset
# ------------------------------------------------------------
input_filename = "3-species_mock.csv"  # Change to your new CSV file path if needed.
# Extract prefix from input filename (portion before '_mock.csv')
prefix = input_filename.split("_mock.csv")[0]

new_data = pd.read_csv(input_filename)

# Check for any missing features and fill with 0 if needed.
missing_features = [feat for feat in features_to_keep if feat not in new_data.columns]
if missing_features:
    print("Warning: The following features are missing in the new dataset and will be filled with 0:")
    print(missing_features)
    for feat in missing_features:
        new_data[feat] = 0

# Order the features as used during training and scale them.
X_new = new_data[features_to_keep].values
X_new_scaled = scaler.transform(X_new)

# ------------------------------------------------------------
# Make predictions using classical models
# ------------------------------------------------------------
pred_rf = best_rf.predict(X_new_scaled)
pred_lr = best_lr.predict(X_new_scaled)
pred_xgb = best_xgb.predict(X_new_scaled)
new_data["RF_Prediction"] = label_encoder.inverse_transform(pred_rf)
new_data["LR_Prediction"] = label_encoder.inverse_transform(pred_lr)
new_data["XGB_Prediction"] = label_encoder.inverse_transform(pred_xgb)

# ------------------------------------------------------------
# Make prediction using NN model
# ------------------------------------------------------------
input_dim = X_new_scaled.shape[1]
classifier.input_dim_ = input_dim
classifier.output_dim_ = len(label_encoder.classes_)
classifier.model_ = ConfigurableNN(
    input_dim=input_dim,
    hidden_size=saved_params['hidden_size'],
    output_dim=classifier.output_dim_,
    num_layers=saved_params['num_layers']
).to(classifier.device)
classifier.model_.load_state_dict(state_dict)
classifier.model_.eval()
pred_nn = classifier.predict(X_new_scaled)
new_data["NN_Prediction"] = label_encoder.inverse_transform(pred_nn)

# Save the combined predictions to a CSV file (with prefix)
output_csv = f"{prefix}_combined_predictions.csv"
new_data.to_csv(output_csv, index=False)
print(f"Combined predictions saved to {output_csv}")

# Define mapping for legend labels and desired hue order.
model_mapping = {
    "LR_Prediction": "Logistic Regression",
    "RF_Prediction": "Random Forest",
    "XGB_Prediction": "XGBoost",
    "NN_Prediction": "NN/MLP"
}
hue_order = ["Logistic Regression", "Random Forest", "XGBoost", "NN/MLP"]

# ------------------------------------------------------------
# Create a grouped barplot showing prediction counts for all models
# ------------------------------------------------------------
# Melt the prediction columns into a long DataFrame.
df_melt = new_data.melt(value_vars=["RF_Prediction", "LR_Prediction", "XGB_Prediction", "NN_Prediction"],
                        var_name="Model", value_name="Label")
# Replace the model names with expanded names.
df_melt["Model"] = df_melt["Model"].replace(model_mapping)

plt.figure(figsize=(10, 6))
sns.countplot(x="Label", hue="Model", data=df_melt,
              order=sorted(df_melt["Label"].unique()), hue_order=hue_order)
plt.title(f"{prefix}: Number of Samples Classified per Label (All Models)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
combined_counts_filename = os.path.join(output_folder, f"{prefix}_barplot_all_models.png")
plt.savefig(combined_counts_filename)
plt.close()
print(f"Saved grouped count barplot: {combined_counts_filename}")

# ------------------------------------------------------------
# Create a grouped barplot showing prediction percentages for all models
# ------------------------------------------------------------
# First, compute counts per Model and Label.
df_grouped = df_melt.groupby(["Model", "Label"]).size().reset_index(name="Count")
# Then, compute the percentage per Model.
df_grouped["Percentage"] = df_grouped.groupby("Model")["Count"].transform(lambda x: 100 * x / x.sum())

plt.figure(figsize=(10, 6))
sns.barplot(x="Label", y="Percentage", hue="Model", data=df_grouped,
            order=sorted(df_melt["Label"].unique()), hue_order=hue_order)
plt.title(f"{prefix}: Percentage of Samples Classified per Label (All Models)")
plt.xlabel("Label")
# Escape the % symbol for LaTeX rendering by using a double backslash.
plt.ylabel("Percentage (\\%)")
plt.xticks(rotation=45)
plt.tight_layout()
combined_perc_filename = os.path.join(output_folder, f"{prefix}_barplot_all_models_percentage.png")
plt.savefig(combined_perc_filename)
plt.close()
print(f"Saved grouped percentage barplot: {combined_perc_filename}")

# ------------------------------------------------------------
# Additional Plots: Ordered by Mean Value over Model Predictions
# ------------------------------------------------------------
# Compute mean count per Label across models.
mean_count = df_grouped.groupby("Label")["Count"].mean()
order_by_mean = mean_count.sort_values(ascending=False).index.tolist()

# Additional Grouped Count Plot, ordered by mean count
plt.figure(figsize=(10, 6))
sns.countplot(x="Label", hue="Model", data=df_melt, order=order_by_mean, hue_order=hue_order)
plt.title(f"{prefix}: Number of Samples Classified per Label (All Models)\nOrdered by Mean Count")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
ordered_counts_filename = os.path.join(output_folder, f"{prefix}_barplot_all_models_ordered_counts.png")
plt.savefig(ordered_counts_filename)
plt.close()
print(f"Saved grouped count barplot ordered by mean count: {ordered_counts_filename}")

# Compute mean percentage per Label across models.
mean_percentage = df_grouped.groupby("Label")["Percentage"].mean()
order_by_mean_perc = mean_percentage.sort_values(ascending=False).index.tolist()

# Additional Grouped Percentage Plot, ordered by mean percentage
plt.figure(figsize=(10, 6))
sns.barplot(x="Label", y="Percentage", hue="Model", data=df_grouped, order=order_by_mean_perc, hue_order=hue_order)
plt.title(f"{prefix}: Percentage of Samples Classified per Label (All Models)\nOrdered by Mean Percentage")
plt.xlabel("Label")
plt.ylabel("Percentage (\\%)")
plt.xticks(rotation=45)
plt.tight_layout()
ordered_perc_filename = os.path.join(output_folder, f"{prefix}_barplot_all_models_ordered_percentage.png")
plt.savefig(ordered_perc_filename)
plt.close()
print(f"Saved grouped percentage barplot ordered by mean percentage: {ordered_perc_filename}")
