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
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.size': 14
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
# Load saved preprocessing objects and features list (once)
# ------------------------------------------------------------
scaler = joblib.load(os.path.join(preprocessing_folder, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(preprocessing_folder, "label_encoder.pkl"))
features_df = pd.read_csv(os.path.join(preprocessing_folder, "features_to_keep.csv"))
features_to_keep = features_df["Feature"].tolist()

# ------------------------------------------------------------
# Load saved classical models (once)
# ------------------------------------------------------------
best_rf = joblib.load(os.path.join(preprocessing_folder, "best_rf_model.pkl"))
best_lr = joblib.load(os.path.join(preprocessing_folder, "best_lr_model.pkl"))
best_xgb = joblib.load(os.path.join(preprocessing_folder, "best_xgb_model.pkl"))

# ------------------------------------------------------------
# Load saved NN model (once)
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
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
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

    def predict(self, X):
        self.model_.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model_(X_torch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

# Create an instance of the classifier (we'll reinitialize its model each loop)
classifier = PyTorchNNClassifierWithVal(**saved_params)

# ------------------------------------------------------------
# Define model mapping for legend and desired hue order
# ------------------------------------------------------------
model_mapping = {
    "LR_Prediction": "Logistic Regression",
    "RF_Prediction": "Random Forest",
    "XGB_Prediction": "XGBoost",
    "NN_Prediction": "NN/MLP"
}
hue_order = ["Logistic Regression", "Random Forest", "XGBoost", "NN/MLP"]

# ------------------------------------------------------------
# Process each species file
# ------------------------------------------------------------
species_files = ["3-species_mock.csv", "6-species_mock.csv", "9-species_mock.csv"]

for input_filename in species_files:
    # Extract prefix from filename (e.g., "3-species")
    prefix = input_filename.split("_mock.csv")[0]
    
    # Read and preprocess new dataset
    new_data = pd.read_csv(input_filename)
    missing_features = [feat for feat in features_to_keep if feat not in new_data.columns]
    if missing_features:
        print(f"Warning: The following features are missing in {input_filename} and will be filled with 0:")
        print(missing_features)
        for feat in missing_features:
            new_data[feat] = 0
    X_new = new_data[features_to_keep].values
    X_new_scaled = scaler.transform(X_new)
    
    # --------------------------------------------------------
    # Classical Models Predictions
    # --------------------------------------------------------
    pred_rf = best_rf.predict(X_new_scaled)
    pred_lr = best_lr.predict(X_new_scaled)
    pred_xgb = best_xgb.predict(X_new_scaled)
    new_data["RF_Prediction"] = label_encoder.inverse_transform(pred_rf)
    new_data["LR_Prediction"] = label_encoder.inverse_transform(pred_lr)
    new_data["XGB_Prediction"] = label_encoder.inverse_transform(pred_xgb)
    
    # --------------------------------------------------------
    # NN Model Predictions
    # --------------------------------------------------------
    classifier.input_dim_ = X_new_scaled.shape[1]
    classifier.output_dim_ = len(label_encoder.classes_)
    classifier.model_ = ConfigurableNN(
        input_dim=classifier.input_dim_,
        hidden_size=saved_params['hidden_size'],
        output_dim=classifier.output_dim_,
        num_layers=saved_params['num_layers']
    ).to(classifier.device)
    classifier.model_.load_state_dict(state_dict)
    classifier.model_.eval()
    pred_nn = classifier.predict(X_new_scaled)
    new_data["NN_Prediction"] = label_encoder.inverse_transform(pred_nn)
    
    # --------------------------------------------------------
    # Save combined predictions to CSV
    # --------------------------------------------------------
    output_csv = f"{prefix}_combined_predictions.csv"
    new_data.to_csv(output_csv, index=False)
    print(f"Combined predictions saved to {output_csv}")
    
    # --------------------------------------------------------
    # Prepare DataFrame for plotting (melt prediction columns)
    # --------------------------------------------------------
    df_melt = new_data.melt(value_vars=["RF_Prediction", "LR_Prediction", "XGB_Prediction", "NN_Prediction"],
                            var_name="Model", value_name="Label")
    df_melt["Model"] = df_melt["Model"].replace(model_mapping)
    
    # --------------------------------------------------------
    # Grouped Count Plot (Raw Counts)
    # --------------------------------------------------------
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
    
    # --------------------------------------------------------
    # Grouped Percentage Plot
    # --------------------------------------------------------
    df_grouped = df_melt.groupby(["Model", "Label"]).size().reset_index(name="Count")
    df_grouped["Percentage"] = df_grouped.groupby("Model")["Count"].transform(lambda x: 100 * x / x.sum())
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Label", y="Percentage", hue="Model", data=df_grouped,
                order=sorted(df_melt["Label"].unique()), hue_order=hue_order)
    plt.title(f"{prefix}: Percentage of Samples Classified per Label (All Models)")
    plt.xlabel("Label")
    plt.ylabel("Percentage (\\%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    combined_perc_filename = os.path.join(output_folder, f"{prefix}_barplot_all_models_percentage.png")
    plt.savefig(combined_perc_filename)
    plt.close()
    print(f"Saved grouped percentage barplot: {combined_perc_filename}")
    
    # --------------------------------------------------------
    # Additional Plot: Count Ordered by Mean Count
    # --------------------------------------------------------
    mean_count = df_grouped.groupby("Label")["Count"].mean()
    order_by_mean = mean_count.sort_values(ascending=False).index.tolist()
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
    print(f"Saved ordered grouped count barplot: {ordered_counts_filename}")
    
    # --------------------------------------------------------
    # Additional Plot: Percentage Ordered by Mean Percentage
    # --------------------------------------------------------
    mean_percentage = df_grouped.groupby("Label")["Percentage"].mean()
    order_by_mean_perc = mean_percentage.sort_values(ascending=False).index.tolist()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Label", y="Percentage", hue="Model", data=df_grouped,
                order=order_by_mean_perc, hue_order=hue_order)
    plt.title(f"{prefix}: Percentage of Samples Classified per Label (All Models)\nOrdered by Mean Percentage")
    plt.xlabel("Label")
    plt.ylabel("Percentage (\\%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    ordered_perc_filename = os.path.join(output_folder, f"{prefix}_barplot_all_models_ordered_percentage.png")
    plt.savefig(ordered_perc_filename)
    plt.close()
    print(f"Saved ordered grouped percentage barplot: {ordered_perc_filename}")
    
    print(f"Finished processing {prefix}\n")
