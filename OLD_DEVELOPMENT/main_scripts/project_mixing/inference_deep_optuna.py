import os
import json
import numpy as np
import torch
import polars as pl
import joblib

from src.model_utils import ConfigurableNN

def load_artifacts(replication_folder="replication_files"):
    """
    Loads:
      - State dict
      - Scaler
      - LabelEncoder
      - Features to keep
      - Best hyperparams from best_params.json
    """
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    label_encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
    features_path = os.path.join(OUTPUT_DIR, "features_used.json")
    model_path = os.path.join(OUTPUT_DIR, "best_model_state.pth")

    if not (os.path.exists(scaler_path) and os.path.exists(label_encoder_path)
            and os.path.exists(features_path) and os.path.exists(model_path)):
        raise FileNotFoundError("One or more required artifacts not found in outputs/ folder.")

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    with open(features_path, "r") as f:
        features_used = json.load(f)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))


def predict_new_data(
    new_data_df,
    scaler,
    label_encoder,
    features_to_keep,
    hidden_size=32,
    num_layers=2,
    output_dim=None,
    num_samples=500
):

OUTPUT_DIR = "outputs"

def main_inference():
    new_data_path = "new_species_data.csv"
    if not os.path.exists(new_data_path):
        print(f"ERROR: '{new_data_path}' not found. Provide a file for inference.")
        return

    new_df = pl.read_csv(new_data_path)
    # Check we have the required columns
    missing_cols = set(features_used) - set(new_df.columns)
    if missing_cols:
        raise ValueError(f"New data is missing required features: {missing_cols}")

    # 3) Prepare data
    X_new = new_df.select(features_used).to_numpy()
    X_new_scaled = scaler.transform(X_new)

    # 4) Rebuild the model with the same architecture
    #    (In a real scenario, parse the best hyperparams from 'metrics.json'!)
    #    Here we'll just guess some values or specify manually:
    hidden_size = 64
    num_layers  = 10
    dropout     = 0.1
    input_dim   = X_new_scaled.shape[1]
    output_dim  = len(label_encoder.classes_)

    model = ConfigurableNN(
        input_dim, hidden_size, output_dim,
        num_layers=num_layers, dropout=dropout
    )
    model.load_state_dict(state_dict)
    model.eval()

    # 5) Inference
    X_torch = torch.from_numpy(X_new_scaled).float()
    with torch.no_grad():
        logits = model(X_torch)
        probs = torch.softmax(logits, dim=1).numpy()
        preds_idx = np.argmax(probs, axis=1)
        preds_labels = label_encoder.inverse_transform(preds_idx)

    # 6) Display results
    for i, label in enumerate(preds_labels):
        print(f"Row {i}: predicted => {label}, probabilities => {probs[i]}")

if __name__ == "__main__":
    main_inference()
