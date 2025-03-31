# inference_nn.py

import os
import json
import csv
import pickle  # if you need it for consistency; otherwise you can remove
import numpy as np
import torch
import polars as pl
import joblib

# Local imports
from src.model_utils import ConfigurableNN

def load_artifacts(replication_folder="replication_files"):
    """
    Loads:
      - State dict (best_model_state.pth)
      - Scaler (scaler.joblib)
      - LabelEncoder (label_encoder.joblib)
      - Features to keep (features_used.json)
      - Best hyperparams from best_params.json
    """
    # Paths to artifacts
    scaler_path        = os.path.join(replication_folder, "scaler.joblib")
    label_encoder_path = os.path.join(replication_folder, "label_encoder.joblib")
    features_path      = os.path.join(replication_folder, "features_used.json")
    model_path         = os.path.join(replication_folder, "best_model_state.pth")
    best_params_path   = os.path.join(replication_folder, "best_params.json")

    # Check existence
    if not all(os.path.exists(p) for p in [
        scaler_path, label_encoder_path, features_path, model_path, best_params_path
    ]):
        raise FileNotFoundError(
            "One or more required artifacts not found in replication_files/ folder."
        )

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load label encoder
    label_encoder = joblib.load(label_encoder_path)

    # Load features
    with open(features_path, "r") as f:
        features_used = json.load(f)

    # Load model state_dict
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Load best hyperparams
    with open(best_params_path, "r") as f:
        best_params = json.load(f)

    return scaler, label_encoder, features_used, state_dict, best_params


def predict_new_data(
    new_data_df,
    scaler,
    label_encoder,
    features_to_keep,
    state_dict,
    hidden_size=32,
    num_layers=2,
    dropout=0.0,
    output_dim=None
):
    """
    Performs inference using a trained neural network.
    
    Args:
        new_data_df (polars.DataFrame): Input dataframe containing features.
        scaler (object): Scaler for data normalization.
        label_encoder (object): Trained label encoder for class labels.
        features_to_keep (list): List of features used by the model.
        state_dict (dict): The trained model's state_dict.
        hidden_size (int): Hidden layer size.
        num_layers (int): Number of hidden layers.
        dropout (float): Dropout rate for intermediate layers.
        output_dim (int): Number of output classes. If None, inferred from label_encoder.
    
    Returns:
        predicted_labels (np.ndarray): Shape [n_data,]. Predicted class labels.
        probs (np.ndarray): Shape [n_data, output_dim]. Softmax probabilities for each class.
    """
    # 1) Check columns
    missing_cols = set(features_to_keep) - set(new_data_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in new data: {missing_cols}")

    # 2) Extract & scale
    X_new = new_data_df.select(features_to_keep).to_numpy()
    X_new_scaled = scaler.transform(X_new)
    X_torch = torch.tensor(X_new_scaled, dtype=torch.float32)

    # 3) Rebuild the model with the same architecture
    if output_dim is None:
        output_dim = len(label_encoder.classes_)
    input_dim = X_new_scaled.shape[1]

    model = ConfigurableNN(
        input_dim,
        hidden_size,
        output_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(state_dict)
    model.eval()

    # 4) Inference
    with torch.no_grad():
        logits = model(X_torch)
        probs = torch.softmax(logits, dim=1).numpy()
        preds_idx = np.argmax(probs, axis=1)
        predicted_labels = label_encoder.inverse_transform(preds_idx)

    return predicted_labels, probs


def main_inference():
    """
    Main entry point for running inference using a standard (non-Bayesian) neural network.
    1) Load artifacts (scaler, encoder, features, state_dict, best_params)
    2) Load new data (new_species_data.csv)
    3) Extract best hyperparams
    4) Run predictions
    5) Save a CSV file with predictions & probabilities
    """
    replication_folder = "replication_files"

    # 1) Load artifacts
    (
        scaler,
        label_encoder,
        features_used,
        state_dict,
        best_params
    ) = load_artifacts(replication_folder)

    # 2) Load new data
    new_data_path = "new_species_data.csv"
    if not os.path.exists(new_data_path):
        print(f"ERROR: '{new_data_path}' not found. Provide a file for inference.")
        return
    new_data_df = pl.read_csv(new_data_path)
    print(f"Loaded new data from '{new_data_path}', shape={new_data_df.shape}")

    # 3) Extract best hyperparams
    hidden_size = best_params["hidden_size"]
    num_layers  = best_params["num_layers"]
    dropout     = best_params.get("dropout", 0.0)
    output_dim  = len(label_encoder.classes_)

    # 4) Perform predictions
    predicted_labels, probs = predict_new_data(
        new_data_df,
        scaler,
        label_encoder,
        features_used,
        state_dict,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_dim
    )

    # 5) Save a CSV summary with Predicted Class and Probabilities
    output_csv = "inference_predictions.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Create header
        header = ["RowIndex", "PredictedClass"]
        for i in range(output_dim):
            header.append(f"Prob_Class{i}")
        writer.writerow(header)

        # Write rows
        for i, pred_class in enumerate(predicted_labels):
            row = [i, pred_class]
            row.extend(probs[i])  # append the probability vector
            writer.writerow(row)

    print(f"Saved inference summary => {output_csv}")
    print("Inference complete!")


if __name__ == "__main__":
    main_inference()
