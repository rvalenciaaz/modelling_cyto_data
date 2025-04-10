# inference_nn.py

import os
import json
import csv
import numpy as np
import torch
import polars as pl
import joblib

# Local imports
from src.model_utils import PyTorchNNClassifierWithVal


def load_artifacts(replication_folder="replication_files"):
    """
    Loads:
      - Trained state dict (best_model_state.pth)
      - Scaler (scaler.joblib)
      - LabelEncoder (label_encoder.joblib)
      - Features to keep (features_used.json)
      - Best hyperparams from best_params.json
    """
    scaler_path        = os.path.join(replication_folder, "scaler.joblib")
    label_encoder_path = os.path.join(replication_folder, "label_encoder.joblib")
    features_path      = os.path.join(replication_folder, "features_used.json")
    model_path         = os.path.join(replication_folder, "best_model_state.pth")
    best_params_path   = os.path.join(replication_folder, "best_params.json")

    # Check everything exists
    for path in [scaler_path, label_encoder_path, features_path, model_path, best_params_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact '{os.path.basename(path)}' not found in '{replication_folder}' folder."
            )

    # 1) Scaler
    scaler = joblib.load(scaler_path)

    # 2) LabelEncoder
    label_encoder = joblib.load(label_encoder_path)

    # 3) Features
    with open(features_path, "r") as f:
        features_to_keep = json.load(f)

    # 4) Model state
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # 5) Best hyperparams
    with open(best_params_path, "r") as f:
        best_params = json.load(f)

    return scaler, label_encoder, features_to_keep, state_dict, best_params


def predict_new_data(
    new_data_df,
    model,
    scaler,
    label_encoder,
    features_to_keep
):
    """
    Performs inference using the PyTorchNNClassifierWithVal model.

    Args:
        new_data_df (polars.DataFrame): Input dataframe containing features.
        model (PyTorchNNClassifierWithVal): A model instance with loaded state_dict.
        scaler (object): Scaler for data normalization.
        label_encoder (object): Label encoder for class labels.
        features_to_keep (list): List of feature columns used by the model.

    Returns:
        predicted_labels (np.ndarray): shape [n_data], predicted class labels.
        probs (np.ndarray): shape [n_data, n_classes], softmax probabilities.
    """
    # 1) Check for missing columns
    missing_cols = set(features_to_keep) - set(new_data_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in new data: {missing_cols}")

    # 2) Scale input
    X_new = new_data_df.select(features_to_keep).to_numpy()
    X_new_scaled = scaler.transform(X_new)

    # 3) Predict
    predicted_labels = label_encoder.inverse_transform(model.predict(X_new_scaled))
    probs = model.predict_proba(X_new_scaled)

    return predicted_labels, probs


def main_inference():
    """
    Main entry point for running inference using the PyTorchNNClassifierWithVal.
    1) Load artifacts (scaler, encoder, features, model state, best_params)
    2) Load new data (CSV)
    3) Rebuild the classifier & load weights (NO dummy fit!)
    4) Predict new data
    5) Save a CSV file with predictions & probabilities
    """
    # 1) Load artifacts
    replication_folder = "replication_files"
    (
        scaler,
        label_encoder,
        features_to_keep,
        state_dict,
        best_params
    ) = load_artifacts(replication_folder)

    # 2) Load new data
    new_data_path = "3-species_mock.csv"
    if not os.path.exists(new_data_path):
        print(f"ERROR: '{new_data_path}' not found. Provide a file for inference.")
        return

    new_data_df = pl.read_csv(new_data_path)
    print(f"Loaded new data from '{new_data_path}', shape={new_data_df.shape}")

    # 3) Rebuild classifier with best hyperparams & load weights
    hidden_size = best_params["hidden_size"]
    num_layers  = best_params["num_layers"]
    dropout     = best_params.get("dropout", 0.0)

    # Create the PyTorchNNClassifierWithVal instance
    model = PyTorchNNClassifierWithVal(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=0,  # we're only doing inference
        learning_rate=best_params.get("learning_rate", 1e-3),
        batch_size=best_params.get("batch_size", 32),
        verbose=False
    )

    # Manually set the input & output dimensions
    n_features = len(features_to_keep)
    n_classes = len(label_encoder.classes_)
    model.input_dim_ = n_features
    model.output_dim_ = n_classes

    # Build the ConfigurableNN (identical architecture to training) and load weights
    from src.model_utils import ConfigurableNN  # or wherever it's defined
    model.model_ = ConfigurableNN(
        input_dim=model.input_dim_,
        hidden_size=model.hidden_size,
        output_dim=model.output_dim_,
        num_layers=model.num_layers,
        dropout=model.dropout
    ).to(model.device)

    model.model_.load_state_dict(state_dict)
    model.model_.eval()

    # 4) Predict
    predicted_labels, probs = predict_new_data(
        new_data_df,
        model,
        scaler,
        label_encoder,
        features_to_keep
    )

    # 5) Save predictions & probabilities
    output_csv = "inference_predictions.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["RowIndex", "PredictedClass"] + [
            f"Prob_Class{i}" for i in range(n_classes)
        ]
        writer.writerow(header)

        # Rows
        for i, label in enumerate(predicted_labels):
            row = [i, label] + list(probs[i])
            writer.writerow(row)

    print(f"Saved inference results => {output_csv}")
    print("Inference complete!")


if __name__ == "__main__":
    main_inference()
