import os
import json
import csv
import pickle
import numpy as np
import torch
import pyro

# scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Local imports
from src.model_utils import bayesian_nn_model, create_guide
from src.prediction_utils import (
    predict_pyro_model,
    predict_pyro_probabilities
)

def load_artifacts(replication_folder="replication_files"):
    """
    Loads:
      1) Pyro guide state dict (bayesian_nn_pyro_params.pkl),
      2) Scaler (scaler.pkl),
      3) Label Encoder (label_encoder.pkl),
      4) Feature list (features_to_keep.json),
      5) Best hyperparams from best_params.json.

    Returns:
      guide, scaler, label_encoder, features_to_keep, best_params
    """
    # 1) Load Pyro guide parameters
    guide_params_path = os.path.join(replication_folder, "bayesian_nn_pyro_params.pkl")
    with open(guide_params_path, "rb") as f:
        guide_state = pickle.load(f)

    guide = create_guide(bayesian_nn_model)
    guide.load_state_dict(guide_state)

    # 2) Load scaler
    scaler_path = os.path.join(replication_folder, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 3) Load label encoder
    encoder_path = os.path.join(replication_folder, "label_encoder.pkl")
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # 4) Load features
    features_path = os.path.join(replication_folder, "features_to_keep.json")
    with open(features_path, "r") as f:
        features_to_keep = json.load(f)

    # 5) Load best hyperparams
    best_params_path = os.path.join(replication_folder, "best_params.json")
    with open(best_params_path, "r") as f:
        best_params = json.load(f)

    return guide, scaler, label_encoder, features_to_keep, best_params


def predict_new_data(
    new_data_df,
    guide,
    scaler,
    label_encoder,
    features_to_keep,
    hidden_size=32,
    num_layers=2,
    output_dim=None,
    num_samples=500
):
    """
    Runs the Bayesian NN on new data and returns:
      (predicted_class_labels, mean_probs, std_probs)
    """
    # 1) Ensure columns are present
    missing_cols = set(features_to_keep) - set(new_data_df.columns)
    if missing_cols:
        raise ValueError(f"New data is missing required features: {missing_cols}")

    # 2) Extract and scale
    X_new = new_data_df.select(features_to_keep).to_numpy()
    X_new = scaler.transform(X_new)
    X_new_t = torch.tensor(X_new, dtype=torch.float32)

    # 3) If not specified, use the label_encoder for output_dim
    if output_dim is None:
        output_dim = len(label_encoder.classes_)

    # 4) Predict classes (majority vote)
    preds_class_ids = predict_pyro_model(
        X_new_t,
        guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples
    )
    preds_class_labels = label_encoder.inverse_transform(preds_class_ids)

    # 5) Probability & uncertainty
    mean_probs, std_probs = predict_pyro_probabilities(
        X_new_t,
        guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples
    )

    return preds_class_labels, mean_probs, std_probs


def main_inference():
    # 1) Load all artifacts
    replication_folder = "replication_files"
    guide, scaler, label_encoder, features_to_keep, best_params = load_artifacts(
        replication_folder
    )

    # 2) Load new data
    import polars as pl
    new_data_csv = "new_species_data.csv"  # adjust as needed
    new_data_df = pl.read_csv(new_data_csv)
    print(f"Loaded new data from {new_data_csv}, shape={new_data_df.shape}")

    # 3) Extract best hyperparams for the architecture
    hidden_size   = best_params["hidden_size"]
    num_layers    = best_params["num_layers"]
    output_dim    = len(label_encoder.classes_)

    # 4) Run predictions
    num_samples = 1000  # number of posterior samples
    predicted_labels, mean_probs, std_probs = predict_new_data(
        new_data_df,
        guide,
        scaler,
        label_encoder,
        features_to_keep,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples
    )

    # 5) Save predictions to CSV
    output_csv = "inference_predictions.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["RowIndex", "PredictedClass"]
        for i in range(output_dim):
            header.append(f"MeanProb_Class{i}")
        for i in range(output_dim):
            header.append(f"StdProb_Class{i}")
        writer.writerow(header)

        # Rows
        for idx, predicted_class in enumerate(predicted_labels):
            row_data = [idx, predicted_class]
            row_data.extend(mean_probs[idx])
            row_data.extend(std_probs[idx])
            writer.writerow(row_data)

    print(f"Inference completed! Saved results to {output_csv}.")


if __name__ == "__main__":
    main_inference()
