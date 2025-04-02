# inference.py

import os
import json
import csv
import pickle
import numpy as np
import torch
import pyro
import polars as pl

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Local imports
from src.model_utils import bayesian_nn_model, create_guide
from src.prediction_utils import predict_pyro_model, predict_pyro_probabilities  # now using external functions

def load_artifacts(replication_folder="replication_files"):
    """
    Loads:
      - Pyro guide state dict (after a dummy forward pass to initialize shapes)
      - Scaler
      - LabelEncoder
      - Features to keep
      - Best hyperparams from best_params.json
    """
    # Load saved guide state dict
    guide_params_path = os.path.join(replication_folder, "bayesian_nn_pyro_params.pkl")
    with open(guide_params_path, "rb") as f:
        guide_state = pickle.load(f)

    # Load scaler
    scaler_path = os.path.join(replication_folder, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load label encoder
    encoder_path = os.path.join(replication_folder, "label_encoder.pkl")
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load features to keep
    features_path = os.path.join(replication_folder, "features_to_keep.json")
    with open(features_path, "r") as f:
        features_to_keep = json.load(f)

    # Load best hyperparams
    best_params_path = os.path.join(replication_folder, "best_params.json")
    with open(best_params_path, "r") as f:
        best_params = json.load(f)

    # Get hyperparameters for the dummy run
    hidden_size = best_params["hidden_size"]
    num_layers = best_params["num_layers"]
    output_dim = len(label_encoder.classes_)

    # Create the guide and perform a dummy forward pass to initialize internal parameters
    guide = create_guide(bayesian_nn_model)
    dummy_x = torch.randn((1, len(features_to_keep)))
    dummy_y = torch.tensor([0])
    guide(dummy_x, dummy_y, hidden_size=hidden_size, num_layers=num_layers, output_dim=output_dim)
    guide.load_state_dict(guide_state)
    guide.eval()

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
    Returns:
      predicted_class_labels: shape [n_data,]
      mean_probs: shape [n_data, output_dim]
      std_probs:  shape [n_data, output_dim]
      prob_samples: shape [num_samples, n_data, output_dim]
    """
    # 1) Check that all required columns are present
    missing_cols = set(features_to_keep) - set(new_data_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in new data: {missing_cols}")

    # 2) Extract features and scale them
    X_new = new_data_df.select(features_to_keep).to_numpy()
    X_new = scaler.transform(X_new)
    X_new_t = torch.tensor(X_new, dtype=torch.float32)

    if output_dim is None:
        output_dim = len(label_encoder.classes_)

    # 3) Get majority-vote predictions using the external predict function
    preds_class_ids = predict_pyro_model(
        X_new_t, guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples
    )
    predicted_class_labels = label_encoder.inverse_transform(preds_class_ids)

    # 4) Get full probability samples along with mean and std using the external function
    mean_probs, std_probs, prob_samples = predict_pyro_probabilities(
        X_new_t, guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples
    )

    # Debug prints to verify shapes
    print("Predicted labels shape:", predicted_class_labels.shape)
    print("Mean probabilities shape:", mean_probs.shape)
    print("Std probabilities shape:", std_probs.shape)

    return predicted_class_labels, mean_probs, std_probs, prob_samples

def main_inference():
    # 1) Load artifacts (guide, scaler, label encoder, feature list, best hyperparameters)
    replication_folder = "replication_files"
    guide, scaler, label_encoder, features_to_keep, best_params = load_artifacts(replication_folder)

    # 2) Load new data
    new_data_path = "6-species_mock.csv"  # Adjust path as needed
    new_data_df = pl.read_csv(new_data_path)
    print(f"Loaded new data from '{new_data_path}', shape={new_data_df.shape}")

    # 3) Set model architecture parameters from best_params
    hidden_size = best_params["hidden_size"]
    num_layers  = best_params["num_layers"]
    output_dim  = len(label_encoder.classes_)

    # 4) Perform predictions
    num_samples = 1000  # number of posterior samples
    (
        predicted_labels,
        mean_probs,
        std_probs,
        prob_samples
    ) = predict_new_data(
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

    # 5A) Save a CSV summary with predicted class, mean, and std probabilities
    output_csv = "inference_predictions.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Create header
        header = ["RowIndex", "PredictedClass"]
        for i in range(output_dim):
            header.append(f"MeanProb_Class{i}")
        for i in range(output_dim):
            header.append(f"StdProb_Class{i}")
        writer.writerow(header)

        # Write prediction rows
        for i, pred_class in enumerate(predicted_labels):
            row = [i, pred_class]
            row.extend(mean_probs[i])  # append mean probability vector
            row.extend(std_probs[i])   # append std probability vector
            writer.writerow(row)

    print(f"Saved inference summary => {output_csv}")

    # 5B) Save the entire distribution of probability samples as a pickle file
    output_pkl = "inference_probability_samples.pkl"
    with open(output_pkl, "wb") as f:
        pickle.dump(prob_samples, f)

    print(f"Saved full probability samples => {output_pkl}")
    print("Inference complete!")

if __name__ == "__main__":
    main_inference()
