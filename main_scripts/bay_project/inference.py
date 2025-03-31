import os
import json
import pickle
import numpy as np
import torch
import pyro

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Local imports
from src.model_utils import bayesian_nn_model, create_guide
from src.prediction_utils import predict_pyro_probabilities, predict_pyro_model

def load_artifacts(replication_folder="replication_files"):
    """
    Load all required artifacts: pyro params, scaler, label encoder, features.
    Returns a tuple of (guide, scaler, label_encoder, features_to_keep).
    """
    # Load model parameters
    guide_params_path = os.path.join(replication_folder, "bayesian_nn_pyro_params.pkl")
    with open(guide_params_path, "rb") as f:
        guide_state = pickle.load(f)

    # Create a fresh guide with the same model definition
    guide = create_guide(bayesian_nn_model)
    guide.load_state_dict(guide_state)

    # Load scaler
    scaler_path = os.path.join(replication_folder, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load label encoder
    encoder_path = os.path.join(replication_folder, "label_encoder.pkl")
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load features
    features_path = os.path.join(replication_folder, "features_to_keep.json")
    with open(features_path, "r") as f:
        features_to_keep = json.load(f)

    return guide, scaler, label_encoder, features_to_keep


def predict_new_data(new_data_df, guide, scaler, label_encoder, features_to_keep,
                     hidden_size=32, num_layers=2, output_dim=None, num_samples=500):
    """
    Given a new Polars/Pandas DataFrame with the same features, produce:
      - predicted classes
      - mean probabilities, and probability std dev (uncertainty)
    """
    # Make sure we select the same features as were kept during training
    if not set(features_to_keep).issubset(set(new_data_df.columns)):
        missing = set(features_to_keep) - set(new_data_df.columns)
        raise ValueError(f"New data is missing required features: {missing}")

    X_new = new_data_df.select(features_to_keep).to_numpy()
    X_new = scaler.transform(X_new)
    X_new_t = torch.tensor(X_new, dtype=torch.float32)

    # If output_dim not provided, infer from label_encoder
    if output_dim is None:
        output_dim = len(label_encoder.classes_)

    # Classification predictions (majority vote)
    preds_class_ids = predict_pyro_model(
        X_new_t, guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples
    )
    # Map back to original labels
    preds_class_labels = label_encoder.inverse_transform(preds_class_ids)

    # Probability & uncertainty
    mean_probs, std_probs = predict_pyro_probabilities(
        X_new_t, guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples
    )

    return preds_class_labels, mean_probs, std_probs


def main_inference():
    # 1. Load final artifacts
    guide, scaler, label_encoder, features_to_keep = load_artifacts()

    # 2. Suppose we have new data in a Polars DataFrame (some "new_species_data.csv")
    #    that has the same columns as training, at least for the features we kept.
    import polars as pl
    new_data_df = pl.read_csv("new_species_data.csv")

    # 3. Predict
    hidden_size = 32   # Must match best hyperparams or your final chosen architecture
    num_layers  = 10   # Example from best_params; adapt to your actual best_params
    # If you stored them in "metrics_pyro.json" or your best_params, fetch from there
    output_dim  = len(label_encoder.classes_)

    class_labels, mean_probs, std_probs = predict_new_data(
        new_data_df, guide, scaler, label_encoder, features_to_keep,
        hidden_size=hidden_size, num_layers=num_layers, output_dim=output_dim,
        num_samples=1000
    )

    # 4. Print results
    for i, row_label in enumerate(class_labels):
        print(f"Row {i}: predicted class => {row_label}")
        print(f"         mean probs => {mean_probs[i]}")
        print(f"         std probs  => {std_probs[i]}")
        print("------------------------------------------------")


if __name__ == "__main__":
    main_inference()
