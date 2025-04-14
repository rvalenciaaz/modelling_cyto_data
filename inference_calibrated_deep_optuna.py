#!/usr/bin/env python
"""
inference_calibrated.py

This script performs inference using a calibrated neural network model.
It:
  1. Loads the saved artifacts including:
       - StandardScaler (scaler.joblib)
       - LabelEncoder (label_encoder.joblib)
       - Features list (features_used.json)
       - Calibrated model (calibrated_model.joblib)
  2. Reads new input data from a CSV file.
  3. Checks for required feature columns and scales the input.
  4. Uses the calibrated model to predict class indices and probabilities.
  5. Inverse transforms indices to class labels.
  6. Saves predictions and probabilities to CSV for later use.

Usage:
    python inference_calibrated.py
"""

import os
import json
import csv
import joblib
import numpy as np
import polars as pl

def load_artifacts(replication_folder="outputs"):
    """
    Loads required artifacts for inference:
      - StandardScaler (scaler.joblib)
      - LabelEncoder (label_encoder.joblib)
      - Features list (features_used.json)
      - Calibrated model (calibrated_model.joblib)
    """
    scaler_path        = os.path.join(replication_folder, "scaler.joblib")
    label_encoder_path = os.path.join(replication_folder, "label_encoder.joblib")
    features_path      = os.path.join(replication_folder, "features_used.json")
    calibrated_model_path = os.path.join(replication_folder, "calibrated_model.joblib")

    for path in [scaler_path, label_encoder_path, features_path, calibrated_model_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact '{os.path.basename(path)}' not found in '{replication_folder}' folder.")

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    with open(features_path, "r") as f:
        features_to_keep = json.load(f)
    calibrator = joblib.load(calibrated_model_path)

    return scaler, label_encoder, features_to_keep, calibrator

def predict_new_data(new_data_df, scaler, label_encoder, features_to_keep, calibrator):
    """
    Prepares new data and performs prediction using the calibrated classifier.
    
    Args:
        new_data_df (polars.DataFrame): DataFrame containing new input data.
        scaler: Fitted StandardScaler instance.
        label_encoder: Fitted LabelEncoder instance.
        features_to_keep (list): List of feature names to select from new_data_df.
        calibrator: Calibrated classifier (scikit-learn CalibratedClassifierCV instance).

    Returns:
        predicted_labels (np.ndarray): Array of predicted class labels.
        probs (np.ndarray): Array of class probabilities.
    """
    # Ensure all required features are present
    missing_cols = set(features_to_keep) - set(new_data_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in new data: {missing_cols}")

    # Select and scale the input features
    X_new = new_data_df.select(features_to_keep).to_numpy()
    X_new_scaled = scaler.transform(X_new)

    # Predict using the calibrated classifier
    predicted_indices = calibrator.predict(X_new_scaled)
    predicted_labels = label_encoder.inverse_transform(predicted_indices)
    probs = calibrator.predict_proba(X_new_scaled)
    return predicted_labels, probs

def main():
    replication_folder = "outputs"

    # Load artifacts
    scaler, label_encoder, features_to_keep, calibrator = load_artifacts(replication_folder)
    
    # Load new data
    new_data_path = "3-species_mock.csv"
    if not os.path.exists(new_data_path):
        print(f"ERROR: '{new_data_path}' not found. Provide a valid CSV file for inference.")
        return

    new_data_df = pl.read_csv(new_data_path)
    print(f"Loaded new data from '{new_data_path}', shape: {new_data_df.shape}")

    # Get predictions from calibrated model
    predicted_labels, probs = predict_new_data(new_data_df, scaler, label_encoder, features_to_keep, calibrator)

    # Save the results to a CSV file
    n_classes = len(label_encoder.classes_)
    output_csv = "inference_calibrated_predictions.csv"
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write header: RowIndex, PredictedClass, Prob_Class0, Prob_Class1, ...
        header = ["RowIndex", "PredictedClass"] + [f"Prob_Class{i}" for i in range(n_classes)]
        writer.writerow(header)

        for i, label in enumerate(predicted_labels):
            row = [i, label] + list(probs[i])
            writer.writerow(row)

    print(f"Saved calibrated inference results to '{output_csv}'")
    print("Inference complete.")

if __name__ == "__main__":
    main()
