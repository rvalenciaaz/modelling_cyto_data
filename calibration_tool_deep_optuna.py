#!/usr/bin/env python
"""
calibrate_nn.py

This script calibrates the predicted probability outputs of a trained neural network
(using PyTorchNNClassifierWithVal) via scikit-learn’s CalibratedClassifierCV.

It:
  1. Loads calibration data from the NPZ file (saved in main_nn.py).
  2. Loads saved artifacts (scaler, label encoder, feature list, best hyperparameters, and model state)
     from the replication folder (outputs/).
  3. Rebuilds the neural network model similarly to the inference routine.
  4. Scales the calibration data.
  5. Wraps the pre-trained model with CalibratedClassifierCV (using sigmoid calibration).
  6. Fits the calibrator on the calibration set.
  7. Optionally plots a calibration curve (for binary classification).
  8. Saves the calibrated model to disk for later use in inference.

Usage:
    python calibrate_nn.py
"""

import os
import json
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Import your model class and (if needed) the configurable model builder
from src.model_utils import PyTorchNNClassifierWithVal, ConfigurableNN

def load_artifacts(replication_folder="outputs"):
    """
    Loads artifacts saved during training.
    
    Expected files:
      - scaler.joblib
      - label_encoder.joblib
      - features_used.json
      - best_model_state.pth
      - best_params.json
    """
    scaler_path        = os.path.join(replication_folder, "scaler.joblib")
    label_encoder_path = os.path.join(replication_folder, "label_encoder.joblib")
    features_path      = os.path.join(replication_folder, "features_used.json")
    model_path         = os.path.join(replication_folder, "best_model_state.pth")
    best_params_path   = os.path.join(replication_folder, "best_params.json")

    for path in [scaler_path, label_encoder_path, features_path, model_path, best_params_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact '{os.path.basename(path)}' not found in '{replication_folder}' folder."
            )
    
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    
    with open(features_path, "r") as f:
        features_to_keep = json.load(f)
        
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    
    with open(best_params_path, "r") as f:
        best_params = json.load(f)
        
    return scaler, label_encoder, features_to_keep, state_dict, best_params

def rebuild_model(best_params, state_dict, features_to_keep, label_encoder):
    """
    Rebuilds the PyTorch neural network classifier with the architecture used during training.
    It sets the proper input and output dimensions, instantiates the model, and loads the state dict.
    """
    hidden_size = best_params["hidden_size"]
    num_layers  = best_params["num_layers"]
    dropout     = best_params.get("dropout", 0.0)

    # Create model instance (for inference, epochs set to 0, no training)
    model = PyTorchNNClassifierWithVal(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=0,  # inference only
        learning_rate=best_params.get("learning_rate", 1e-3),
        batch_size=best_params.get("batch_size", 32),
        verbose=False
    )

    # Set correct input/output dimensions
    n_features = len(features_to_keep)
    n_classes = len(label_encoder.classes_)
    model.input_dim_ = n_features
    model.output_dim_ = n_classes

    # Instantiate the underlying neural network with identical architecture as used during training.
    model.model_ = ConfigurableNN(
        input_dim=model.input_dim_,
        hidden_size=model.hidden_size,
        output_dim=model.output_dim_,
        num_layers=model.num_layers,
        dropout=model.dropout
    ).to(model.device)
    
    # Load the trained state dictionary
    model.model_.load_state_dict(state_dict)
    model.model_.eval()
    
    return model

def main():
    # 1. Load calibration data from NPZ file.
    npz_path = "data_for_calibration.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Calibration data file '{npz_path}' not found!")
    
    # The NPZ file contains X_train, y_train, X_test, y_test.
    # Here we choose to use the test split as our calibration set.
    data = np.load(npz_path)
    X_cal, y_cal = data["X_test"], data["y_test"]
    print(f"Loaded calibration data from '{npz_path}': X_cal shape = {X_cal.shape}, y_cal shape = {y_cal.shape}")

    # 2. Load artifacts saved from training.
    replication_folder = "outputs"
    scaler, label_encoder, features_to_keep, state_dict, best_params = load_artifacts(replication_folder)
    
    # 3. Rebuild the trained model.
    base_model = rebuild_model(best_params, state_dict, features_to_keep, label_encoder)
    
    # 4. Scale the calibration data.
    X_cal_scaled = scaler.transform(X_cal)
    
    # 5. Wrap the pre-trained model with a calibrator.
    #
    # NOTE:
    # - We use cv="prefit" because the base_model is already trained.
    # - The base_model must implement a predict_proba method.
    calibrator = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv="prefit")
    calibrator.fit(X_cal_scaled, y_cal)
    print("Calibration complete. Calibrator has been fitted on the calibration dataset.")
    
    # 6. Optionally plot a calibration curve for binary classification.
    if len(label_encoder.classes_) == 2:
        # For binary classification, plot calibration curve using class index 1.
        prob_pos = calibrator.predict_proba(X_cal_scaled)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_cal, prob_pos, n_bins=10)
        
        plt.figure()
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.tight_layout()
        cal_plot_path = os.path.join(replication_folder, "calibration_curve.png")
        plt.savefig(cal_plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved calibration curve plot to '{cal_plot_path}'")
    else:
        print("Calibration curve plot is available only for binary classification. Skipping plot for multiclass.")

    # 7. Save the calibrated model for later use in inference.
    output_path = os.path.join(replication_folder, "calibrated_model.joblib")
    joblib.dump(calibrator, output_path)
    print(f"Calibrated model saved to '{output_path}'")

if __name__ == "__main__":
    main()
