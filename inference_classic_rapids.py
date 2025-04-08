# inference.py

import os
import joblib
import pandas as pd
import numpy as np
import cupy as cp
from xgboost import XGBClassifier
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC

def load_artifacts(output_dir="outputs"):
    """
    Load inference artifacts saved by the training script:
      - best_randomforest_model.pkl
      - best_logisticregression_model.pkl
      - best_xgboost_model.pkl
      - best_svm_model.pkl
      - scaler.pkl
      - label_encoder.pkl
      - features_to_keep.csv

    Returns a dictionary with the models, scaler, label_encoder, and features list.
    """
    artifacts = {}

    # Paths to models
    rf_path  = os.path.join(output_dir, "best_randomforest_model.pkl")
    lr_path  = os.path.join(output_dir, "best_logisticregression_model.pkl")
    xgb_path = os.path.join(output_dir, "best_xgboost_model.pkl")
    svm_path = os.path.join(output_dir, "best_svm_model.pkl")

    # Load each model if it exists; None if not found
    artifacts["RandomForest"] = joblib.load(rf_path) if os.path.exists(rf_path) else None
    artifacts["LogisticRegression"] = joblib.load(lr_path) if os.path.exists(lr_path) else None
    artifacts["XGBoost"] = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    artifacts["SVM"] = joblib.load(svm_path) if os.path.exists(svm_path) else None

    # Scaler & LabelEncoder
    artifacts["scaler"] = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    artifacts["label_encoder"] = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))

    # Features list
    feat_path = os.path.join(output_dir, "features_to_keep.csv")
    features_df = pd.read_csv(feat_path)
    artifacts["features_to_keep"] = features_df["Feature"].tolist()

    return artifacts

def predict_with_models(
    input_csv="new_data.csv",
    output_csv="inference_predictions.csv",
    output_dir="outputs"
):
    """
    Given a new CSV containing feature columns, this function:
      1. Loads the trained models, scaler, label_encoder, and features list.
      2. Reads the new data, checks if all required features are present.
      3. Scales the new data using the saved scaler.
      4. Generates predictions with each model.
      5. Inversely transforms the numeric predictions using label_encoder.
      6. Writes a new CSV with added columns: "Pred_{ModelName}"

    Args:
        input_csv (str): Path to the new data CSV.
        output_csv (str): Path to save the CSV containing predictions.
        output_dir (str): Directory containing the saved artifacts.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"ERROR: '{input_csv}' not found. Provide a valid CSV for inference."
        )

    artifacts = load_artifacts(output_dir=output_dir)
    models = {
        "RandomForest": artifacts["RandomForest"],
        "LogisticRegression": artifacts["LogisticRegression"],
        "XGBoost": artifacts["XGBoost"],
        "SVM": artifacts["SVM"]
    }
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    features_to_keep = artifacts["features_to_keep"]

    # Load new data
    new_data_df = pd.read_csv(input_csv)
    print(f"Loaded new data from '{input_csv}', shape={new_data_df.shape}")

    # Ensure all required features are present
    missing_cols = set(features_to_keep) - set(new_data_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in new data: {missing_cols}")

    # Scale the features
    X_new_cpu = new_data_df[features_to_keep].to_numpy()
    X_new_scaled_cpu = scaler.transform(X_new_cpu)
    X_new_scaled_gpu = cp.asarray(X_new_scaled_cpu)

    # Predict with each model
    predictions = {}
    for model_name, model in models.items():
        if model is None:
            predictions[model_name] = None
            continue

        if model_name == "XGBoost":
            # XGBoost GPU model
            y_pred_gpu = model.predict(X_new_scaled_gpu)
            y_pred = cp.asnumpy(y_pred_gpu)
        else:
            # cuml CPU model
            y_pred = model.predict(X_new_scaled_cpu)

        # Convert numeric predictions back to original label strings
        y_pred_labels = label_encoder.inverse_transform(y_pred.astype(int))
        predictions[model_name] = y_pred_labels

    # Add prediction columns to the output dataframe
    output_df = new_data_df.copy()
    for model_name, y_pred_labels in predictions.items():
        if y_pred_labels is None:
            output_df[f"Pred_{model_name}"] = "ModelNotFound"
        else:
            output_df[f"Pred_{model_name}"] = y_pred_labels

    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"Saved inference predictions to '{output_csv}'.")

def main_inference():
    """
    Simple main function to demonstrate usage of predict_with_models().
    """
    input_csv = "new_data.csv"
    output_csv = "inference_predictions.csv"
    output_dir = "outputs"

    predict_with_models(
        input_csv=input_csv,
        output_csv=output_csv,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main_inference()
