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
    Loads:
      - best_{model_name}_model.pkl for each model
      - scaler.pkl
      - label_encoder.pkl
      - features_to_keep.csv
    from the specified output directory.
    Returns a dict of models, plus the scaler, label_encoder, and features list.
    """
    artifacts = {}

    # Models
    rf_path  = os.path.join(output_dir, "best_randomforest_model.pkl")
    lr_path  = os.path.join(output_dir, "best_logisticregression_model.pkl")
    xgb_path = os.path.join(output_dir, "best_xgboost_model.pkl")
    svm_path = os.path.join(output_dir, "best_svm_model.pkl")

    artifacts["RandomForest"] = joblib.load(rf_path) if os.path.exists(rf_path) else None
    artifacts["LogisticRegression"] = joblib.load(lr_path) if os.path.exists(lr_path) else None
    artifacts["XGBoost"] = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    artifacts["SVM"] = joblib.load(svm_path) if os.path.exists(svm_path) else None

    # Scaler & LabelEncoder
    artifacts["scaler"] = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    artifacts["label_encoder"] = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))

    # Features
    feat_path = os.path.join(output_dir, "features_to_keep.csv")
    features_df = pd.read_csv(feat_path)
    artifacts["features_to_keep"] = features_df["Feature"].tolist()

    return artifacts

def predict_with_models(input_csv="new_data.csv", output_csv="inference_predictions.csv", output_dir="outputs"):
    """
    Loads new unlabeled data, applies the saved scaler, and runs inference 
    on each of the four trained models (RF, LR, XGB, SVM).
    Saves a CSV with predictions from each model.
    """
    if not os.path.exists(input_csv):
        print(f"ERROR: '{input_csv}' not found. Provide a CSV with features for inference.")
        return

    artifacts = load_artifacts(output_dir)
    models = {
        "RandomForest": artifacts["RandomForest"],
        "LogisticRegression": artifacts["LogisticRegression"],
        "XGBoost": artifacts["XGBoost"],
        "SVM": artifacts["SVM"],
    }
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    features_to_keep = artifacts["features_to_keep"]

    new_data_df = pd.read_csv(input_csv)
    print(f"Loaded new data from '{input_csv}', shape={new_data_df.shape}")

    missing_cols = set(features_to_keep) - set(new_data_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in new data: {missing_cols}")

    X_new = new_data_df[features_to_keep].to_numpy()
    X_new_scaled_cpu = scaler.transform(X_new)
    X_new_scaled_gpu = cp.asarray(X_new_scaled_cpu)

    predictions = {}
    for m_name, model in models.items():
        if model is None:
            predictions[m_name] = None
            continue

        if m_name == "XGBoost":
            y_pred = model.predict(X_new_scaled_gpu)
            y_pred = cp.asnumpy(y_pred)
        else:
            y_pred = model.predict(X_new_scaled_cpu)

        y_pred_labels = label_encoder.inverse_transform(y_pred.astype(int))
        predictions[m_name] = y_pred_labels

    output_df = new_data_df.copy()
    for m_name, y_pred_labels in predictions.items():
        if y_pred_labels is not None:
            output_df[f"Pred_{m_name}"] = y_pred_labels
        else:
            output_df[f"Pred_{m_name}"] = "ModelNotFound"

    output_df.to_csv(output_csv, index=False)
    print(f"Saved inference predictions to '{output_csv}'.")

def main_inference():
    input_csv = "new_data.csv"
    output_csv = "inference_predictions.csv"
    output_dir = "outputs"

    predict_with_models(input_csv=input_csv, output_csv=output_csv, output_dir=output_dir)

if __name__ == "__main__":
    main_inference()
