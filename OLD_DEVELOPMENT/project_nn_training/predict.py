# predict.py
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from logging_utils import get_logger
from pipeline import build_semi_supervised_pipeline

def main():
    parser = argparse.ArgumentParser(description="Predict classes for a new dataset.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV (with same features).")
    parser.add_argument('--output', type=str, required=True, help="Path to output CSV with predictions.")
    parser.add_argument('--model_file', type=str, default="best_estimator.pth",
                        help="Path to the saved PyTorch model data.")
    args = parser.parse_args()

    logger = get_logger(log_file='prediction.log')
    logger.info("===== Starting Prediction Script =====")

    # 1. Load the saved model info
    logger.info(f"Loading model data from {args.model_file}")
    best_estimator_data = torch.load(args.model_file, map_location="cpu")

    # 2. Rebuild the pipeline
    from sklearn.preprocessing import LabelEncoder

    pipeline = build_semi_supervised_pipeline(verbose=False)
    pipeline.set_params(**best_estimator_data["params"])

    # Load PyTorch state_dict into the pipeline's NN model
    pipeline.base_estimator.named_steps['nn'].model_.load_state_dict(best_estimator_data["state_dict"])

    # Rebuild the label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(best_estimator_data["label_encoder_classes"])

    # 3. Read the new dataset
    feature_names = best_estimator_data["feature_names"]
    df_new = pd.read_csv(args.input)
    # Ensure the new data has exactly the same columns as training
    # If your real data doesn't, you must adapt or check for missing columns
    X_new = df_new[feature_names].values

    # 4. Predict
    logger.info("Generating predictions...")
    y_pred_numeric = pipeline.predict(X_new)
    y_pred_labels = label_encoder.inverse_transform(y_pred_numeric)

    # 5. Save predictions
    df_new["PredictedLabel"] = y_pred_labels
    df_new.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")

    logger.info("===== Prediction Complete! =====")


if __name__ == "__main__":
    main()
