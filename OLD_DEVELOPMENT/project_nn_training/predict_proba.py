# predict_proba.py
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from logging_utils import get_logger
from pipeline import build_semi_supervised_pipeline

def main():
    parser = argparse.ArgumentParser(description="Predict class probabilities for a new dataset.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV (with same features).")
    parser.add_argument('--output', type=str, required=True, help="Path to output CSV with probabilities.")
    parser.add_argument('--model_file', type=str, default="best_estimator.pth",
                        help="Path to the saved PyTorch model data.")
    args = parser.parse_args()

    logger = get_logger(log_file='prediction.log')
    logger.info("===== Starting Probability Prediction Script =====")

    # 1. Load model data
    logger.info(f"Loading model data from {args.model_file}")
    best_estimator_data = torch.load(args.model_file, map_location="cpu")

    # 2. Rebuild the pipeline
    from sklearn.preprocessing import LabelEncoder

    pipeline = build_semi_supervised_pipeline(verbose=False)
    pipeline.set_params(**best_estimator_data["params"])

    pipeline.base_estimator.named_steps['nn'].model_.load_state_dict(best_estimator_data["state_dict"])

    # Rebuild label encoder to know class names
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(best_estimator_data["label_encoder_classes"])

    # 3. Read the new data
    feature_names = best_estimator_data["feature_names"]
    df_new = pd.read_csv(args.input)
    X_new = df_new[feature_names].values

    # 4. Predict probabilities
    logger.info("Generating probability predictions...")
    y_probas = pipeline.predict_proba(X_new)  # shape: (num_samples, num_classes)

    # 5. Save to CSV
    class_names = label_encoder.classes_
    proba_cols = [f"Prob_{cls}" for cls in class_names]
    proba_df = pd.DataFrame(y_probas, columns=proba_cols)
    df_out = pd.concat([df_new.reset_index(drop=True), proba_df], axis=1)
    df_out.to_csv(args.output, index=False)
    logger.info(f"Predicted probabilities saved to {args.output}")

    logger.info("===== Probability Prediction Complete! =====")


if __name__ == "__main__":
    main()
