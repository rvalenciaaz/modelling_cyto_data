# inference.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from model_utils import TabTransformerClassifierWithVal

def load_artifacts(artifact_dir="artifacts"):
    """
    Loads required artifacts for inference:
      - scaler.pkl
      - numeric_cols.json
      - main_label_encoder.pkl
      - best_params_tabtransformer.json
      - tabtransformer_model_state.pth
    """
    scaler_path       = os.path.join(artifact_dir, "scaler.pkl")
    numeric_cols_path = os.path.join(artifact_dir, "numeric_cols.json")
    label_enc_path    = os.path.join(artifact_dir, "main_label_encoder.pkl")
    params_path       = os.path.join(artifact_dir, "best_params_tabtransformer.json")
    model_state_path  = os.path.join(artifact_dir, "tabtransformer_model_state.pth")

    scaler             = joblib.load(scaler_path)
    numeric_cols       = json.load(open(numeric_cols_path, "r"))
    main_label_encoder = joblib.load(label_enc_path)
    best_params        = json.load(open(params_path, "r"))
    # model_state_path will be loaded separately in create_model()

    return scaler, numeric_cols, main_label_encoder, best_params, model_state_path

def create_model(best_params, n_numeric_features):
    """
    Instantiates the TabTransformerClassifierWithVal model using the best hyperparameters.
    A dummy forward pass is used to initialize internal dimensions properly.
    
    Because we have no categorical features, n_cat = 0.
    """
    model = TabTransformerClassifierWithVal(
        transformer_dim=best_params["transformer_dim"],
        depth=best_params["depth"],
        heads=best_params["heads"],
        dim_forward=best_params["dim_forward"],
        dropout=best_params["dropout"],
        mlp_hidden_dims=[
            best_params["mlp_hidden_dim1"],
            best_params["mlp_hidden_dim2"]
        ],
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        epochs=0,  # No training needed for inference
        verbose=False
    )

    # Create dummy inputs to initialize the model
    n_cat = 0  # No categorical features
    dummy_cat = np.empty((1, n_cat), dtype=np.int64)       # shape: (1, 0)
    dummy_num = np.zeros((1, n_numeric_features), dtype=np.float32) if n_numeric_features > 0 else np.empty((1, 0), dtype=np.float32)

    # Perform a dummy fit() call to initialize model weights
    model.fit(dummy_cat, dummy_num, np.array([0]))

    return model

def predict_new_data(new_data_csv, artifact_dir="artifacts", output_csv="inference_predictions.csv"):
    """
    Loads artifacts, processes new CSV data, performs predictions with the trained model,
    and saves a CSV with the predicted labels.
    """
    # 1. Load artifacts
    scaler, numeric_cols, main_label_encoder, best_params, model_state_path = load_artifacts(artifact_dir)

    # 2. Load new data
    df = pd.read_csv(new_data_csv)

    # Ensure all required numeric columns are present
    missing_num = set(numeric_cols) - set(df.columns)
    if missing_num:
        raise ValueError(f"Missing required numeric columns: {missing_num}")

    # 3. Extract and scale numeric features
    X_numeric = df[numeric_cols].values
    X_numeric = scaler.transform(X_numeric)

    # For TabTransformer, we supply X_categ = empty array
    X_categ = np.empty((len(df), 0))

    # 4. Instantiate the model and load state
    model = create_model(best_params, n_numeric_features=len(numeric_cols))
    state_dict = torch.load(model_state_path, map_location=torch.device("cpu"))
    model.model_.load_state_dict(state_dict)
    model.model_.eval()

    # 5. Perform prediction
    preds = model.predict(X_categ, X_numeric)
    pred_labels = main_label_encoder.inverse_transform(preds)

    # 6. Save predictions to CSV
    df["PredictedLabel"] = pred_labels
    df.to_csv(output_csv, index=False)
    print(f"Saved inference predictions to '{output_csv}'.")

if __name__ == "__main__":
    # Example usage:
    predict_new_data("new_species_data.csv")
