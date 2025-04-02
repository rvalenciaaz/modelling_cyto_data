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
      - categorical_cols.json (list of categorical feature names)
      - numeric_cols.json (list of numeric feature names)
      - main_label_encoder.pkl (for decoding predicted labels)
      - best_params_tabtransformer.json (for model architecture)
      - categorical_label_encoders.pkl (dict mapping col name to fitted LabelEncoder)
      - tabtransformer_model_state.pth (the trained model state dict)
    """
    scaler = joblib.load(os.path.join(artifact_dir, "scaler.pkl"))
    
    with open(os.path.join(artifact_dir, "categorical_cols.json"), "r") as f:
        categorical_cols = json.load(f)
    
    with open(os.path.join(artifact_dir, "numeric_cols.json"), "r") as f:
        numeric_cols = json.load(f)
    
    main_label_encoder = joblib.load(os.path.join(artifact_dir, "main_label_encoder.pkl"))
    best_params = json.load(open(os.path.join(artifact_dir, "best_params_tabtransformer.json")))
    cat_label_encoders = joblib.load(os.path.join(artifact_dir, "categorical_label_encoders.pkl"))
    
    return scaler, categorical_cols, numeric_cols, main_label_encoder, best_params, cat_label_encoders

def create_model(best_params, categorical_cardinalities):
    """
    Instantiates the TabTransformerClassifierWithVal model using the best hyperparameters.
    A dummy forward pass is used to initialize internal dimensions.
    """
    model = TabTransformerClassifierWithVal(
        transformer_dim=best_params["transformer_dim"],
        depth=best_params["depth"],
        heads=best_params["heads"],
        dim_forward=best_params["dim_forward"],
        dropout=best_params["dropout"],
        mlp_hidden_dims=best_params["mlp_hidden_dims"],
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        epochs=0,  # No training epochs needed for inference
        verbose=False
    )
    # Create dummy inputs to initialize the model
    n_cat = len(categorical_cardinalities)
    dummy_cat = np.zeros((1, n_cat), dtype=np.int64) if n_cat > 0 else np.empty((1, 0), dtype=np.int64)
    n_num = best_params.get("num_continuous", 0)
    dummy_num = np.zeros((1, n_num), dtype=np.float32) if n_num > 0 else np.empty((1, 0), dtype=np.float32)
    model.fit(dummy_cat, dummy_num, np.array([0]))
    return model

def predict_new_data(new_data_csv, artifact_dir="artifacts", output_csv="inference_predictions.csv"):
    """
    Loads artifacts, processes new CSV data, performs predictions with the trained model,
    and saves a CSV with the predicted labels.
    """
    # Load artifacts
    scaler, categorical_cols, numeric_cols, main_label_encoder, best_params, cat_label_encoders = load_artifacts(artifact_dir)
    
    # Load new data
    df = pd.read_csv(new_data_csv)
    
    # Check that required columns exist
    missing_cat = set(categorical_cols) - set(df.columns)
    missing_num = set(numeric_cols) - set(df.columns)
    if missing_cat or missing_num:
        raise ValueError(f"Missing required columns. Missing categorical: {missing_cat}, numeric: {missing_num}")
    
    # Process categorical features: apply saved label encoders
    X_categ = df[categorical_cols].copy()
    for col in categorical_cols:
        le = cat_label_encoders[col]
        X_categ[col] = le.transform(X_categ[col].astype(str))
    X_categ = X_categ.values
    
    # Process numeric features: scale them
    if numeric_cols:
        X_numeric = df[numeric_cols].values
        X_numeric = scaler.transform(X_numeric)
    else:
        X_numeric = np.empty((len(df), 0))
    
    # Compute categorical cardinalities from the saved label encoders
    categorical_cardinalities = []
    for col in categorical_cols:
        le = cat_label_encoders[col]
        categorical_cardinalities.append(int(len(le.classes_)))
    
    # Save the number of continuous features in best_params (for model instantiation)
    best_params["num_continuous"] = len(numeric_cols)
    
    # Instantiate the model and load its state
    model = create_model(best_params, categorical_cardinalities)
    state_dict = torch.load(os.path.join(artifact_dir, "tabtransformer_model_state.pth"), map_location=torch.device("cpu"))
    model.model_.load_state_dict(state_dict)
    model.model_.eval()
    
    # Perform prediction
    preds = model.predict(X_categ, X_numeric)
    pred_labels = main_label_encoder.inverse_transform(preds)
    
    # Save predictions to CSV
    df["PredictedLabel"] = pred_labels
    df.to_csv(output_csv, index=False)
    print(f"Saved inference predictions to '{output_csv}'.")

if __name__ == "__main__":
    # Example: Predict on new data from 'new_species_data.csv'
    predict_new_data("new_species_data.csv")
