import os
import polars as pl
import numpy as np
import torch
import joblib
import json

from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn as nn

OUTPUT_DIR = "outputs"  # same folder where training artifacts are stored

class ConfigurableNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=1, dropout=0.0):
        super(ConfigurableNN, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PyTorchNNClassifierWithVal(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=32, learning_rate=1e-3, batch_size=32,
                 epochs=10, num_layers=1, dropout=0.0, verbose=True):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_layers = num_layers
        self.dropout = dropout
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.input_dim_ = None
        self.output_dim_ = None

    def fit(self, X, y):
        # Dummy fit to initialize the model layers
        self.input_dim_ = X.shape[1]
        # If you have multiple classes, you'd set self.output_dim_ to the # of classes
        # For inference, we won't actually train, so let's just set it to e.g. 10 or something.
        # But ideally, you'd know how many classes from your training set or label encoder.
        self.output_dim_ = 10  # or pass from outside if needed

        self.model_ = ConfigurableNN(
            input_dim=self.input_dim_,
            hidden_size=self.hidden_size,
            output_dim=self.output_dim_,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        return self

    def predict(self, X):
        self.model_.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model_(X_torch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        self.model_.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model_(X_torch)
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

def load_best_model_and_predict(new_csv_path):
    """
    :param new_csv_path: Path to a new CSV with at least the columns used in training
    :return: predictions, predicted probabilities
    """

    # 1. Load the final feature list from "features_used.json"
    features_json_path = os.path.join(OUTPUT_DIR, "features_used.json")
    with open(features_json_path, "r") as f:
        features_used = json.load(f)

    # 2. Read the new CSV with polars, then select only the columns used
    new_df = pl.read_csv(new_csv_path)
    # You might have extra columns, or a 'Label' column. We'll ignore those:
    new_df = new_df.select(features_used)

    # Convert to numpy
    X_new = new_df.to_numpy()

    # 3. Load the trained scaler
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)
    X_new_scaled = scaler.transform(X_new)

    # 4. Load best hyperparams from "metrics.json"
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "r") as f:
        metrics_dict = json.load(f)
    best_params = metrics_dict["optuna_best_params"]

    # We don't need many epochs or training for inference
    best_params["epochs"] = 1
    best_params["verbose"] = False

    # 5. Create classifier with the same hyperparams
    model_inference = PyTorchNNClassifierWithVal(**best_params)

    # 6. Initialize the model with a dummy fit
    #    Just to ensure the layer shapes are created
    dummy_y = np.zeros(X_new_scaled.shape[0], dtype=np.int64)  # or however many classes you have
    model_inference.fit(X_new_scaled, dummy_y)

    # 7. Load the trained model weights
    model_path = os.path.join(OUTPUT_DIR, "best_model_state.pth")
    state_dict = torch.load(model_path)
    model_inference.model_.load_state_dict(state_dict)

    # 8. Predict
    preds = model_inference.predict(X_new_scaled)
    probs = model_inference.predict_proba(X_new_scaled)

    return preds, probs

if __name__ == "__main__":
    # Example usage
    new_data_path = "new_species_data.csv"  # or something relevant
    preds, probs = load_best_model_and_predict(new_data_path)

    print("Predictions (integer-encoded):", preds)
    print("Probabilities:", probs)
