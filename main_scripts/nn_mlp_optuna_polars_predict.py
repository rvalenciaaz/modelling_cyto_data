import polars as pl
import numpy as np
import torch
import joblib
import json

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import torch.nn as nn

# Same network definition as training
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

# Minimal classifier to host the loaded PyTorch model
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
        # Dummy fit to initialize the model with the correct shapes
        self.input_dim_ = X.shape[1]
        self.output_dim_ = len(np.unique(y))

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

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

def load_best_model_and_predict(new_csv_path, feature_list=None):
    """
    :param new_csv_path: Path to a new CSV with the same schema as training
    :param feature_list: List of columns (in the correct order) that match training
    :return: predictions, predicted probabilities
    """
    # 1. Load best hyperparams from metrics.json
    with open("metrics.json", "r") as f:
        metrics_dict = json.load(f)
    best_params = metrics_dict["optuna_best_params"]

    # We don't need many epochs for inference:
    best_params["epochs"] = 1
    best_params["verbose"] = False

    # 2. Create classifier with the same hyperparams
    model_inference = PyTorchNNClassifierWithVal(**best_params)

    # 3. Load the *trained* scaler
    scaler = joblib.load("scaler.joblib")

    # 4. Optionally, load the label encoder if you want to decode predictions
    #    e.g.: label_encoder = joblib.load("label_encoder.joblib")

    # 5. Read & select relevant columns in the correct order
    new_df = pl.read_csv(new_csv_path)
    # Ensure it has the same features as training:
    if feature_list:
        new_df = new_df.select(feature_list)
    X_new = new_df.to_numpy()

    # 6. Scale with the *training* scaler
    X_new_scaled = scaler.transform(X_new)

    # 7. We must do a dummy "fit" on the new classifier to initialize the model shape
    dummy_y = np.zeros(X_new_scaled.shape[0], dtype=np.int64)  # dummy, content doesn't matter
    model_inference.fit(X_new_scaled, dummy_y)

    # 8. Load the trained weights
    state_dict = torch.load("best_model_state.pth")
    model_inference.model_.load_state_dict(state_dict)

    # 9. Predict
    preds = model_inference.predict(X_new_scaled)
    probs = model_inference.predict_proba(X_new_scaled)

    # If you want to decode labels:
    # original_labels = label_encoder.inverse_transform(preds)

    return preds, probs


if __name__ == "__main__":
    # EXAMPLE usage
    new_data_path = "new_species_data.csv"
    
    # If you know the final set of columns used in training, specify them:
    # e.g. feature_list = ["Length", "Width", "SepalWidth", "PetalLength", ...]
    # or None if the CSV already has the correct subset in the correct order
    feature_list = None

    predictions, probabilities = load_best_model_and_predict(new_data_path, feature_list)

    print("Integer-encoded predictions:", predictions)
    print("Predicted probabilities:\n", probabilities)
