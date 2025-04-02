# models.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional

class ConfigurableNN(nn.Module):
    """
    A feedforward network with 'num_layers' hidden layers,
    each with 'hidden_size' units and ReLU activation.
    """
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=1):
        super(ConfigurableNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PyTorchNNClassifierWithVal(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible PyTorch classifier with:
      - GPU support (if available)
      - optional validation & early stopping
    """
    def __init__(
        self,
        hidden_size=32,
        learning_rate=1e-3,
        batch_size=32,
        epochs=10,
        num_layers=1,
        verbose=True,
        early_stopping=False,
        patience=5
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_layers = num_layers
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim_ = None
        self.output_dim_ = None
        self.model_ = None
        self.classes_ = None
        self.train_losses_ = []
        self.val_losses_ = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "PyTorchNNClassifierWithVal":
        """
        Train the model on (X, y). If y has -1 for unlabeled, those rows are excluded
        from the loss. If X_val, y_val provided, track validation loss each epoch.
        """
        # Identify number of features and classes (excluding unlabeled = -1)
        self.input_dim_ = X.shape[1]
        y_labeled = y[y != -1]  # skip unlabeled
        self.classes_ = np.unique(y_labeled)
        self.output_dim_ = len(self.classes_)

        # Build model
        self.model_ = ConfigurableNN(
            input_dim=self.input_dim_,
            hidden_size=self.hidden_size,
            output_dim=self.output_dim_,
            num_layers=self.num_layers
        ).to(self.device)

        # Parameter count for info
        if self.verbose:
            param_count = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
            print(f"ConfigurableNN -> hidden_size={self.hidden_size}, layers={self.num_layers}, "
                  f"Trainable params={param_count}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        # Filter out unlabeled samples
        labeled_idx = np.where(y != -1)[0]
        X_labeled = X[labeled_idx]
        y_labeled = y[labeled_idx]

        # Convert to torch Tensors
        X_torch = torch.from_numpy(X_labeled).float().to(self.device)
        y_torch = torch.from_numpy(y_labeled).long().to(self.device)

        dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        has_val = (X_val is not None) and (y_val is not None)
        if has_val:
            X_val_torch = torch.from_numpy(X_val).float().to(self.device)
            y_val_torch = torch.from_numpy(y_val).long().to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0

        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.epochs):
            # Training
            self.model_.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            self.train_losses_.append(avg_epoch_loss)

            # Validation (if available)
            val_loss = np.nan
            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val_torch)
                    val_loss = criterion(val_outputs, y_val_torch).item()
            self.val_losses_.append(val_loss)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} -> Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping (if enabled)
            if self.early_stopping and has_val:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns hard class predictions.
        """
        self.model_.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model_(X_torch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability predictions for each class.
        """
        self.model_.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model_(X_torch)
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns accuracy on given data and labels.
        """
        from sklearn.metrics import accuracy_score
        preds = self.predict(X)
        return accuracy_score(y, preds)

    def get_train_val_losses(self):
        return self.train_losses_, self.val_losses_
