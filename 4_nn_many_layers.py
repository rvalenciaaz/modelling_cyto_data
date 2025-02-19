import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import json

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, ClassifierMixin

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Fix seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------
# 1. READING & SUBSAMPLING CSV FILES
# ---------------------------------------------------------
print("Reading CSV files and subsampling...")
csv_files = glob.glob("species*.csv")  # Adjust pattern to your needs
df_list = []

for file_path in csv_files:
    temp_df = pd.read_csv(file_path)
    # Optional subsample
    temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)
    label = file_path.split('.')[0]  # e.g. "species1.csv" -> "species1"
    temp_df['Label'] = label
    df_list.append(temp_df)

combined_df = pd.concat(df_list, ignore_index=True)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)
print(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. FILTERING FEATURES BASED ON MAD
# ---------------------------------------------------------
print("Filtering numeric features based on MAD...")
numerical_data = combined_df.select_dtypes(include=[np.number])
cv_results = {}
for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val = numerical_data[col].std()
    cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    mad = median_abs_deviation(numerical_data[col].values, scale='normal')
    cv_results[col] = [col, cv, mad]

cv_df = pd.DataFrame(cv_results.values(), columns=['Feature', 'CV', 'MAD'])

MAD_THRESHOLD = 5
features_to_keep = cv_df.loc[cv_df["MAD"] >= MAD_THRESHOLD, "Feature"].tolist()

cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()

print(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. FEATURE SCALING & TRAIN/TEST SPLIT
# ---------------------------------------------------------
print("Splitting into train/test and scaling features...")
X = final_df.drop(columns=["Label"])
y = final_df["Label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ---------------------------------------------------------
# 4. DEFINING A CONFIGURABLE DEEPER MODEL
# ---------------------------------------------------------
class ConfigurableNN(nn.Module):
    """
    A feedforward neural network with a variable number of hidden layers.
    Each hidden layer has 'hidden_size' units, followed by ReLU.
    """
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=1):
        super(ConfigurableNN, self).__init__()
        layers = []
        # First layer (input -> hidden_size)
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Final layer to output
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 5. PYTORCH WRAPPER (with parameter counting)
# ---------------------------------------------------------
class PyTorchNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn style wrapper for a configurable PyTorch feedforward network.
    """
    def __init__(self, hidden_size=32, learning_rate=1e-3,
                 batch_size=32, epochs=10, num_layers=1, verbose=False):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_layers = num_layers
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Using device: {self.device}")

        # Will be set after fitting:
        self.input_dim_ = None
        self.output_dim_ = None
        self.model_ = None
        self.classes_ = None
        self.train_losses_ = []

    def fit(self, X, y):
        self.input_dim_ = X.shape[1]
        self.output_dim_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        # Build the model
        self.model_ = ConfigurableNN(
            input_dim=self.input_dim_,
            hidden_size=self.hidden_size,
            output_dim=self.output_dim_,
            num_layers=self.num_layers
        ).to(self.device)

        # Count trainable parameters
        param_count = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        print(f"[INFO] Building model with {self.num_layers} hidden layer(s); "
              f"hidden_size={self.hidden_size}. "
              f"Total trainable parameters: {param_count}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        X_torch = torch.from_numpy(X).float().to(self.device)
        y_torch = torch.from_numpy(y).long().to(self.device)

        dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.train_losses_.clear()
        self.model_.train()
        for epoch in range(self.epochs):
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

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_epoch_loss:.4f}")

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

# ---------------------------------------------------------
# 6. GRID SEARCH FOR DEEPER NN
# ---------------------------------------------------------
print("\n=== Neural Network: Grid Search ===")
nn_estimator = PyTorchNNClassifier(verbose=False)

param_grid_nn = {
    "hidden_size":   [64, 128],
    "num_layers":    [1, 2, 3],   # Test various depths
    "learning_rate": [1e-3, 1e-4],
    "batch_size":    [64, 128],
    "epochs":        [30]     # fewer epochs for grid search
}

grid_nn = GridSearchCV(
    estimator=nn_estimator,
    param_grid=param_grid_nn,
    cv=3,
    scoring='accuracy',
    n_jobs=1,  # safer for PyTorch
    verbose=2
)

print("Starting GridSearchCV for configurable PyTorch NN...")
grid_nn.fit(X_train_scaled, y_train)
print("Neural Network Grid Search complete.")
print(f"Best NN Parameters: {grid_nn.best_params_}")

# ---------------------------------------------------------
# 7. FINAL REFIT WITH MORE EPOCHS
# ---------------------------------------------------------
print("\nRe-fitting best NN model on entire training set with 50 epochs...")

best_nn = grid_nn.best_estimator_
best_nn.epochs = 100
best_nn.verbose = True
best_nn.fit(X_train_scaled, y_train)

# Plot training loss
plt.figure()
plt.plot(range(1, best_nn.epochs + 1), best_nn.train_losses_, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Epoch (Final Model)")
plt.tight_layout()
plt.savefig("training_loss.png", bbox_inches='tight')
plt.close()
print("Final model training complete. Loss plot saved to 'training_loss.png'.")

# ---------------------------------------------------------
# 8. EVALUATE ON TEST SET & SAVE METRICS/MODEL
# ---------------------------------------------------------
print("\nEvaluating final model on test set...")
y_pred = best_nn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {accuracy:.4f}")

class_report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)
print("\nClassification Report:")
print(class_report)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Final Model)")
plt.tight_layout()
plt.savefig("confusion_matrix_nn.png", bbox_inches='tight')
plt.close()

print("Saving metrics to 'metrics.json' ...")
metrics_dict = {
    "accuracy": float(accuracy),
    "classification_report": class_report
}
with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

print("Saving final model's state_dict to 'best_model_state.pth' ...")
torch.save(best_nn.model_.state_dict(), "best_model_state.pth")

# (Optional) Save the entire estimator info: weights + hyperparameters
print("Saving final estimator (weights + params) to 'best_estimator.pth' ...")
best_estimator_data = {
    "state_dict": best_nn.model_.state_dict(),
    "params": best_nn.get_params()
}
torch.save(best_estimator_data, "best_estimator.pth")

print("\nAll done! Check 'metrics.json' for metrics, 'best_model_state.pth' or 'best_estimator.pth' for the model.")
