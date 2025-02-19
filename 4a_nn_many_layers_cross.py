import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import json
import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
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
# 0. LOGGING UTILITY FOR TIMESTAMPS
# ---------------------------------------------------------
log_steps = []

def log_message(message):
    """Append a timestamped message to log_steps."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# ---------------------------------------------------------
# 1. READING & SUBSAMPLING CSV FILES
# ---------------------------------------------------------
log_message("Reading CSV files and subsampling...")
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
log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. FILTERING FEATURES BASED ON MAD
# ---------------------------------------------------------
log_message("Filtering numeric features based on MAD...")
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

log_message(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. FEATURE SCALING & TRAIN/TEST SPLIT
# ---------------------------------------------------------
log_message("Splitting into train/test and scaling features...")
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

log_message(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

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

        # Lists to store the per-epoch losses
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
        if self.verbose:
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
        """By default, returns accuracy."""
        preds = self.predict(X)
        return accuracy_score(y, preds)

# ---------------------------------------------------------
# 6. GRID SEARCH FOR DEEPER NN
# ---------------------------------------------------------
log_message("=== Neural Network: Grid Search ===")
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
    verbose=2,
    return_train_score=True
)

log_message("Starting GridSearchCV for configurable PyTorch NN...")
grid_nn.fit(X_train_scaled, y_train)
log_message("Neural Network Grid Search complete.")

best_params = grid_nn.best_params_
log_message(f"Best NN Parameters: {best_params}")

# ---------------------------------------------------------
# 6.1. PRINT & SAVE NIC FOR EACH MODEL IN GRID SEARCH
# ---------------------------------------------------------
# We'll define a simple "NIC" measure as:
# NIC = 2 * (number_of_parameters) - 2 * (log_likelihood)
# But we only have accuracies. So as a toy example:
# We'll define NIC ~ (#params) - (some function of accuracy).
# This is *not* a formal AIC/BIC, just a demonstration!

def approximate_nic(param_count, accuracy):
    """
    A toy "Network Information Criterion" measure.
    Higher accuracy -> lower NIC (better).
    Higher parameter count -> higher NIC (worse).
    """
    # In real life, you'd use negative log-likelihood. We'll do a linear shift for demonstration.
    return param_count - 1000.0 * accuracy

all_results = []
for i in range(len(grid_nn.cv_results_["params"])):
    params_i = grid_nn.cv_results_["params"][i]

    # We don't have the direct model object, but we can guess param_count if we build a temporary model:
    temp_model = ConfigurableNN(
        input_dim=X_train.shape[1],
        hidden_size=params_i["hidden_size"],
        output_dim=len(np.unique(y_train)),
        num_layers=params_i["num_layers"]
    )
    param_count_i = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)

    # Mean cross-val accuracy
    mean_val_accuracy = grid_nn.cv_results_["mean_test_score"][i]

    # Toy NIC
    nic_i = approximate_nic(param_count_i, mean_val_accuracy)

    row = {
        "params": params_i,
        "mean_train_accuracy": grid_nn.cv_results_["mean_train_score"][i],
        "mean_val_accuracy": mean_val_accuracy,
        "param_count": param_count_i,
        "approx_NIC": nic_i
    }
    all_results.append(row)

nic_df = pd.DataFrame(all_results)
log_message("\nNIC Results for Each Grid-Search Model:")
log_message(nic_df.to_string())

nic_df.to_csv("gridsearch_nic_results.csv", index=False)
log_message("Saved NIC results to 'gridsearch_nic_results.csv'.")

# ---------------------------------------------------------
# 7. FINAL REFIT WITH 100 EPOCHS
# ---------------------------------------------------------
log_message("\nRe-fitting best NN model on entire training set with 100 epochs...")

best_nn = grid_nn.best_estimator_
best_nn.epochs = 100
best_nn.verbose = True
best_nn.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 7.1. 5-FOLD CROSS-VALIDATION WITH THE BEST HYPERPARAMS FOR UNCERTAINTY
#     (TRACK TRAINING LOSS & VALIDATION LOSS)
# ---------------------------------------------------------
log_message("Performing 5-fold cross-validation with the best hyperparams for final uncertainty estimates...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_train_losses = []
fold_val_losses = []

# We need a manual training loop so we can store train & val loss every epoch.
class PyTorchNNClassifierWithVal(PyTorchNNClassifier):
    def fit(self, X, y, X_val=None, y_val=None):
        # Build the model
        self.input_dim_ = X.shape[1]
        self.output_dim_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        self.model_ = ConfigurableNN(
            input_dim=self.input_dim_,
            hidden_size=self.hidden_size,
            output_dim=self.output_dim_,
            num_layers=self.num_layers
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        X_torch = torch.from_numpy(X).float().to(self.device)
        y_torch = torch.from_numpy(y).long().to(self.device)

        dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            X_val_torch = torch.from_numpy(X_val).float().to(self.device)
            y_val_torch = torch.from_numpy(y_val).long().to(self.device)

        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.epochs):
            # ----- Training -----
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

            # ----- Validation -----
            if X_val is not None and y_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val_torch)
                    val_loss = criterion(val_outputs, y_val_torch).item()
            else:
                val_loss = np.nan

            self.val_losses_.append(val_loss)

        return self

    def get_train_val_losses(self):
        return self.train_losses_, self.val_losses_

# Use the best hyperparams but with 100 epochs
best_params_for_cv = best_nn.get_params()
best_params_for_cv["epochs"] = 100
best_params_for_cv["verbose"] = False  # we'll handle prints ourselves

fold_idx = 1
for train_index, val_index in kf.split(X_train_scaled, y_train):
    X_tr_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_tr_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Build new classifier with best hyperparams
    clf_fold = PyTorchNNClassifierWithVal(**best_params_for_cv)
    clf_fold.fit(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold)

    tr_losses, val_losses = clf_fold.get_train_val_losses()
    fold_train_losses.append(tr_losses)
    fold_val_losses.append(val_losses)

    log_message(f"Fold {fold_idx} complete.")
    fold_idx += 1

# ---------------------------------------------------------
# 7.2. PLOT TRAIN/VAL LOSS PER FOLD & CONFIDENCE INTERVAL
# ---------------------------------------------------------
epochs_range = np.arange(1, best_params_for_cv["epochs"] + 1)

# Plot: individual lines
plt.figure(figsize=(10, 6))

for i in range(len(fold_train_losses)):
    plt.plot(epochs_range, fold_train_losses[i], 
             label=f"Train Loss (Fold {i+1})", alpha=0.6)
    plt.plot(epochs_range, fold_val_losses[i], 
             label=f"Val Loss (Fold {i+1})", alpha=0.6, linestyle='--')

plt.title("Train & Validation Loss Curves (All Folds)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("cv_all_folds_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved plot of train/val loss per fold to 'cv_all_folds_loss.png'")

# Aggregate with mean ± std (or any CI)
train_losses_arr = np.array(fold_train_losses)  # shape: (5, epochs)
val_losses_arr   = np.array(fold_val_losses)    # shape: (5, epochs)

mean_train = train_losses_arr.mean(axis=0)
std_train  = train_losses_arr.std(axis=0)

mean_val = val_losses_arr.mean(axis=0)
std_val  = val_losses_arr.std(axis=0)

plt.figure(figsize=(8, 5))

plt.plot(epochs_range, mean_train, label="Mean Train Loss", color='blue')
plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train,
                 alpha=0.2, color='blue')

plt.plot(epochs_range, mean_val, label="Mean Val Loss", color='orange')
plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val,
                 alpha=0.2, color='orange')

plt.title("Mean ± 1 Std Train/Val Loss (5-fold CV)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("cv_mean_confidence_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved mean ± std train/val loss plot to 'cv_mean_confidence_loss.png'")

# ---------------------------------------------------------
# 8. EVALUATE ON TEST SET & SAVE METRICS/MODEL
# ---------------------------------------------------------
log_message("Evaluating final model on test set...")
y_pred = best_nn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
log_message(f"Final Model Accuracy on Test Set: {accuracy:.4f}")

class_report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)
log_message("\nClassification Report:")
log_message(class_report)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Final Model)")
plt.tight_layout()
plt.savefig("confusion_matrix_nn.png", bbox_inches='tight')
plt.close()
log_message("Saved confusion matrix to 'confusion_matrix_nn.png'.")

metrics_dict = {
    "accuracy": float(accuracy),
    "classification_report": class_report
}

with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
log_message("Saved metrics to 'metrics.json'.")

# Save the final model's weights
torch.save(best_nn.model_.state_dict(), "best_model_state.pth")
log_message("Saved final model's state_dict to 'best_model_state.pth'.")

# (Optional) Save the entire estimator info: weights + hyperparameters
best_estimator_data = {
    "state_dict": best_nn.model_.state_dict(),
    "params": best_nn.get_params()
}
torch.save(best_estimator_data, "best_estimator.pth")
log_message("Saved final estimator (weights + params) to 'best_estimator.pth'.")

# ---------------------------------------------------------
# 9. SAVE STEP-BY-STEP LOG WITH TIMESTAMPS
# ---------------------------------------------------------
with open("log_steps.json", "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log with timestamps to 'log_steps.json'.")

log_message("All done!")
