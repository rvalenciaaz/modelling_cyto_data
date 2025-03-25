import glob
import numpy as np
import polars as pl
from scipy.stats import median_abs_deviation
import json
import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Sklearn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns  # (You can remove this if you don't need Seaborn features)

# Optuna import
import optuna

# Fix seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------
# 0. LOGGING UTILITY FOR TIMESTAMPS
# ---------------------------------------------------------
log_steps = []
def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# ---------------------------------------------------------
# 1. READING & SUBSAMPLING CSV FILES (using Polars)
# ---------------------------------------------------------
log_message("Reading CSV files and subsampling with Polars...")

csv_files = glob.glob("species*.csv")  # Will get a list of species*.csv filenames
df_list = []

for file_path in csv_files:
    temp_df = pl.read_csv(file_path)
    # Optional: subsample if desired (uncomment and adjust to your needs)
    # temp_df = temp_df.sample(n=min(temp_df.height, 10_000), seed=42)

    # Extract a label from the filename, e.g. "species1.csv" -> "species1"
    label = file_path.split('.')[0]
    # Add a column "Label" to the Polars DataFrame
    temp_df = temp_df.with_columns(pl.lit(label).alias("Label"))
    df_list.append(temp_df)

# Concatenate all species data
combined_df = pl.concat(df_list)

# Remove the prefix "species" from Label (e.g. "species1" -> "1")
combined_df = combined_df.with_columns(
    pl.col("Label").str.replace("species", "")
)

log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. FILTERING FEATURES BASED ON MAD (using Polars)
# ---------------------------------------------------------
log_message("Filtering numeric features based on MAD...")

# Identify numeric columns
numeric_dtypes = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
]
numerical_columns = [col for col, dtype in combined_df.schema.items() if dtype in numeric_dtypes]
numerical_data = combined_df.select(numerical_columns)

# Calculate Coefficient of Variation (CV) and MAD for each numeric column
cv_results = {}
for col in numerical_data.columns:
    values = numerical_data[col].to_numpy()
    mean_val = np.mean(values)
    std_val  = np.std(values)
    cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    mad = median_abs_deviation(values, scale='normal')
    cv_results[col] = [col, cv, mad]

# Convert cv_results into a Polars DataFrame
cv_df = pl.DataFrame(
    list(cv_results.values()),
    schema=["Feature", "CV", "MAD"]
)

MAD_THRESHOLD = 5
features_to_keep = (
    cv_df
    .filter(pl.col("MAD") >= MAD_THRESHOLD)
    .select("Feature")
    .to_series()
    .to_list()
)

cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df.select(cols_to_keep)

log_message(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT (RAW DATA; SCALING IS PART OF THE PIPELINE)
# ---------------------------------------------------------
log_message("Splitting data into training and test sets...")

# Convert to NumPy for scikit-learn and PyTorch
X = final_df.select(pl.all().exclude("Label")).to_numpy()
y = final_df.select("Label").to_numpy().ravel()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
log_message(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ---------------------------------------------------------
# 4. DEFINE THE CONFIGURABLE NN MODEL
# ---------------------------------------------------------
class ConfigurableNN(nn.Module):
    """
    A feedforward neural network with a variable number of hidden layers,
    a configurable hidden_size, and optional dropout after each hidden layer.
    """
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

# ---------------------------------------------------------
# 5. DEFINE A SINGLE PYTORCH CLASSIFIER THAT TRACKS VALIDATION LOSS
# ---------------------------------------------------------
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
        self.input_dim_ = None
        self.output_dim_ = None
        self.model_ = None
        self.classes_ = None
        self.train_losses_ = []
        self.val_losses_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.input_dim_ = X.shape[1]
        self.output_dim_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        # Build the model
        self.model_ = ConfigurableNN(
            input_dim=self.input_dim_,
            hidden_size=self.hidden_size,
            output_dim=self.output_dim_,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Print parameter count (optional)
        param_count = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        if self.verbose:
            print(f"Model with {self.num_layers} hidden layers, hidden_size={self.hidden_size}, "
                  f"dropout={self.dropout:.1f}, params={param_count}.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        # Prepare data for training
        X_torch = torch.from_numpy(X).float().to(self.device)
        y_torch = torch.from_numpy(y).long().to(self.device)
        dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.epochs):
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

            # Validation loss (if X_val provided)
            if (X_val is not None) and (y_val is not None):
                self.model_.eval()
                X_val_torch = torch.from_numpy(X_val).float().to(self.device)
                y_val_torch = torch.from_numpy(y_val).long().to(self.device)
                with torch.no_grad():
                    val_outputs = self.model_(X_val_torch)
                    val_loss = criterion(val_outputs, y_val_torch).item()
            else:
                val_loss = np.nan

            self.val_losses_.append(val_loss)

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_epoch_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")

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

    def get_train_val_losses(self):
        return self.train_losses_, self.val_losses_

# ---------------------------------------------------------
# 6. OPTUNA OBJECTIVE FUNCTION (HYPERPARAMETER SEARCH)
# ---------------------------------------------------------
def objective(trial):
    """
    Defines the hyperparameter search space and returns the average
    cross-validation accuracy for a given set of hyperparams.
    """
    # Define hyperparameter search space
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 3, 30)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Fewer epochs during the search
    epochs = 30

    # We'll do 3-fold cross-validation to get an average validation accuracy
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    accuracy_scores = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Scale the data within each fold
        scaler_fold = StandardScaler()
        X_tr_fold_scaled = scaler_fold.fit_transform(X_tr_fold)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold)

        # Build the classifier with trial's hyperparams
        clf = PyTorchNNClassifierWithVal(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            num_layers=num_layers,
            dropout=dropout,
            verbose=False  # Set False for faster runs
        )

        # Fit on the fold's training data, track validation
        clf.fit(X_tr_fold_scaled, y_tr_fold, X_val_fold_scaled, y_val_fold)

        # Evaluate on the fold's validation data
        preds = clf.predict(X_val_fold_scaled)
        fold_acc = accuracy_score(y_val_fold, preds)
        accuracy_scores.append(fold_acc)

    # Optuna tries to maximize this objective
    mean_val_acc = np.mean(accuracy_scores)
    return mean_val_acc

log_message("=== Starting Optuna hyperparameter optimization ===")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # adjust n_trials as needed

best_params = study.best_params
best_score = study.best_value
log_message(f"Optuna best params: {best_params}")
log_message(f"Optuna best CV accuracy: {best_score:.4f}")

# ---------------------------------------------------------
# 7. FINAL REFIT ON TRAINING SET WITH MORE EPOCHS
# ---------------------------------------------------------
log_message("Re-fitting with best hyperparameters on entire training set (50 epochs)...")
best_params_for_final = best_params.copy()
best_params_for_final["epochs"] = 50  # Increase epochs for final training
best_params_for_final["verbose"] = True

# Build a pipeline for final training on the entire training set
best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nn', PyTorchNNClassifierWithVal(**best_params_for_final))
])
best_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------
# 7.1. 5-FOLD CROSS-VALIDATION WITH BEST HYPERPARAMS
# ---------------------------------------------------------
log_message("Performing 5-fold CV with the best hyperparameters for uncertainty estimates...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_train_losses = []
fold_val_losses = []
fold_val_accuracies = []

# Extract the classifier's params for manual CV
final_nn_params = best_pipeline.named_steps['nn'].get_params()

fold_idx = 1
for train_index, val_index in kf.split(X_train, y_train):
    X_tr_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_tr_fold, y_val_fold = y_train[train_index], y_train[val_index]

    scaler_fold = StandardScaler()
    X_tr_fold_scaled = scaler_fold.fit_transform(X_tr_fold)
    X_val_fold_scaled = scaler_fold.transform(X_val_fold)

    # Rebuild the classifier for each fold
    clf_fold = PyTorchNNClassifierWithVal(**final_nn_params)
    # Fit with validation
    clf_fold.fit(X_tr_fold_scaled, y_tr_fold, X_val_fold_scaled, y_val_fold)

    tr_losses, val_losses = clf_fold.get_train_val_losses()
    fold_train_losses.append(tr_losses)
    fold_val_losses.append(val_losses)

    # Evaluate validation accuracy
    y_val_pred = clf_fold.predict(X_val_fold_scaled)
    fold_val_acc = accuracy_score(y_val_fold, y_val_pred)
    fold_val_accuracies.append(fold_val_acc)

    log_message(f"Fold {fold_idx} complete. Validation Accuracy: {fold_val_acc:.4f}")
    fold_idx += 1

mean_cv_val_acc = np.mean(fold_val_accuracies)
std_cv_val_acc = np.std(fold_val_accuracies)
log_message(f"5-Fold CV Validation Accuracy: Mean = {mean_cv_val_acc:.4f}, Std = {std_cv_val_acc:.4f}")

# Define the epoch range (all folds use the same number of epochs)
epochs_range = np.arange(1, final_nn_params["epochs"] + 1)

# -------------------------------
# SAVE DATA FOR PLOT REPRODUCTION
# -------------------------------
np.savez("cv_plot_data.npz",
         epochs_range=epochs_range,
         fold_train_losses=np.array(fold_train_losses),
         fold_val_losses=np.array(fold_val_losses),
         mean_train=np.array(fold_train_losses).mean(axis=0),
         std_train=np.array(fold_train_losses).std(axis=0),
         mean_val=np.array(fold_val_losses).mean(axis=0),
         std_val=np.array(fold_val_losses).std(axis=0),
         fold_val_accuracies=np.array(fold_val_accuracies))
log_message("Saved CV plot data to 'cv_plot_data.npz'.")

# Plot training/validation loss for each fold
plt.figure(figsize=(10, 6))
for i in range(len(fold_train_losses)):
    plt.plot(epochs_range, fold_train_losses[i], label=f"Train Loss (Fold {i+1})", alpha=0.6)
    plt.plot(epochs_range, fold_val_losses[i], label=f"Val Loss (Fold {i+1})", alpha=0.6, linestyle='--')
plt.title("Train & Validation Loss Curves (All Folds)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("cv_all_folds_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved plot of train/val loss per fold to 'cv_all_folds_loss.png'.")

# Plot mean ± std for train/val loss
train_losses_arr = np.array(fold_train_losses)
val_losses_arr   = np.array(fold_val_losses)
mean_train = train_losses_arr.mean(axis=0)
std_train  = train_losses_arr.std(axis=0)
mean_val   = val_losses_arr.mean(axis=0)
std_val    = val_losses_arr.std(axis=0)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, mean_train, label="Mean Train Loss")
plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train, alpha=0.2)
plt.plot(epochs_range, mean_val, label="Mean Val Loss")
plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val, alpha=0.2)
plt.title("Mean ± 1 Std Train/Val Loss (5-fold CV)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("cv_mean_confidence_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved mean ± std train/val loss plot to 'cv_mean_confidence_loss.png'.")

# ---------------------------------------------------------
# 8. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ---------------------------------------------------------
log_message("Evaluating final model on test set...")
y_pred = best_pipeline.predict(X_test)  # Pipeline applies scaling automatically
test_accuracy = accuracy_score(y_test, y_pred)
log_message(f"Final Model Accuracy on Test Set: {test_accuracy:.4f}")

class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
log_message("\nClassification Report:")
log_message(class_report)

# Save confusion matrix data
cm = confusion_matrix(y_test, y_pred)
np.save("confusion_matrix.npy", cm)
log_message("Saved confusion matrix data to 'confusion_matrix.npy'.")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Final Model)")
plt.tight_layout()
plt.savefig("confusion_matrix_nn.png", bbox_inches='tight')
plt.close()
log_message("Saved confusion matrix to 'confusion_matrix_nn.png'.")

# Save metrics
metrics_dict = {
    "test_accuracy": float(test_accuracy),
    "classification_report": class_report,
    "optuna_best_params": best_params,
    "optuna_best_cv_score": float(best_score),
    "cv_val_accuracies": [float(acc) for acc in fold_val_accuracies],
    "cv_val_accuracy_mean": float(mean_cv_val_acc),
    "cv_val_accuracy_std": float(std_cv_val_acc)
}
with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
log_message("Saved metrics to 'metrics.json'.")

# Save the final model state
torch.save(best_pipeline.named_steps['nn'].model_.state_dict(), "best_model_state.pth")
log_message("Saved final model's state_dict to 'best_model_state.pth'.")

best_estimator_data = {
    "state_dict": best_pipeline.named_steps['nn'].model_.state_dict(),
    "params": best_pipeline.named_steps['nn'].get_params()
}
torch.save(best_estimator_data, "best_estimator.pth")
log_message("Saved final estimator (weights + params) to 'best_estimator.pth'.")

# Save the log of steps
with open("log_steps.json", "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log with timestamps to 'log_steps.json'.")
log_message("All done!")
