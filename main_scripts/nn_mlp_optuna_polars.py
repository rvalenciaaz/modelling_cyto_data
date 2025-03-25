import glob
import os
import polars as pl
import numpy as np
import json
import datetime
import joblib  # for saving the scaler & label encoder

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Sklearn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, ClassifierMixin

# For plotting
import matplotlib.pyplot as plt
import optuna

# Make the "outputs" folder if it doesn't exist
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# 1. READING & SUBSAMPLING CSV FILES
# ---------------------------------------------------------
log_message("Reading CSV files and subsampling with Polars...")

csv_files = glob.glob("species*.csv")  # Will get a list of species*.csv filenames
df_list = []

for file_path in csv_files:
    temp_df = pl.read_csv(file_path)
    # Optional: subsample if desired
    # temp_df = temp_df.sample(n=min(temp_df.height, 10_000), seed=42)

    label = file_path.split('.')[0]  # e.g. "species1.csv" -> "species1"
    temp_df = temp_df.with_columns(pl.lit(label).alias("Label"))
    df_list.append(temp_df)

combined_df = pl.concat(df_list)

# Remove the "species" prefix if your filenames contain that
combined_df = combined_df.with_columns(
    pl.col("Label").str.replace("species", "")
)

log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. FILTERING FEATURES BASED ON MAD
# ---------------------------------------------------------
from scipy.stats import median_abs_deviation

log_message("Filtering numeric features based on MAD...")

numeric_dtypes = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
]
numerical_columns = [col for col, dtype in combined_df.schema.items() if dtype in numeric_dtypes]
numerical_data = combined_df.select(numerical_columns)

cv_results = {}
for col in numerical_data.columns:
    values = numerical_data[col].to_numpy()
    mean_val = np.mean(values)
    std_val  = np.std(values)
    cv = (std_val / mean_val)*100 if mean_val != 0 else np.nan
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
# 3. TRAIN/TEST SPLIT
# ---------------------------------------------------------
log_message("Splitting data into training and test sets...")

X = final_df.select(pl.all().exclude("Label")).to_numpy()
y = final_df.select("Label").to_numpy().ravel()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
log_message(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ---------------------------------------------------------
# 4. DEFINE NN CLASSES
# ---------------------------------------------------------
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
        self.train_losses_ = []
        self.val_losses_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.input_dim_ = X.shape[1]
        self.output_dim_ = len(np.unique(y))

        self.model_ = ConfigurableNN(
            input_dim=self.input_dim_,
            hidden_size=self.hidden_size,
            output_dim=self.output_dim_,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        X_torch = torch.from_numpy(X).float().to(self.device)
        y_torch = torch.from_numpy(y).long().to(self.device)
        dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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

            avg_loss = epoch_loss / len(dataloader)
            self.train_losses_.append(avg_loss)

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
                print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

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
# 5. OPTUNA OBJECTIVE FUNCTION
# ---------------------------------------------------------
def objective(trial):
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers  = trial.suggest_int('num_layers', 3, 30)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [64, 128, 256])
    epochs      = 30  # fewer epochs for quick search

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    accuracy_scores = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Scale within this fold
        scaler_fold = StandardScaler()
        X_tr_fold_scaled = scaler_fold.fit_transform(X_tr_fold)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold)

        clf = PyTorchNNClassifierWithVal(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            num_layers=num_layers,
            dropout=dropout,
            verbose=False
        )
        clf.fit(X_tr_fold_scaled, y_tr_fold, X_val_fold_scaled, y_val_fold)

        preds = clf.predict(X_val_fold_scaled)
        fold_acc = accuracy_score(y_val_fold, preds)
        accuracy_scores.append(fold_acc)

    return np.mean(accuracy_scores)

# ---------------------------------------------------------
# 6. RUN OPTUNA STUDY
# ---------------------------------------------------------
log_message("=== Starting Optuna hyperparameter optimization ===")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)  # set to your desired # of trials

best_params = study.best_params
best_score = study.best_value
log_message(f"Optuna best params: {best_params}")
log_message(f"Optuna best CV accuracy: {best_score:.4f}")

# ---------------------------------------------------------
# 7. FINAL REFIT (WITH A SINGLE SCALER FOR THE WHOLE TRAIN SET)
# ---------------------------------------------------------
log_message("Re-fitting with best hyperparameters on entire training set (30 epochs)...")
best_params_for_final = best_params.copy()
best_params_for_final["epochs"] = 30
best_params_for_final["verbose"] = True

# Create & fit the final scaler
pipeline_scaler = StandardScaler()
X_train_scaled = pipeline_scaler.fit_transform(X_train)

# Save the scaler to the outputs folder
scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
joblib.dump(pipeline_scaler, scaler_path)
log_message(f"Saved fitted scaler to '{scaler_path}'.")

# Also save the fitted label encoder (if needed for decoding predictions later)
label_encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
joblib.dump(label_encoder, label_encoder_path)
log_message(f"Saved fitted label encoder to '{label_encoder_path}'.")

# Build & train final model
final_clf = PyTorchNNClassifierWithVal(**best_params_for_final)
final_clf.fit(X_train_scaled, y_train)

# Evaluate on Test
X_test_scaled = pipeline_scaler.transform(X_test)
y_pred = final_clf.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
log_message(f"Final Test Accuracy: {test_acc:.4f}")

# Save confusion matrix & classification report
cm = confusion_matrix(y_test, y_pred)
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.npy")
np.save(cm_path, cm)
log_message(f"Saved confusion matrix to '{cm_path}'.")

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Final Model)")
plt.tight_layout()
cm_fig_path = os.path.join(OUTPUT_DIR, "confusion_matrix_nn.png")
plt.savefig(cm_fig_path, bbox_inches='tight')
plt.close()

class_report = classification_report(y_test, y_pred)
log_message("\nClassification Report:")
log_message(class_report)

# Save metrics + best params
metrics_dict = {
    "test_accuracy": float(test_acc),
    "classification_report": class_report,
    "optuna_best_params": best_params,
    "optuna_best_cv_score": float(best_score)
}
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=2)

log_message(f"Saved metrics to '{metrics_path}'.")

# Save final model weights
model_path = os.path.join(OUTPUT_DIR, "best_model_state.pth")
torch.save(final_clf.model_.state_dict(), model_path)
log_message(f"Saved final model's state_dict to '{model_path}'.")

# Save a log of steps
log_path = os.path.join(OUTPUT_DIR, "log_steps.json")
with open(log_path, "w") as f:
    json.dump(log_steps, f, indent=2)

log_message(f"Saved detailed log with timestamps to '{log_path}'.")
log_message("All done!")
