import os
import glob
import json
import datetime
import pickle
import numpy as np
import polars as pl
from scipy.stats import median_abs_deviation

# PyTorch & Pyro
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
import pyro.infer.autoguide as autoguide
import pyro.optim as pyro_optim

# scikit-learn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Hyperparameter Tuning
import optuna

# Plotting
import matplotlib.pyplot as plt

# Fix random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
pyro.set_rng_seed(42)

# ---------------------------------------------------------
# 0. LOGGING UTILITY & OUTPUT FOLDER
# ---------------------------------------------------------
log_steps = []
def log_message(message: str):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# Create folder for saving replication files
OUTPUT_FOLDER = "replication_files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# 1. READ CSV FILES (POLARS) & COMBINE
# ---------------------------------------------------------
log_message("Reading CSV files (species*.csv) with Polars...")
csv_files = glob.glob("species*.csv")  # e.g., species1.csv, species2.csv, etc.

df_list = []
for file_path in csv_files:
    temp_df = pl.read_csv(file_path)
    label_str = file_path.split('.')[0]  # e.g., "species1"
    # Add label column
    temp_df = temp_df.with_column(pl.lit(label_str).alias("Label"))
    df_list.append(temp_df)

combined_df = pl.concat(df_list, how="vertical")
# Strip "species" from Label: "species1" => "1"
combined_df = combined_df.with_column(
    pl.col("Label").str.replace("species", "")
)
log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. MAD-BASED FEATURE FILTER
# ---------------------------------------------------------
log_message("Computing MAD for numeric columns & filtering...")

numeric_cols = [
    c for c in combined_df.columns
    if (combined_df.schema[c] in [pl.Float64, pl.Int64])
]

cv_results = {}
for col in numeric_cols:
    col_data = combined_df[col].drop_nulls().to_numpy()
    if len(col_data) == 0:
        continue
    mean_val = np.mean(col_data)
    std_val  = np.std(col_data, ddof=1)
    mad_val  = median_abs_deviation(col_data, scale='normal')
    cv       = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    cv_results[col] = {
        "Feature": col,
        "CV": cv,
        "MAD": mad_val
    }

cv_df = pl.DataFrame(list(cv_results.values()))
MAD_THRESHOLD = 5
features_to_keep = cv_df.filter(pl.col("MAD") >= MAD_THRESHOLD)["Feature"].to_list()

cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df.select(cols_to_keep)
log_message(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT & SCALING
# ---------------------------------------------------------
log_message("Performing train/test split & scaling...")

X = final_df.drop("Label", axis=1).to_numpy()
y = final_df["Label"].to_numpy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

log_message(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# ---------------------------------------------------------
# 4. BAYESIAN NEURAL NETWORK MODEL
# ---------------------------------------------------------
def bayesian_nn_model(x, y=None, hidden_size=32, num_layers=2, output_dim=2):
    """
    A Bayesian neural network with standard Normal priors for each weight/bias.
    """
    input_dim = x.shape[1]
    hidden = x
    current_dim = input_dim

    for i in range(num_layers):
        w = pyro.sample(
            f"w{i+1}",
            dist.Normal(
                torch.zeros(current_dim, hidden_size),
                torch.ones(current_dim, hidden_size)
            ).to_event(2)
        )
        b = pyro.sample(
            f"b{i+1}",
            dist.Normal(
                torch.zeros(hidden_size),
                torch.ones(hidden_size)
            ).to_event(1)
        )
        hidden = torch.tanh(torch.matmul(hidden, w) + b)
        current_dim = hidden_size

    w_out = pyro.sample(
        "w_out",
        dist.Normal(
            torch.zeros(hidden_size, output_dim),
            torch.ones(hidden_size, output_dim)
        ).to_event(2)
    )
    b_out = pyro.sample(
        "b_out",
        dist.Normal(
            torch.zeros(output_dim),
            torch.ones(output_dim)
        ).to_event(1)
    )

    logits = torch.matmul(hidden, w_out) + b_out

    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

    return logits

def train_pyro_model(
    X_train_tensor,
    y_train_tensor,
    hidden_size=32,
    num_layers=2,
    output_dim=2,
    learning_rate=1e-3,
    num_epochs=1000,
    verbose=False
):
    """
    Trains a Bayesian NN with Pyro (no val-loss tracking).
    Returns (guide, list_of_train_losses).
    """
    pyro.clear_param_store()
    guide = autoguide.AutoNormal(bayesian_nn_model)
    optimizer = pyro_optim.Adam({"lr": learning_rate})
    svi = SVI(bayesian_nn_model, guide, optimizer, loss=Trace_ELBO())

    train_losses = []
    for epoch in range(num_epochs):
        loss = svi.step(
            X_train_tensor,
            y_train_tensor,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=output_dim
        )
        train_losses.append(loss)
        if verbose and epoch % 100 == 0:
            log_message(f"[Epoch {epoch}] Train Loss: {loss:.4f}")

    return guide, train_losses

def train_pyro_model_with_val(
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    hidden_size=32,
    num_layers=2,
    output_dim=2,
    learning_rate=1e-3,
    num_epochs=1000,
    verbose=False
):
    """
    Trains with SVI, returning (guide, train_losses, val_losses) for each epoch.
    """
    pyro.clear_param_store()
    guide = autoguide.AutoNormal(bayesian_nn_model)
    optimizer = pyro_optim.Adam({"lr": learning_rate})
    svi = SVI(bayesian_nn_model, guide, optimizer, loss=Trace_ELBO())

    train_losses = []
    val_losses   = []
    for epoch in range(num_epochs):
        # Single training step
        train_loss = svi.step(
            X_train_tensor,
            y_train_tensor,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=output_dim
        )
        train_losses.append(train_loss)

        # Evaluate on validation set
        val_loss = svi.evaluate_loss(
            X_val_tensor,
            y_val_tensor,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=output_dim
        )
        val_losses.append(val_loss)

        if verbose and epoch % 100 == 0:
            log_message(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return guide, train_losses, val_losses

def predict_pyro_model(
    X_tensor,
    guide,
    hidden_size=32,
    num_layers=2,
    output_dim=2,
    num_samples=500
):
    """
    Draw posterior samples & majority vote for class predictions.
    """
    predictive = Predictive(bayesian_nn_model, guide=guide, num_samples=num_samples)
    samples = predictive(X_tensor, None, hidden_size=hidden_size, num_layers=num_layers, output_dim=output_dim)
    obs_samples = samples["obs"].cpu().numpy()  # shape: (num_samples, n_data)

    def majority_vote(row_samples):
        return np.bincount(row_samples, minlength=output_dim).argmax()

    final_preds = np.apply_along_axis(majority_vote, 0, obs_samples)
    return final_preds

# ---------------------------------------------------------
# 5. OPTUNA HYPERPARAMETER OPTIMIZATION
# ---------------------------------------------------------
def objective(trial):
    hidden_size   = trial.suggest_int("hidden_size", 16, 128, step=16)
    num_layers    = trial.suggest_int("num_layers", 3, 50)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs_tune = 5000  # fewer epochs for faster tuning

    X_tune_train, X_val, y_tune_train, y_val = train_test_split(
        X_train_t, y_train_t, test_size=0.2, random_state=42, stratify=y_train_t
    )

    guide, _ = train_pyro_model(
        X_tune_train, y_tune_train,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=len(np.unique(y_train_t)),
        learning_rate=learning_rate,
        num_epochs=num_epochs_tune,
        verbose=False
    )

    val_preds = predict_pyro_model(
        X_val, guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=len(np.unique(y_train_t)),
        num_samples=300
    )
    return accuracy_score(y_val, val_preds)

study = optuna.create_study(direction="maximize")
log_message("Starting Optuna hyperparameter optimization...")
study.optimize(objective, n_trials=10, timeout=None)  # Adjust n_trials as needed
log_message("Optuna hyperparameter optimization done.")

best_params = study.best_params
best_hidden_size   = best_params["hidden_size"]
best_num_layers    = best_params["num_layers"]
best_learning_rate = best_params["learning_rate"]
log_message(f"Best hyperparameters: {best_params}")

# ---------------------------------------------------------
# 6. 5-FOLD CROSS VALIDATION (Track Train & Val Loss)
# ---------------------------------------------------------
log_message("Starting 5-fold CV with best hyperparams, tracking losses...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_epochs_cv = 20000

fold_train_losses = []
fold_val_losses   = []
fold_accuracies  = []

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train[tr_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[tr_idx], y_train[val_idx]

    X_fold_train_t = torch.tensor(X_fold_train, dtype=torch.float32)
    y_fold_train_t = torch.tensor(y_fold_train, dtype=torch.long)
    X_fold_val_t   = torch.tensor(X_fold_val,   dtype=torch.float32)
    y_fold_val_t   = torch.tensor(y_fold_val,   dtype=torch.long)

    guide_cv, train_losses_cv, val_losses_cv = train_pyro_model_with_val(
        X_fold_train_t, y_fold_train_t,
        X_fold_val_t,   y_fold_val_t,
        hidden_size=best_hidden_size,
        num_layers=best_num_layers,
        output_dim=len(np.unique(y_train)),
        learning_rate=best_learning_rate,
        num_epochs=num_epochs_cv,
        verbose=False
    )

    fold_train_losses.append(train_losses_cv)
    fold_val_losses.append(val_losses_cv)

    # Predict on this fold's val set
    val_preds_fold = predict_pyro_model(
        X_fold_val_t, guide_cv,
        hidden_size=best_hidden_size,
        num_layers=best_num_layers,
        output_dim=len(np.unique(y_train)),
        num_samples=300
    )
    fold_acc = accuracy_score(y_fold_val_t, val_preds_fold)
    fold_accuracies.append(fold_acc)
    log_message(f"[Fold {fold_idx+1}] Accuracy = {fold_acc:.4f}")

cv_mean_accuracy = np.mean(fold_accuracies)
cv_std_accuracy  = np.std(fold_accuracies)
log_message(f"CV Accuracy: {cv_mean_accuracy:.4f} ± {cv_std_accuracy:.4f}")

# Save fold losses for replication
cv_fold_losses_path = os.path.join(OUTPUT_FOLDER, "cv_fold_losses.pkl")
with open(cv_fold_losses_path, "wb") as f:
    pickle.dump({"train_losses": fold_train_losses, "val_losses": fold_val_losses}, f)
log_message(f"Saved fold train/val losses => {cv_fold_losses_path}")

# ---------------------------------------------------------
# 6a. PLOT ALL FOLD LOSSES IN ONE FIGURE
# ---------------------------------------------------------
# For each fold, we overlay train and val on the same plot (10 lines total for 5 folds).
plt.figure(figsize=(10, 6))
for i in range(len(fold_train_losses)):
    epochs = range(1, len(fold_train_losses[i]) + 1)
    plt.plot(epochs, fold_train_losses[i], label=f"Train Fold {i+1}")
    plt.plot(epochs, fold_val_losses[i],   label=f"Val Fold {i+1}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("All Folds Train & Val Loss Curves")
plt.legend(loc="upper right", ncol=2, fontsize="small")
plt.grid(True)
plt.tight_layout()
all_folds_plot = os.path.join(OUTPUT_FOLDER, "cv_folds_loss_trends.png")
plt.savefig(all_folds_plot)
plt.close()
log_message(f"Saved all-fold train/val loss plot => {all_folds_plot}")

# ---------------------------------------------------------
# 6b. PLOT AGGREGATED (MEAN ± STD) ACROSS FOLDS
# ---------------------------------------------------------
all_train_arr = np.array(fold_train_losses)  # shape (n_folds, n_epochs)
all_val_arr   = np.array(fold_val_losses)    # shape (n_folds, n_epochs)

mean_train = all_train_arr.mean(axis=0)
std_train  = all_train_arr.std(axis=0)
mean_val   = all_val_arr.mean(axis=0)
std_val    = all_val_arr.std(axis=0)

epochs = range(1, all_train_arr.shape[1] + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_train, label="Train Loss (mean)")
plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.3)
plt.plot(epochs, mean_val, label="Val Loss (mean)")
plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.3)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Aggregated CV Loss (Mean ± Std)")
plt.legend()
plt.grid(True)
plt.tight_layout()
agg_folds_plot = os.path.join(OUTPUT_FOLDER, "cv_aggregated_loss.png")
plt.savefig(agg_folds_plot)
plt.close()
log_message(f"Saved aggregated CV loss plot => {agg_folds_plot}")

# ---------------------------------------------------------
# 7. FINAL TRAINING ON FULL TRAINING SET + TEST EVAL
# ---------------------------------------------------------
log_message("Final training on all training data...")

num_epochs_final = 20000
final_guide, final_train_losses = train_pyro_model(
    X_train_t, y_train_t,
    hidden_size=best_hidden_size,
    num_layers=best_num_layers,
    output_dim=len(np.unique(y_train)),
    learning_rate=best_learning_rate,
    num_epochs=num_epochs_final,
    verbose=True
)

# Save final training loss array
final_losses_file = os.path.join(OUTPUT_FOLDER, "final_losses.pkl")
with open(final_losses_file, "wb") as f:
    pickle.dump(final_train_losses, f)
log_message(f"Saved final training losses => {final_losses_file}")

# Plot final training loss
epochs_fin = range(1, len(final_train_losses)+1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_fin, final_train_losses, label='Final Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Final Model Training Loss vs. Epoch")
plt.legend()
plt.grid(True)
final_plot_path = os.path.join(OUTPUT_FOLDER, "final_training_loss.png")
plt.savefig(final_plot_path)
plt.close()
log_message(f"Saved final training loss plot => {final_plot_path}")

# Predict on test set
log_message("Predicting on test set with final model...")
test_preds = predict_pyro_model(
    X_test_t, final_guide,
    hidden_size=best_hidden_size,
    num_layers=best_num_layers,
    output_dim=len(np.unique(y_train)),
    num_samples=1000
)
test_acc = accuracy_score(y_test_t, test_preds)
class_rep = classification_report(y_test_t, test_preds, target_names=label_encoder.classes_)

log_message(f"Test Accuracy = {test_acc:.4f}")
log_message("\nClassification Report:\n" + class_rep)

# ---------------------------------------------------------
# 8. SAVE METRICS, MODEL, SCALER, ENCODER, FEATURES, LOGS
# ---------------------------------------------------------
metrics_dict = {
    "best_hyperparams": best_params,
    "cv_mean_accuracy": float(cv_mean_accuracy),
    "cv_std_accuracy": float(cv_std_accuracy),
    "test_accuracy": float(test_acc),
    "classification_report": class_rep
}
metrics_path = os.path.join(OUTPUT_FOLDER, "metrics_pyro.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=2)
log_message(f"Saved metrics => {metrics_path}")

# Save final Pyro guide parameters
guide_state = final_guide.state_dict()
guide_params_path = os.path.join(OUTPUT_FOLDER, "bayesian_nn_pyro_params.pkl")
with open(guide_params_path, "wb") as f:
    pickle.dump(guide_state, f)
log_message(f"Saved final guide params => {guide_params_path}")

# Save scaler, label encoder
scaler_path = os.path.join(OUTPUT_FOLDER, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
log_message(f"Saved scaler => {scaler_path}")

encoder_path = os.path.join(OUTPUT_FOLDER, "label_encoder.pkl")
with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)
log_message(f"Saved label encoder => {encoder_path}")

# Save MAD-filtered features
features_path = os.path.join(OUTPUT_FOLDER, "features_to_keep.json")
with open(features_path, "w") as f:
    json.dump(features_to_keep, f, indent=2)
log_message(f"Saved features to keep => {features_path}")

# Save timestamp log
log_path = os.path.join(OUTPUT_FOLDER, "log_steps_pyro.json")
with open(log_path, "w") as f:
    json.dump(log_steps, f, indent=2)
log_message(f"Saved log => {log_path}")

log_message("All done! Training, CV, final model, and artifact saving complete.")
