import glob
import os
import json
import pickle
import datetime

import numpy as np
from scipy.stats import median_abs_deviation

import polars as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

# If you haven't installed Optuna, run: pip install optuna
import optuna

# ---------------------------------------------------------
# 0. Logging utility for timestamps
# ---------------------------------------------------------
log_steps = []
def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[now: {now_str}] {message}")
    print(message)

# Create an output folder to store all results.
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------
# 1. Reading CSV files and subsampling with Polars
# ---------------------------------------------------------
log_message("Reading CSV files and subsampling using Polars...")
csv_files = glob.glob("species*.csv")  # e.g. species1.csv, species2.csv, etc.
df_list = []

for file_path in csv_files:
    # Read with Polars
    temp_df = pl.read_csv(file_path)
    # Derive the label from the filename
    label = os.path.splitext(os.path.basename(file_path))[0]  # e.g. "species1"
    # Add the label column
    temp_df = temp_df.with_columns(pl.lit(label).alias("Label"))
    df_list.append(temp_df)

# Concatenate all data
combined_df = pl.concat(df_list, how="vertical")

# Strip "species" from the Label for clarity
combined_df = combined_df.with_columns(
    pl.col("Label").str.replace("species", "")
)

log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. Filtering numeric features based on MAD
# ---------------------------------------------------------
log_message("Filtering numeric features based on MAD...")

# Select numeric columns
numerical_cols = [col for col in combined_df.columns if pl.datatypes.is_numeric_dtype(combined_df[col].dtype)]
numerical_data = combined_df.select(numerical_cols)

cv_results = {}
for col in numerical_cols:
    # Convert column to NumPy array for calculations
    arr = numerical_data[col].to_numpy()
    mean_val = arr.mean()
    std_val = arr.std()
    cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan
    mad = median_abs_deviation(arr, scale='normal')
    cv_results[col] = [col, cv, mad]

# Convert the dictionary to a Polars DataFrame
cv_pl = pl.DataFrame(
    data=list(cv_results.values()),
    columns=["Feature", "CV", "MAD"]
)

# Apply a MAD threshold
MAD_THRESHOLD = 5
features_to_keep = cv_pl.filter(pl.col("MAD") >= MAD_THRESHOLD)["Feature"].to_list()

# Keep only these features + Label
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df.select(cols_to_keep)

log_message(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. Train/Test split
#    (We're doing unsupervised training, but we keep labels 
#     for the latent space usage later on.)
# ---------------------------------------------------------
log_message("Splitting data into training and test sets...")

# Convert to numpy
X = final_df.drop("Label", axis=1).to_numpy()
y = final_df.select("Label").to_numpy().ravel()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded
)
log_message(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Pre-scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Define the VAE model in PyTorch (with Dropout)
# ---------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim, num_layers, dropout_rate=0.0):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            encoder_layers.append(nn.Linear(prev_dim, hidden_size))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_size
        self.encoder_net = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for _ in range(num_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_size))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_size
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder_net = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder_net(z)
        return x_recon, mu, logvar

def vae_loss_function(x, x_recon, mu, logvar):
    # Mean squared error reconstruction loss
    recon_loss = torch.mean((x - x_recon)**2)
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return recon_loss + kl_loss

# ---------------------------------------------------------
# 5. Define training and evaluation loops
# ---------------------------------------------------------
def train_vae(model, optimizer, dataloader, device):
    model.train()
    train_loss = 0.0
    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss = vae_loss_function(x, x_recon, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    return train_loss / len(dataloader.dataset)

def evaluate_vae(model, dataloader, device):
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            x_recon, mu, logvar = model(x)
            loss = vae_loss_function(x, x_recon, mu, logvar)
            eval_loss += loss.item() * x.size(0)
    return eval_loss / len(dataloader.dataset)

# ---------------------------------------------------------
# 6. Use Optuna for hyperparameter optimization
# ---------------------------------------------------------
log_message("Determining device for training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_message(f"Using device: {device}")

# Create a validation split from X_train_scaled for Optuna
X_train_grid, X_val_grid = train_test_split(X_train_scaled, test_size=0.2, random_state=42)
X_train_grid_tensor = torch.tensor(X_train_grid, dtype=torch.float32)
X_val_grid_tensor = torch.tensor(X_val_grid, dtype=torch.float32)

train_dataset_grid = TensorDataset(X_train_grid_tensor)
val_dataset_grid = TensorDataset(X_val_grid_tensor)

def objective(trial):
    """
    Defines the objective function for Optuna to minimize.
    It samples hyperparameters, constructs the model, trains,
    and returns the validation loss.
    """
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    latent_dim = trial.suggest_categorical("latent_dim", [5, 10, 20, 30, 50])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-3])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    # Fixed epoch count for searching
    epochs = 20

    model = VAE(
        input_dim=X_train_scaled.shape[1],
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)
    optimizer_local = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset_grid, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_grid, batch_size=batch_size, shuffle=False)

    for _ in range(epochs):
        train_vae(model, optimizer_local, train_loader, device)

    val_loss = evaluate_vae(model, val_loader, device)
    return val_loss

study = optuna.create_study(direction="minimize")
n_trials = 40  # Increase if you want a more thorough search
log_message("Starting Optuna hyperparameter optimization...")
study.optimize(objective, n_trials=n_trials)

best_trial = study.best_trial
best_params = best_trial.params
best_val_loss = best_trial.value
log_message(f"Best Val Loss: {best_val_loss}")
log_message(f"Best Hyperparameters: {best_params}")

best_params["epochs"] = 20  # The same as search, or override if desired
with open(os.path.join("output", "best_hyperparameters.json"), "w") as f:
    json.dump(best_params, f, indent=2)
log_message(f"Saved best hyperparameters to 'output/best_hyperparameters.json': {best_params}")

# ---------------------------------------------------------
# 7. 5-Fold Nested Cross-Validation using Best Hyperparameters
#    (We will track train/val loss per epoch for plotting.)
# ---------------------------------------------------------
log_message("Starting 5-fold cross-validation using best hyperparameters...")
n_splits = 5
outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_losses = []

all_folds_train_losses = []
all_folds_val_losses = []

input_dim = X_train_scaled.shape[1]

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_scaled), start=1):
    log_message(f"Starting outer CV fold {fold}/{n_splits}...")
    X_train_fold = torch.tensor(X_train_scaled[train_idx], dtype=torch.float32)
    X_val_fold = torch.tensor(X_train_scaled[val_idx], dtype=torch.float32)

    train_dataset_fold = TensorDataset(X_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold)

    train_loader_fold = DataLoader(train_dataset_fold,
                                   batch_size=best_params["batch_size"],
                                   shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold,
                                 batch_size=best_params["batch_size"],
                                 shuffle=False)

    model_cv = VAE(input_dim=input_dim,
                   hidden_size=best_params["hidden_size"],
                   latent_dim=best_params["latent_dim"],
                   num_layers=best_params["num_layers"],
                   dropout_rate=best_params["dropout_rate"]).to(device)

    optimizer_cv = optim.Adam(model_cv.parameters(), lr=best_params["learning_rate"])

    fold_train_losses = []
    fold_val_losses = []

    for epoch in range(best_params["epochs"]):
        train_loss_epoch = train_vae(model_cv, optimizer_cv, train_loader_fold, device)
        val_loss_epoch = evaluate_vae(model_cv, val_loader_fold, device)
        fold_train_losses.append(train_loss_epoch)
        fold_val_losses.append(val_loss_epoch)

    # Save the per-epoch losses for this fold
    all_folds_train_losses.append(fold_train_losses)
    all_folds_val_losses.append(fold_val_losses)

    # Final validation loss for this fold
    fold_loss = fold_val_losses[-1]  # last epoch's val loss
    cv_losses.append(fold_loss)
    log_message(f"Fold {fold} final validation loss: {fold_loss:.4f}")

mean_cv_loss = np.mean(cv_losses)
std_cv_loss = np.std(cv_losses)
log_message(f"CV complete. Mean validation loss: {mean_cv_loss:.4f}, Std: {std_cv_loss:.4f}")

# Save the raw fold losses and final CV results to disk
cv_info = {
    "fold_train_losses": all_folds_train_losses,
    "fold_val_losses": all_folds_val_losses,
    "cv_losses": cv_losses,
    "mean_cv_loss": mean_cv_loss,
    "std_cv_loss": std_cv_loss
}
with open(os.path.join("output", "fold_losses_data.pkl"), "wb") as f:
    pickle.dump(cv_info, f)

with open(os.path.join("output", "nested_cv_results.json"), "w") as f:
    json.dump({
        "cv_losses": cv_losses,
        "mean_cv_loss": mean_cv_loss,
        "std_cv_loss": std_cv_loss
    }, f, indent=2)

# ---------------------------------------------------------
# 7a. Plot fold losses for the 5-Fold CV (all folds in a single plot)
# ---------------------------------------------------------
epochs_range = np.arange(1, best_params['epochs'] + 1)

plt.figure(figsize=(8, 5))
for fold_idx in range(n_splits):
    # Plot training and validation losses for each fold
    plt.plot(epochs_range, all_folds_train_losses[fold_idx], label=f"Fold {fold_idx+1} Train", linestyle="--")
    plt.plot(epochs_range, all_folds_val_losses[fold_idx], label=f"Fold {fold_idx+1} Val")

plt.title("5-Fold CV - Training & Validation Losses per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join("output", "cv_fold_losses.png"), bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 7b. Aggregated plot (mean ± std across folds)
# ---------------------------------------------------------
mean_train_loss_per_epoch = np.mean(all_folds_train_losses, axis=0)
std_train_loss_per_epoch = np.std(all_folds_train_losses, axis=0)
mean_val_loss_per_epoch = np.mean(all_folds_val_losses, axis=0)
std_val_loss_per_epoch = np.std(all_folds_val_losses, axis=0)

plt.figure(figsize=(8, 5))

# Train
plt.plot(epochs_range, mean_train_loss_per_epoch, label="Mean Train Loss")
plt.fill_between(
    epochs_range,
    mean_train_loss_per_epoch - std_train_loss_per_epoch,
    mean_train_loss_per_epoch + std_train_loss_per_epoch,
    alpha=0.2
)

# Val
plt.plot(epochs_range, mean_val_loss_per_epoch, label="Mean Validation Loss")
plt.fill_between(
    epochs_range,
    mean_val_loss_per_epoch - std_val_loss_per_epoch,
    mean_val_loss_per_epoch + std_val_loss_per_epoch,
    alpha=0.2
)

plt.title("Aggregated Training & Validation Loss (Mean ± Std) - 5-Fold CV")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("output", "cv_aggregated_loss.png"), bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 8. Train final VAE model using best hyperparameters (increase epochs)
# ---------------------------------------------------------
best_params["epochs"] = 100  # Increase epochs for final training
log_message("Training final VAE model with best hyperparameters (final run)...")

X_train_final, X_val_final = train_test_split(X_train_scaled, test_size=0.2, random_state=42)
X_train_final_tensor = torch.tensor(X_train_final, dtype=torch.float32)
X_val_final_tensor = torch.tensor(X_val_final, dtype=torch.float32)

train_dataset_final = TensorDataset(X_train_final_tensor)
val_dataset_final = TensorDataset(X_val_final_tensor)

train_loader_final = DataLoader(train_dataset_final,
                                batch_size=best_params["batch_size"],
                                shuffle=True)
val_loader_final = DataLoader(val_dataset_final,
                              batch_size=best_params["batch_size"],
                              shuffle=False)

model_final = VAE(
    input_dim=input_dim,
    hidden_size=best_params["hidden_size"],
    latent_dim=best_params["latent_dim"],
    num_layers=best_params["num_layers"],
    dropout_rate=best_params["dropout_rate"]
).to(device)

optimizer_final = optim.Adam(model_final.parameters(), lr=best_params["learning_rate"])

train_loss_history = []
val_loss_history = []

for epoch in range(best_params['epochs']):
    train_loss_epoch = train_vae(model_final, optimizer_final, train_loader_final, device)
    val_loss_epoch = evaluate_vae(model_final, val_loader_final, device)
    train_loss_history.append(train_loss_epoch)
    val_loss_history.append(val_loss_epoch)
    log_message(f"Epoch {epoch+1}/{best_params['epochs']} - Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}")

# Save training history for replication
np.savez(os.path.join("output", "vae_training_history.npz"),
         loss=np.array(train_loss_history),
         val_loss=np.array(val_loss_history))

# ---------------------------------------------------------
# 9. Evaluate on the test set and extract latent space encoding
# ---------------------------------------------------------
log_message("Evaluating final VAE model on test set...")
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)
test_loss = evaluate_vae(model_final, test_loader, device)
log_message(f"Test loss: {test_loss:.4f}")

log_message("Extracting latent space encoding for test set...")
model_final.eval()
with torch.no_grad():
    X_test_tensor_gpu = X_test_tensor.to(device)
    _, mu_test, _ = model_final(X_test_tensor_gpu)
    z_test_mean = mu_test.cpu().numpy()

# Also extract latent space for the full training set (useful for XGBoost training)
X_train_tensor_gpu = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    _, mu_train, _ = model_final(X_train_tensor_gpu)
    z_train_mean = mu_train.cpu().numpy()

# Save latent space encodings and labels
np.savez(os.path.join("output", "latent_space_data.npz"),
         latent_train=z_train_mean,
         latent_test=z_test_mean,
         y_train=y_train,
         y_test=y_test)

# ---------------------------------------------------------
# If latent_dim != 2, use PCA for 2D visualization
# ---------------------------------------------------------
if z_test_mean.shape[1] == 2:
    latent_2d = z_test_mean
else:
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(z_test_mean)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    latent_2d[:, 0],
    latent_2d[:, 1],
    c=label_encoder.transform(y_test),
    alpha=0.7
)
plt.title("Latent Space Encoding (Test Set)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

cbar = plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
cbar.ax.set_ylabel("Encoded Label")

plt.tight_layout()
plt.savefig(os.path.join("output", "latent_space_plot.png"), bbox_inches='tight')
plt.close()
log_message("Saved latent space plot to 'output/latent_space_plot.png'.")

# ---------------------------------------------------------
# 10. Plot and save final training history (loss curves)
# ---------------------------------------------------------
epochs_range = np.arange(1, best_params['epochs'] + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_loss_history, label="Train Loss")
plt.plot(epochs_range, val_loss_history, label="Validation Loss")
plt.title("Final VAE Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("output", "vae_training_loss.png"), bbox_inches='tight')
plt.close()
log_message("Saved final VAE training loss plot to 'output/vae_training_loss.png'.")

# ---------------------------------------------------------
# 11. Save final model weights, scaler, label encoder, selected features, logs
# ---------------------------------------------------------
torch.save(model_final.state_dict(), os.path.join("output", "best_vae_weights.pth"))
log_message("Saved final VAE model weights to 'output/best_vae_weights.pth'.")

with open(os.path.join("output", "data_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
log_message("Saved data scaler to 'output/data_scaler.pkl'.")

with open(os.path.join("output", "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)
log_message("Saved label encoder to 'output/label_encoder.pkl'.")

with open(os.path.join("output", "selected_features.json"), "w") as f:
    json.dump(features_to_keep, f, indent=2)
log_message("Saved selected features to 'output/selected_features.json'.")

with open(os.path.join("output", "vae_log_steps.json"), "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log with timestamps to 'output/vae_log_steps.json'.")

# ---------------------------------------------------------
# 12. OPTIONAL: Use latent space (encoder outputs) as features for XGBoost
# ---------------------------------------------------------
log_message("Training an XGBoost classifier on the latent space to assess quality...")

# Train on z_train_mean, predict on z_test_mean
xgb_clf = XGBClassifier(random_state=42)
xgb_clf.fit(z_train_mean, y_train)

y_pred = xgb_clf.predict(z_test_mean)

acc = accuracy_score(y_test, y_pred)
log_message(f"XGBoost classifier accuracy (latent space) on test set: {acc:.4f}")

clf_report = classification_report(y_test, y_pred, digits=4)
log_message("Classification Report:\n" + clf_report)

# Save the classifier, accuracy, and classification report
with open(os.path.join("output", "latent_space_xgb.pkl"), "wb") as f:
    pickle.dump(xgb_clf, f)
with open(os.path.join("output", "latent_space_xgb_results.json"), "w") as f:
    json.dump(
        {
            "accuracy": acc,
            "classification_report": clf_report
        },
        f,
        indent=2
    )

log_message("All done!")
