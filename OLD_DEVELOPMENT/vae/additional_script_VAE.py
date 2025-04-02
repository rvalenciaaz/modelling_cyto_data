#import glob
import os
import json
import pickle
import datetime
import itertools

import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
#import scienceplots
#plt.style.use(['science', 'nature'])

# ---------------------------------------------------------
# 0. Logging utility for timestamps
# ---------------------------------------------------------
log_steps = []
def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# ---------------------------------------------------------
# 1. Reading & subsampling CSV files
# ---------------------------------------------------------
log_message("Reading CSV files and subsampling...")
csv_files = glob.glob("species*.csv")  # e.g. species1.csv, species2.csv, etc.
df_list = []
for file_path in csv_files:
    temp_df = pd.read_csv(file_path)
    # Optionally subsample here if needed
    label = os.path.splitext(os.path.basename(file_path))[0]  # e.g. "species1"
    temp_df['Label'] = label
    df_list.append(temp_df)
combined_df = pd.concat(df_list, ignore_index=True)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)
log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. Filtering features based on MAD
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
features_to_keep = cv_df.loc[cv_df["MAD"] >= MAD_THRESHOLD, "Feature"].tolist()  # filter by MAD
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()
log_message(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. Train/Test split (unsupervised; labels are kept for latent space plotting)
# ---------------------------------------------------------
log_message("Splitting data into training and test sets...")
X = final_df.drop(columns=["Label"]).values
y = final_df["Label"].values  # kept for later visualization
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded
)
log_message(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Pre-scale data (the scaler will be saved for replication if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Define the VAE model in PyTorch
# ---------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim, num_layers):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            encoder_layers.append(nn.Linear(prev_dim, hidden_size))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_size
        self.encoder_net = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for i in range(num_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_size))
            decoder_layers.append(nn.ReLU())
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
# 6. Grid Search for Hyperparameter Optimization
# ---------------------------------------------------------
# Define grid values
grid_hidden_size = [16, 32]
grid_latent_dim = [5, 30]
grid_num_layers = [1, 2]
grid_learning_rate = [1e-4, 1e-3]
grid_batch_size = [16, 32]
grid_epochs = 20  # Fixed epochs for grid search

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_message(f"Using device: {device}")

# Create a validation split from X_train_scaled for grid search
X_train_grid, X_val_grid = train_test_split(X_train_scaled, test_size=0.2, random_state=42)
X_train_grid_tensor = torch.tensor(X_train_grid, dtype=torch.float32)
X_val_grid_tensor = torch.tensor(X_val_grid, dtype=torch.float32)

train_dataset_grid = TensorDataset(X_train_grid_tensor)
val_dataset_grid = TensorDataset(X_val_grid_tensor)

best_val_loss = float('inf')
best_params = None

log_message("Starting grid search for hyperparameter optimization...")

for hidden_size, latent_dim, num_layers, learning_rate, batch_size in itertools.product(
        grid_hidden_size, grid_latent_dim, grid_num_layers, grid_learning_rate, grid_batch_size):

    log_message(f"Testing configuration: hidden_size={hidden_size}, latent_dim={latent_dim}, "
                f"num_layers={num_layers}, learning_rate={learning_rate}, batch_size={batch_size}")

    model = VAE(input_dim=X_train_scaled.shape[1],
                hidden_size=hidden_size,
                latent_dim=latent_dim,
                num_layers=num_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset_grid, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_grid, batch_size=batch_size, shuffle=False)

    # Train for the fixed number of epochs
    for epoch in range(grid_epochs):
        train_vae(model, optimizer, train_loader, device)

    val_loss = evaluate_vae(model, val_loader, device)
    log_message(f"Configuration val_loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {
            "hidden_size": hidden_size,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": grid_epochs
        }
        log_message(f"New best configuration found: {best_params} with val_loss: {best_val_loss:.4f}")

# Save best hyperparameters to file
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=2)
log_message(f"Best hyperparameters: {best_params}")

# ---------------------------------------------------------
# 7. 5-Fold Nested Cross-Validation using Best Hyperparameters
# ---------------------------------------------------------
log_message("Starting 5-fold nested cross-validation using best hyperparameters...")
n_splits = 5
outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_losses = []
input_dim = X_train_scaled.shape[1]

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_scaled), start=1):
    log_message(f"Starting outer CV fold {fold}/{n_splits}...")
    X_train_fold = torch.tensor(X_train_scaled[train_idx], dtype=torch.float32)
    X_val_fold = torch.tensor(X_train_scaled[val_idx], dtype=torch.float32)

    train_dataset_fold = TensorDataset(X_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold)

    train_loader_fold = DataLoader(train_dataset_fold, batch_size=best_params["batch_size"], shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=best_params["batch_size"], shuffle=False)

    model_cv = VAE(input_dim=input_dim,
                   hidden_size=best_params["hidden_size"],
                   latent_dim=best_params["latent_dim"],
                   num_layers=best_params["num_layers"]).to(device)

    optimizer_cv = optim.Adam(model_cv.parameters(), lr=best_params["learning_rate"])

    for epoch in range(best_params["epochs"]):  # using grid search epochs (20 epochs) for CV
        train_vae(model_cv, optimizer_cv, train_loader_fold, device)

    fold_loss = evaluate_vae(model_cv, val_loader_fold, device)
    cv_losses.append(fold_loss)
    log_message(f"Fold {fold} validation loss: {fold_loss:.4f}")

mean_cv_loss = np.mean(cv_losses)
std_cv_loss = np.std(cv_losses)
log_message(f"Nested CV complete. Mean validation loss: {mean_cv_loss:.4f}, Std: {std_cv_loss:.4f}")

# Save nested CV results
with open("nested_cv_results.json", "w") as f:
    json.dump({
        "cv_losses": cv_losses,
        "mean_cv_loss": mean_cv_loss,
        "std_cv_loss": std_cv_loss
    }, f, indent=2)

# ---------------------------------------------------------
# 8. Train final VAE model using best hyperparameters (increase epochs)
# ---------------------------------------------------------
# Increase epochs for the final training run
best_params['epochs'] = 100
log_message("Training final VAE model with best hyperparameters (final run)...")

# Create a training and validation split from the full training set
X_train_final, X_val_final = train_test_split(X_train_scaled, test_size=0.2, random_state=42)
X_train_final_tensor = torch.tensor(X_train_final, dtype=torch.float32)
X_val_final_tensor = torch.tensor(X_val_final, dtype=torch.float32)
train_dataset_final = TensorDataset(X_train_final_tensor)
val_dataset_final = TensorDataset(X_val_final_tensor)
train_loader_final = DataLoader(train_dataset_final, batch_size=best_params["batch_size"], shuffle=True)
val_loader_final = DataLoader(val_dataset_final, batch_size=best_params["batch_size"], shuffle=False)

model_final = VAE(input_dim=input_dim,
                  hidden_size=best_params["hidden_size"],
                  latent_dim=best_params["latent_dim"],
                  num_layers=best_params["num_layers"]).to(device)

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
np.savez("vae_training_history.npz",
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
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    # Forward pass to obtain the latent mean (mu)
    _, mu, _ = model_final(X_test_tensor)
    z_mean = mu.cpu().numpy()

np.savez("latent_space_data.npz", latent_encodings=z_mean, labels=y_test)

# If the latent dimension is not 2, reduce via PCA for plotting
if z_mean.shape[1] == 2:
    latent_2d = z_mean
else:
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(z_mean)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                      c=label_encoder.transform(y_test), cmap='viridis', alpha=0.7)
plt.title("Latent Space Encoding (Test Set)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
cbar = plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
cbar.ax.set_ylabel("Encoded Label")
plt.tight_layout()
plt.savefig("latent_space_plot.png", bbox_inches='tight')
plt.close()
log_message("Saved latent space plot to 'latent_space_plot.png'.")

# ---------------------------------------------------------
# 10. Plot and save training history (loss curves)
# ---------------------------------------------------------
epochs_range = np.arange(1, best_params['epochs'] + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_loss_history, label="Train Loss", color='blue')
plt.plot(epochs_range, val_loss_history, label="Validation Loss", color='orange')
plt.title("Training and Validation Loss (VAE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("vae_training_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved training loss plot to 'vae_training_loss.png'.")

# ---------------------------------------------------------
# 11. Save final model weights, scaler, and logs
# ---------------------------------------------------------
torch.save(model_final.state_dict(), "best_vae_weights.pth")
log_message("Saved final VAE model weights to 'best_vae_weights.pth'.")

with open("data_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
log_message("Saved data scaler to 'data_scaler.pkl'.")

with open("vae_log_steps.json", "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log with timestamps to 'vae_log_steps.json'.")

log_message("All done!")
