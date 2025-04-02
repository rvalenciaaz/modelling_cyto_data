#!/usr/bin/env python3

import glob
import os
import json
import datetime
import argparse

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. LOGGING UTILITY FOR TIMESTAMPS
# -------------------------------------------------------------
log_steps = []
def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# -------------------------------------------------------------
# 1. DEFINE A VAE CLASS (Similar to Script 2)
# -------------------------------------------------------------
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for reconstruction.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 2 * latent_dim)  # outputs [mu, logvar]
        )
        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x, eps=1e-8):
        """
        Returns a torch.distributions.MultivariateNormal for reparameterization.
        """
        x_enc = self.encoder(x)
        mu, logvar = torch.chunk(x_enc, 2, dim=-1)
        scale = self.softplus(logvar) + eps  # ensure positivity
        scale_tril = torch.diag_embed(scale)  # diagonal covariance
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Sample z ~ N(mu, sigma^2) using reparam trick.
        """
        return dist.rsample()

    def decode(self, z):
        """
        Decode latent vector z into reconstruction in input space.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass returning (loss, recon_loss, kl_loss).
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon = self.decode(z)

        # Reconstruction loss (MSE)
        recon_loss = ((x - recon) ** 2).sum(-1).mean()

        # KL divergence with standard normal
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        kl_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    def reconstruct(self, x):
        """
        Given an input x, return the reconstructed output without computing the loss.
        """
        dist = self.encode(x)
        z = dist.mean  # or sample from dist
        return self.decode(z)

# -------------------------------------------------------------
# 2. SKLEARN-COMPATIBLE WRAPPER FOR THE VAE
# -------------------------------------------------------------
class SklearnVAE(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for performing a reconstruction-based
    VAE training, so that we can do GridSearchCV.

    Because it's unsupervised, we:
      - treat y=None (ignore it)
      - 'score' is negative reconstruction MSE on the provided data
    """
    def __init__(self, latent_dim=2, hidden_dim=128, epochs=20, batch_size=64,
                 learning_rate=1e-3, weight_decay=1e-2, device='auto', verbose=False):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device == 'auto' else torch.device(device)
        self.verbose = verbose

        self.input_dim_ = None
        self.model_ = None

    def fit(self, X, y=None):
        """
        Train the VAE to reconstruct X.
        """
        # X is shape (n_samples, n_features)
        X = np.array(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]
        dataset = TensorDataset(torch.from_numpy(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build a new VAE model
        self.model_ = VAE(self.input_dim_, self.hidden_dim, self.latent_dim).to(self.device)

        optimizer = optim.AdamW(self.model_.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

        # Train loop
        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                loss, recon_loss, kl_loss = self.model_(batch_x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if self.verbose:
                avg_loss = total_loss / len(dataloader)
                print(f"[Epoch {epoch+1}/{self.epochs}] Loss={avg_loss:.4f}")

        return self

    def score(self, X, y=None):
        """
        Return a *higher-is-better* score for GridSearchCV:
        We compute the negative MSE reconstruction error on X.

        If you prefer a different measure, you can adapt accordingly.
        """
        X = np.array(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        mse_losses = []
        with torch.no_grad():
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                recon = self.model_.reconstruct(batch_x)
                # MSE across features
                mse = ((batch_x - recon) ** 2).sum(-1).mean().item()
                mse_losses.append(mse)
        # Mean of MSE across all batches
        final_mse = float(np.mean(mse_losses))
        # Return the negative (so that *maximizing* this score = minimizing MSE)
        return -final_mse

    def transform(self, X):
        """
        Optional: Return the latent representation. Could be used in a pipeline.
        """
        X = np.array(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        zs = []
        with torch.no_grad():
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                dist = self.model_.encode(batch_x)
                zs.append(dist.mean.cpu().numpy())
        return np.concatenate(zs, axis=0)

# -------------------------------------------------------------
# 3. MAIN SCRIPT: DATA LOADING, PREPROCESSING, GRID SEARCH CV
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VAE + GridSearchCV (MAD filtering, multi-CSV).")
    parser.add_argument("--csv_pattern", type=str, default="species*.csv",
                        help="Glob pattern for CSV files to load (e.g. 'species*.csv').")
    parser.add_argument("--mad_threshold", type=float, default=5.0,
                        help="MAD threshold for feature filtering.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split size if you want a final hold-out. 0.2 = 20%.")
    parser.add_argument("--cv_splits", type=int, default=3,
                        help="Number of folds for cross-validation.")
    parser.add_argument("--output_dir", type=str, default="vae_gridsearch_results",
                        help="Output directory to store logs and results.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------
    # 3A. READ CSV FILES
    # -----------------------------
    log_message(f"Reading CSV files with pattern '{args.csv_pattern}'...")
    csv_files = glob.glob(args.csv_pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern '{args.csv_pattern}'")

    df_list = []
    for fp in csv_files:
        tmp = pd.read_csv(fp)
        label = os.path.splitext(os.path.basename(fp))[0]  # e.g. 'species1' from 'species1.csv'
        tmp["Label"] = label
        df_list.append(tmp)
    combined_df = pd.concat(df_list, ignore_index=True)
    log_message(f"Combined data shape: {combined_df.shape}")

    # Remove 'species' prefix if desired
    combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

    # -----------------------------
    # 3B. FILTER FEATURES BY MAD
    # -----------------------------
    log_message("Filtering numeric features by MAD threshold...")
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    mad_values = {}
    for col in numeric_cols:
        mad_val = median_abs_deviation(combined_df[col].dropna().values, scale='normal')
        mad_values[col] = mad_val

    mad_series = pd.Series(mad_values)
    features_to_keep = mad_series[mad_series >= args.mad_threshold].index.tolist()

    # We keep these features + optional 'Label' for reference
    final_df = combined_df[features_to_keep + ["Label"]].copy()
    log_message(f"Features retained after MAD filtering: {len(features_to_keep)}")

    # -----------------------------
    # 3C. SPLIT INTO TRAIN/TEST (OPTIONAL)
    # -----------------------------
    # Because it's unsupervised, the "Label" is not used to create y,
    # but we might do a final hold-out to evaluate reconstruction error out of sample
    X_all = final_df[features_to_keep].values.astype(np.float32)
    # Optional log-transform if your data demands it, e.g.:
    # X_all = np.log10(X_all + 1e-10).astype(np.float32)

    # If you want to do a final hold-out:
    X_train, X_test = train_test_split(X_all, test_size=args.test_size, random_state=42)

    # -----------------------------
    # 4. BUILD A PIPELINE & GRIDSEARCH
    #    - We'll scale data, then pass to SklearnVAE
    # -----------------------------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("vae", SklearnVAE(verbose=False))  # pass any default you want
    ])

    param_grid = {
        "vae__latent_dim":  [2, 5, 10],       # optimize latent dimension
        "vae__hidden_dim":  [128, 256],       # and hidden dim
        "vae__epochs":      [20],             # can fix or add multiple if you want
        "vae__batch_size":  [32, 64],
        "vae__learning_rate": [1e-3, 1e-4],
        "vae__weight_decay": [1e-2, 1e-3]
    }

    # We use a negative MSE-based scoring (the 'score' method returns -MSE)
    # so let's use "maximize" = scoring='neg_mean_squared_error'
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # consistent with our 'score' returning -MSE
        cv=args.cv_splits,
        verbose=2,
        n_jobs=1,     # set to -1 if you want parallel
        return_train_score=True
    )

    log_message("Starting GridSearchCV for VAE (unsupervised, neg MSE as scoring).")
    grid_search.fit(X_train)  # X_train only (no y)
    log_message("GridSearchCV complete.")

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    log_message(f"Best Params: {best_params}")
    log_message(f"Best CV Score (neg MSE): {best_score:.4f}")

    # -----------------------------
    # 5. REFIT ON THE FULL TRAIN SET & EVALUATE ON TEST
    # -----------------------------
    best_model = grid_search.best_estimator_  # pipeline with the best found config
    # Already fitted by GridSearchCV with .fit(X_train)

    # Evaluate on test set
    test_score = best_model.score(X_test)  # returns negative MSE
    # "test_score" is negative MSE, so MSE = -test_score
    test_mse = -test_score
    log_message(f"Final test MSE = {test_mse:.4f}")

    # -----------------------------
    # 6. SAVE RESULTS & LOGS
    # -----------------------------
    result_dict = {
        "best_params": best_params,
        "best_cv_neg_mse": float(best_score),
        "test_mse": float(test_mse),
        "cv_results": grid_search.cv_results_
    }
    out_json = os.path.join(args.output_dir, "grid_search_results.json")
    with open(out_json, "w") as f:
        json.dump(result_dict, f, indent=2)
    log_message(f"Saved grid search results to '{out_json}'.")

    # Optionally, you can also save the final VAE state_dict
    vae_estimator = best_model.named_steps["vae"]
    if vae_estimator.model_ is not None:
        torch.save(vae_estimator.model_.state_dict(), os.path.join(args.output_dir, "best_vae_state.pth"))
        log_message("Saved best VAE state_dict.")

    # Save a log file with steps
    log_json = os.path.join(args.output_dir, "log_steps.json")
    with open(log_json, "w") as f:
        json.dump(log_steps, f, indent=2)
    log_message(f"Saved log steps to '{log_json}'.")
    log_message("All done!")

if __name__ == "__main__":
    main()
