# models_vae.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from typing import Optional, Tuple

# ------------------------------------------------------------------
# 1. UNSUPERVISED VAE
# ------------------------------------------------------------------
class VAEBase(nn.Module):
    """
    A simple Variational Autoencoder (VAE):
      - Encoder: maps input -> latent distribution (mean, log_var).
      - Decoder: reconstructs input from latent vector z.
    """
    def __init__(self, input_dim, latent_dim, hidden_size=128):
        super(VAEBase, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_size, latent_dim)
        self.logvar_layer = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),  
            # For real-valued reconstruction, typically no final activation 
            # (assumes MSE or similar). For image data, you might use a sigmoid.
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def vae_loss_function(recon_x, x, mu, logvar):
        """
        Typical VAE loss = reconstruction loss + KL divergence.
        Here we use MSE for reconstruction, but you could use BCE, etc.
        """
        mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
        # KL divergence: D_KL( q(z|x) || p(z) )
        # = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
        kld = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
        return (mse + kld) / x.size(0)  # average per batch

class PyTorchVAEWrapper(BaseEstimator, TransformerMixin):
    """
    Scikit-learn wrapper for an unsupervised VAE. It:
      - Trains the VAE to reconstruct X.
      - 'score' returns a negative reconstruction error on the validation/test set
        (since scikit-learn wants higher=better).
      - 'transform' can return the latent representation for further usage.
    """
    def __init__(
        self,
        latent_dim=8,
        hidden_size=128,
        learning_rate=1e-3,
        batch_size=64,
        epochs=10,
        verbose=True
    ):
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim_ = None
        self.model_ = None

    def fit(self, X, y=None):
        """
        Trains the VAE on X (ignoring y, because it's unsupervised).
        """
        X = np.array(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]

        self.model_ = VAEBase(
            input_dim=self.input_dim_,
            latent_dim=self.latent_dim,
            hidden_size=self.hidden_size
        ).to(self.device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0.0
            for (x_batch,) in dataloader:
                x_batch = x_batch.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model_(x_batch)
                loss = VAEBase.vae_loss_function(recon, x_batch, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x_batch)

            avg_loss = total_loss / len(dataloader.dataset)
            if self.verbose:
                print(f"[Epoch {epoch+1}/{self.epochs}] Avg VAE Loss: {avg_loss:.4f}")

        return self

    def transform(self, X):
        """
        Returns the latent representation (mu) of X.
        """
        self.model_.eval()
        X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            mu, logvar = self.model_.encode(X)
        return mu.cpu().numpy()

    def score(self, X, y=None):
        """
        scikit-learn wants higher=better, so we return -reconstruction_error
        as the 'score'.
        """
        X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            recon, mu, logvar = self.model_(X)
            loss = VAEBase.vae_loss_function(recon, X, mu, logvar)
        return -loss.item()  # negative is better

# ------------------------------------------------------------------
# 2. SUPERVISED VAE
# ------------------------------------------------------------------
class SupervisedVAEBase(nn.Module):
    """
    A VAE plus a classification head:
      - We still do encoder->latent->decoder for reconstruction.
      - We add a classifier layer from latent -> class logits.
    """
    def __init__(self, input_dim, latent_dim, hidden_size=128, num_classes=2):
        super(SupervisedVAEBase, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # VAE encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_size, latent_dim)
        self.logvar_layer = nn.Linear(hidden_size, latent_dim)

        # VAE decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, mu, logvar, logits

    @staticmethod
    def vae_classification_loss(recon_x, x, mu, logvar, logits, y):
        """
        Combined VAE loss + classification (cross-entropy) loss.
        """
        # Reconstruction (MSE) + KL for VAE
        mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)

        # Classification
        ce = nn.CrossEntropyLoss()(logits, y)

        # Weighted sum, or simply add them. 
        # You may introduce a hyperparam (alpha) to scale classification vs. reconstruction.
        total = (mse + kld) / x.size(0) + ce
        return total

class PyTorchSupervisedVAEWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn wrapper for a supervised VAE:
      - Combines reconstruction + classification objectives.
      - 'predict_proba' is from the classification head.
    """
    def __init__(
        self,
        latent_dim=8,
        hidden_size=128,
        learning_rate=1e-3,
        batch_size=64,
        epochs=10,
        verbose=True
    ):
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim_ = None
        self.num_classes_ = None
        self.model_ = None

    def fit(self, X, y):
        """
        Train the supervised VAE to reconstruct X and classify y.
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        self.input_dim_ = X.shape[1]
        self.num_classes_ = len(np.unique(y))

        self.model_ = SupervisedVAEBase(
            input_dim=self.input_dim_,
            latent_dim=self.latent_dim,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes_
        ).to(self.device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0.0

            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                recon, mu, logvar, logits = self.model_(x_batch)
                loss = SupervisedVAEBase.vae_classification_loss(
                    recon, x_batch, mu, logvar, logits, y_batch
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x_batch)

            avg_loss = total_loss / len(dataloader.dataset)
            if self.verbose:
                print(f"[Epoch {epoch+1}/{self.epochs}] Supervised VAE Loss: {avg_loss:.4f}")

        return self

    def predict(self, X):
        self.model_.eval()
        X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            _, _, _, logits = self.model_(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        self.model_.eval()
        X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            _, _, _, logits = self.model_(X)
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

    def score(self, X, y):
        """
        Classification accuracy on (X, y).
        """
        from sklearn.metrics import accuracy_score
        preds = self.predict(X)
        return accuracy_score(y, preds)
