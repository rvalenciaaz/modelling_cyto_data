import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.infer.autoguide as autoguide

import torch.nn as nn
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


def bayesian_nn_model(x, y=None, hidden_size=32, num_layers=2, output_dim=2):
    """
    A Bayesian neural network with standard Normal priors for each weight/bias.
    Also returns a deterministic node 'logits' so we can extract its samples.
    """
    input_dim = x.shape[1]
    hidden = x
    current_dim = input_dim

    # Build hidden layers
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
        hidden = torch.tanh(hidden @ w + b)
        current_dim = hidden_size

    # Output layer
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

    logits = hidden @ w_out + b_out

    # Create a deterministic node for logits so we can retrieve them in predictions
    pyro.deterministic("logits", logits)

    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

    return logits

def create_guide(model):
    """
    Creates an AutoNormal guide for the given model.
    """
    return autoguide.AutoNormal(model)

def create_svi(model, guide, learning_rate=1e-3):
    """
    Creates an SVI object with Adam optimizer and Trace_ELBO loss.
    """
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    return svi
    
class ConfigurableNN(nn.Module):
    """
    A feedforward network with variable # of layers, hidden size, and optional dropout.
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

class PyTorchNNClassifierWithVal(BaseEstimator, ClassifierMixin):
    """
    Sklearn-style classifier for a feedforward PyTorch NN, with optional val-loss tracking.
    """
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

        # Create DataLoader
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
                print(f"Epoch [{epoch+1}/{self.epochs}] - "
                      f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

class TabTransformer(nn.Module):
    def __init__(
        self,
        categories: list,        # list of cardinalities for each categorical column
        num_continuous: int,
        transformer_dim: int = 32,
        depth: int = 2,
        heads: int = 2,
        dim_forward: int = 64,
        dropout: float = 0.1,
        mlp_hidden_dims: list = [64, 32],
        num_classes: int = 2
    ):
        super().__init__()
        self.num_categs = len(categories)
        self.num_continuous = num_continuous
        self.transformer_dim = transformer_dim

        self.category_embeds = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=transformer_dim)
            for cardinality in categories
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(dim=transformer_dim,
                                    num_heads=heads,
                                    mlp_hidden_dim=dim_forward,
                                    dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(transformer_dim)

        mlp_input_dim = transformer_dim + num_continuous
        mlp_layers = []
        in_dim = mlp_input_dim
        for hdim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, hdim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = hdim
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x_categ, x_cont):
        batch_size = x_categ.shape[0]

        if self.num_categs > 0:
            cat_embeds = []
            for i, embed in enumerate(self.category_embeds):
                cat_embeds.append(embed(x_categ[:, i]))
            x_cat = torch.stack(cat_embeds, dim=1)
        else:
            x_cat = torch.zeros(batch_size, 1, self.transformer_dim, device=x_categ.device)

        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        if self.num_categs > 0:
            x_cat = torch.cat([cls_token_expanded, x_cat], dim=1)
        else:
            x_cat = cls_token_expanded

        for encoder in self.transformer_encoders:
            x_cat = encoder(x_cat)
        x_cat = self.norm(x_cat)

        x_cls = x_cat[:, 0, :]
        x_full = torch.cat([x_cls, x_cont], dim=-1)
        out = self.mlp_head(x_full)
        return out

class TabTransformerClassifierWithVal:
    def __init__(self,
                 transformer_dim=32,
                 depth=2,
                 heads=2,
                 dim_forward=64,
                 dropout=0.1,
                 mlp_hidden_dims=[64, 32],
                 learning_rate=1e-3,
                 batch_size=32,
                 epochs=10,
                 verbose=True):
        self.transformer_dim = transformer_dim
        self.depth = depth
        self.heads = heads
        self.dim_forward = dim_forward
        self.dropout = dropout
        self.mlp_hidden_dims = mlp_hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.classes_ = None
        self.train_losses_ = []
        self.val_losses_ = []
        self.num_continuous_ = 0
        self.categorical_cardinalities_ = []

    def fit(self, X_categ, X_cont, y, X_categ_val=None, X_cont_val=None, y_val=None):
        X_categ_t = torch.from_numpy(X_categ).long() if X_categ.size > 0 else torch.empty((len(y), 0), dtype=torch.long)
        X_cont_t  = torch.from_numpy(X_cont).float() if X_cont.size > 0 else torch.empty((len(y), 0), dtype=torch.float)
        y_t       = torch.from_numpy(y).long()

        self.num_continuous_ = X_cont_t.shape[1]
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        has_val = (X_categ_val is not None) and (X_cont_val is not None) and (y_val is not None)

        X_categ_t = X_categ_t.to(self.device_)
        X_cont_t  = X_cont_t.to(self.device_)
        y_t       = y_t.to(self.device_)

        if has_val:
            X_categ_val_t = torch.from_numpy(X_categ_val).long() if X_categ_val.size > 0 else torch.empty((len(y_val), 0), dtype=torch.long)
            X_cont_val_t  = torch.from_numpy(X_cont_val).float() if X_cont_val.size > 0 else torch.empty((len(y_val), 0), dtype=torch.float)
            y_val_t       = torch.from_numpy(y_val).long()

            X_categ_val_t = X_categ_val_t.to(self.device_)
            X_cont_val_t  = X_cont_val_t.to(self.device_)
            y_val_t       = y_val_t.to(self.device_)

        if X_categ.shape[1] > 0:
            max_per_column = (X_categ.max(axis=0) + 1).tolist()
            self.categorical_cardinalities_ = [int(m) for m in max_per_column]
        else:
            self.categorical_cardinalities_ = []

        self.model_ = TabTransformer(
            categories=self.categorical_cardinalities_,
            num_continuous=self.num_continuous_,
            transformer_dim=self.transformer_dim,
            depth=self.depth,
            heads=self.heads,
            dim_forward=self.dim_forward,
            dropout=self.dropout,
            mlp_hidden_dims=self.mlp_hidden_dims,
            num_classes=num_classes
        ).to(self.device_)

        param_count = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        if self.verbose:
            print(f"TabTransformer param count: {param_count}")

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_categ_t, X_cont_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for batch_cat, batch_cont, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_cat, batch_cont)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            self.train_losses_.append(avg_epoch_loss)

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_categ_val_t, X_cont_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
            else:
                val_loss = float('nan')

            self.val_losses_.append(val_loss)

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        return self

    def predict(self, X_categ, X_cont):
        self.model_.eval()
        X_categ_t = torch.from_numpy(X_categ).long() if X_categ.size > 0 else torch.empty((len(X_categ), 0), dtype=torch.long)
        X_cont_t  = torch.from_numpy(X_cont).float() if X_cont.size > 0 else torch.empty((len(X_categ), 0), dtype=torch.float)
        X_categ_t = X_categ_t.to(self.device_)
        X_cont_t  = X_cont_t.to(self.device_)
        with torch.no_grad():
            logits = self.model_(X_categ_t, X_cont_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X_categ, X_cont):
        self.model_.eval()
        X_categ_t = torch.from_numpy(X_categ).long() if X_categ.size > 0 else torch.empty((len(X_categ), 0), dtype=torch.long)
        X_cont_t  = torch.from_numpy(X_cont).float() if X_cont.size > 0 else torch.empty((len(X_categ), 0), dtype=torch.float)
        X_categ_t = X_categ_t.to(self.device_)
        X_cont_t  = X_cont_t.to(self.device_)
        with torch.no_grad():
            logits = self.model_(X_categ_t, X_cont_t)
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

    def score(self, X_categ, X_cont, y_true):
        preds = self.predict(X_categ, X_cont)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, preds)

    def get_train_val_losses(self):
        return self.train_losses_, self.val_losses_

