import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.infer.autoguide as autoguide

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
