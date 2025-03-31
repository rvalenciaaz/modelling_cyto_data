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
