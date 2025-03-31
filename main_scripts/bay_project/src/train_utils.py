import torch
import pyro
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import optuna

from .model_utils import bayesian_nn_model, create_guide, create_svi

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
    Trains a Bayesian NN with Pyro (no separate validation).
    Returns (guide, list_of_train_losses).
    """
    pyro.clear_param_store()
    guide = create_guide(bayesian_nn_model)
    svi = create_svi(bayesian_nn_model, guide, learning_rate=learning_rate)

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
            print(f"[Epoch {epoch}] Train Loss: {loss:.4f}")

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
    guide = create_guide(bayesian_nn_model)
    svi = create_svi(bayesian_nn_model, guide, learning_rate=learning_rate)

    train_losses = []
    val_losses = []
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
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return guide, train_losses, val_losses


def objective_optuna(trial, X_train_t, y_train_t):
    """
    Objective function for Optuna to tune hyperparameters.
    Splits off a validation set from the training data and returns val accuracy.
    """
    hidden_size   = trial.suggest_int("hidden_size", 16, 128, step=16)
    num_layers    = trial.suggest_int("num_layers", 3, 50)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs_tune = 5000  # fewer epochs for faster tuning, this may be passed as a argument in tune hyperparameters

    X_tune_train, X_val, y_tune_train, y_val = train_test_split(
        X_train_t, y_train_t, test_size=0.2, random_state=42, stratify=y_train_t
    )

    guide, _ = train_pyro_model(
        X_tune_train,
        y_tune_train,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=len(np.unique(y_train_t)),
        learning_rate=learning_rate,
        num_epochs=num_epochs_tune,
        verbose=False
    )

    from .prediction_utils import predict_pyro_model
    val_preds = predict_pyro_model(
        X_val,
        guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=len(np.unique(y_train_t)),
        num_samples=300
    )
    return accuracy_score(y_val, val_preds)


def tune_hyperparameters(X_train_t, y_train_t, n_trials=40):
    """
    Runs an Optuna study to maximize accuracy on a hold-out validation subset.
    Returns the best hyperparams dict.
    """
    def objective(trial):
        return objective_optuna(trial, X_train_t, y_train_t)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=None)
    return study.best_params
