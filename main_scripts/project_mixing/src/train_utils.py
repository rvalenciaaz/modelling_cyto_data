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


def objective_optuna(
    trial,
    X_train_t,
    y_train_t,
    num_epochs_tune
):
    """
    Objective function for Optuna to tune hyperparameters.
    Uses 'num_epochs_tune' from the caller for faster or slower tuning.
    """
    hidden_size   = trial.suggest_int("hidden_size", 16, 128, step=16)
    num_layers    = trial.suggest_int("num_layers", 3, 50)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # Split off a validation set from the training data
    from sklearn.model_selection import train_test_split
    X_tune_train, X_val, y_tune_train, y_val = train_test_split(
        X_train_t, y_train_t, test_size=0.2, random_state=42, stratify=y_train_t
    )

    # Train with the given epochs
    guide, _ = train_pyro_model(
        X_tune_train,
        y_tune_train,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=len(np.unique(y_train_t)),
        learning_rate=learning_rate,
        num_epochs=num_epochs_tune,  # <= used here
        verbose=False
    )

    # Predict on validation
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


def tune_hyperparameters(
    X_train_t,
    y_train_t,
    n_trials=40,
    num_epochs_tune=5000
):
    """
    Runs an Optuna study to maximize accuracy on a hold-out validation subset.
    'num_epochs_tune' can now be specified by the caller.
    """
    def objective(trial):
        return objective_optuna(trial, X_train_t, y_train_t, num_epochs_tune)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=None)
    return study.best_params
    
def nn_objective(trial, X_train, y_train, n_splits=3):
    """
    Optuna objective for the feedforward PyTorch approach.
    Uses K-Fold to get an average accuracy.
    """
    hidden_size   = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers    = trial.suggest_int('num_layers', 3, 30)
    dropout       = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [64, 128, 256])
    epochs        = 20  # fewer epochs for quick search

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Scale each fold
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

        preds_fold = clf.predict(X_val_fold_scaled)
        fold_acc = accuracy_score(y_val_fold, preds_fold)
        accuracy_scores.append(fold_acc)

    return np.mean(accuracy_scores)

def run_nn_optuna(X_train, y_train, n_trials=30):
    """
    Runs an Optuna study for the feedforward PyTorch classifier.
    Returns best_params, best_score.
    """
    def func(trial):
        return nn_objective(trial, X_train, y_train, n_splits=3)

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=n_trials)
    return study.best_params, study.best_value
