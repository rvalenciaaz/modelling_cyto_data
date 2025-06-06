# prediction_utils.py

import torch
import numpy as np
from pyro.infer import Predictive

from .model_utils import bayesian_nn_model

def softmax_3d(logits_3d):
    """
    Applies softmax along the last dimension of a 3D array:
      logits_3d shape: (num_samples, n_data, output_dim)
    Returns probabilities of the same shape.
    """
    exp_vals = np.exp(logits_3d - np.max(logits_3d, axis=-1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def predict_pyro_model(
    X_tensor,
    guide,
    hidden_size=32,
    num_layers=2,
    output_dim=2,
    num_samples=500
):
    """
    Draw posterior samples for 'obs' and do a majority vote for final class predictions.
    """
    predictive = Predictive(
        bayesian_nn_model,
        guide=guide,
        num_samples=num_samples,
        return_sites=["obs"]
    )
    samples = predictive(
        X_tensor, None,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim
    )
    obs_samples = samples["obs"].cpu().numpy()  # shape: (num_samples, n_data)

    def majority_vote(row_samples):
        return np.bincount(row_samples, minlength=output_dim).argmax()

    final_preds = np.apply_along_axis(majority_vote, 0, obs_samples)
    return final_preds


def predict_pyro_probabilities(
    X_tensor,
    guide,
    hidden_size=32,
    num_layers=2,
    output_dim=2,
    num_samples=500
):
    """
    Draw posterior samples for 'logits' and apply softmax to get class probabilities.
    Returns:
      mean_probs: shape [n_data, output_dim]
      std_probs:  shape [n_data, output_dim]
      probs_samples: the full distribution of probabilities, shape [num_samples, n_data, output_dim]
    """
    predictive = Predictive(
        bayesian_nn_model,
        guide=guide,
        num_samples=num_samples,
        return_sites=["logits"]
    )
    samples = predictive(
        X_tensor, None,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim
    )
    # Get logits samples; expected shape might be (num_samples, 1, n_data, output_dim)
    logits_samples = samples["logits"].detach().cpu().numpy()

    # Squeeze the extra dimension if it's present
    if logits_samples.ndim == 4 and logits_samples.shape[1] == 1:
        logits_samples = np.squeeze(logits_samples, axis=1)
    
    # Now logits_samples should have shape (num_samples, n_data, output_dim)
    probs_samples = softmax_3d(logits_samples)
    
    # Compute summary stats along the sample axis
    mean_probs = probs_samples.mean(axis=0)  # shape: (n_data, output_dim)
    std_probs  = probs_samples.std(axis=0)   # shape: (n_data, output_dim)
    
    return mean_probs, std_probs, probs_samples

def predict_pyro_logits(
    X_tensor,
    guide,
    hidden_size=32,
    num_layers=2,
    output_dim=2,
    num_samples=500
):
    """
    Draw posterior samples for 'logits' and apply softmax to get class probabilities.
    Returns:
      mean_probs: shape [n_data, output_dim]
      std_probs:  shape [n_data, output_dim]
      probs_samples: the full distribution of probabilities, shape [num_samples, n_data, output_dim]
    """
    predictive = Predictive(
        bayesian_nn_model,
        guide=guide,
        num_samples=num_samples,
        return_sites=["logits"]
    )
    samples = predictive(
        X_tensor, None,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim
    )
    # Get logits samples; expected shape might be (num_samples, 1, n_data, output_dim)
    logits_samples = samples["logits"].detach().cpu().numpy()

    # Squeeze the extra dimension if it's present
    if logits_samples.ndim == 4 and logits_samples.shape[1] == 1:
        logits_samples = np.squeeze(logits_samples, axis=1)
    
    # Now logits_samples should have shape (num_samples, n_data, output_dim)
    #probs_samples = softmax_3d(logits_samples)
    
    # Compute summary stats along the sample axis
    mean_probs = logits_samples.mean(axis=0)  # shape: (n_data, output_dim)
    std_probs  = logits_samples.std(axis=0)   # shape: (n_data, output_dim)
    
    return mean_probs, std_probs, probs_samples
    
def predict_pytorch_proba(clf, X):
    """
    If you specifically want a separate function for prob. prediction
    with the PyTorchNNClassifierWithVal, but it's basically clf.predict_proba(X).
    """
    return clf.predict_proba(X)
