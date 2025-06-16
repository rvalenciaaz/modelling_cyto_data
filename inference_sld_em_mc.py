#!/usr/bin/env python3
# inference_sld_em_mc_bootstrap.py

"""
Inference + SLD-EM + Monte-Carlo-within-Bootstrap posterior plot
================================================================

This script

1. loads the saved Bayesian neural-network guide,
2. predicts soft class probabilities for every Monte-Carlo (MC) sample
   drawn from the Pyro guide (via `predict_pyro_probabilities` in src/prediction_utils.py),
3. feeds each MC draw through Saerens-Latinne-Decaestecker EM and wraps a
   row-level bootstrap around it, producing a posterior of π that reflects both
   model and sampling uncertainty,
4. writes the usual artefacts, and
5. plots the posterior densities.

Run:

    python inference_sld_em_mc_bootstrap.py

Outputs per input file
----------------------

- `<base>_predictions.csv`       — per-row predicted label, mean & std probs
- `<base>_class_mix_sld.json`    — MLE π̂ (K floats)
- `<base>_pi_samples.npy`        — B×K posterior draws of π
- `<base>_class_mix_posterior.png` — overlayed histograms for visual check
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt  # policy: no colours specified
import numpy as np
import polars as pl
import torch
import pyro  # noqa: F401 — required when unpickling guides
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ───────────────────────────── Local imports ──────────────────────────────── #
from src.model_utils import bayesian_nn_model, create_guide  # type: ignore
from src.prediction_utils import predict_pyro_model, predict_pyro_probabilities  # type: ignore

#####################################################################
#                        Hyper-parameters & paths                  #
#####################################################################

REPLICATION_FOLDER = "copied_files"
DEFAULT_NUM_SAMPLES = 1_000       # MC samples drawn from the Pyro guide
SLD_TOL = 1e-6
SLD_MAX_ITER = 1_000
BOOTSTRAP_B = 4_000               # posterior draws of π to produce

#####################################################################
#                           File helpers                           #
#####################################################################


def load_artifacts(replication_folder: str = REPLICATION_FOLDER):
    """Load guide + preprocessing objects saved during training."""
    def _load(name: str):
        with open(os.path.join(replication_folder, name), "rb") as fh:
            return pickle.load(fh)

    guide_state = _load("bayesian_nn_pyro_params.pkl")
    scaler: StandardScaler = _load("scaler.pkl")
    label_encoder: LabelEncoder = _load("label_encoder.pkl")

    with open(os.path.join(replication_folder, "features_to_keep.json")) as fh:
        features_to_keep: List[str] = json.load(fh)

    with open(os.path.join(replication_folder, "best_params.json")) as fh:
        best_params: dict = json.load(fh)

    hidden_size = best_params["hidden_size"]
    num_layers = best_params["num_layers"]
    output_dim = len(label_encoder.classes_)

    guide = create_guide(bayesian_nn_model)
    guide(
        torch.randn((1, len(features_to_keep))),
        torch.tensor([0]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
    )
    guide.load_state_dict(guide_state)
    guide.eval()

    return guide, scaler, label_encoder, features_to_keep, best_params


def read_file_list(list_path: str = "file_mock_list.txt") -> List[str]:
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"File list '{list_path}' not found.")
    with open(list_path, encoding="utf-8") as fh:
        return [
            ln.strip()
            for ln in fh
            if ln.strip() and not ln.lstrip().startswith("#")
        ]


#####################################################################
#                       SLD-EM implementation                      #
#####################################################################


def sld_em(
    prob_mat: np.ndarray,
    *,
    tol: float = SLD_TOL,
    max_iter: int = SLD_MAX_ITER,
    init: Literal["uniform", "soft"] | np.ndarray = "soft",
) -> np.ndarray:
    """Saerens-Latinne-Decaestecker EM algorithm (maximum likelihood)."""
    N, K = prob_mat.shape
    if isinstance(init, str):
        if init == "uniform":
            pi = np.full(K, 1.0 / K)
        elif init == "soft":
            pi = prob_mat.sum(axis=0)
            pi /= pi.sum()
        else:
            raise ValueError("init must be 'uniform', 'soft' or array-like")
    else:
        pi = np.asarray(init, dtype=float)
        pi /= pi.sum()

    for _ in range(max_iter):
        num = pi * prob_mat
        r = num / (num.sum(axis=1, keepdims=True) + 1e-12)
        pi_new = r.mean(axis=0)
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            return pi_new
        pi = pi_new

    raise RuntimeError("SLD-EM failed to converge within max_iter")


#####################################################################
#             Monte-Carlo-within-Bootstrap around SLD-EM            #
#####################################################################


def mc_within_bootstrap_pi(
    prob_samples: np.ndarray,                    # (S, N, K)
    *,
    B_total: int = BOOTSTRAP_B,
    tol: float = SLD_TOL,
    max_iter: int = SLD_MAX_ITER,
    init: Literal["uniform", "soft"] | np.ndarray = "soft",
) -> np.ndarray:
    """
    Combine BNN posterior uncertainty (S MC draws) with row-level bootstrap.

    Returns
    -------
    np.ndarray of shape (B_total, K)
    """
    S, N, K = prob_samples.shape
    B_per_mc = int(np.ceil(B_total / S))
    rng = np.random.default_rng()
    draws: list[np.ndarray] = []

    for s in range(S):
        probs_s = prob_samples[s]
        for _ in range(B_per_mc):
            idx = rng.choice(N, size=N, replace=True)
            draws.append(
                sld_em(
                    probs_s[idx],
                    tol=tol,
                    max_iter=max_iter,
                    init=init,
                )
            )

    return np.vstack(draws)[:B_total]


#####################################################################
#                              Plotting                            #
#####################################################################


def plot_class_mix_posterior(
    pi_samples: np.ndarray,
    class_ids: list[str] | np.ndarray,
    outfile: str | Path,
) -> None:
    """Overlay kernel-smoothed histograms for each class."""
    K = pi_samples.shape[1]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = 100
    for k in range(K):
        ax.hist(
            pi_samples[:, k],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=str(class_ids[k]),
        )
    ax.set_xlabel("Class proportion πₖ")
    ax.set_ylabel("Posterior density")
    ax.set_title("Class-mix posterior (MC-within-bootstrap)")
    ax.legend(title="Class")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"    ↳ Posterior plot saved to {outfile}")
    plt.close(fig)


#####################################################################
#                     Per-row prediction routine                   #
#####################################################################


def predict_new_data(
    df: pl.DataFrame,
    guide,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    features_to_keep: List[str],
    *,
    hidden_size: int,
    num_layers: int,
    output_dim: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
):
    """Return predictions + full probability tensor."""
    missing = set(features_to_keep) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df.select(features_to_keep).to_numpy()
    X = scaler.transform(X)
    X_t = torch.as_tensor(X, dtype=torch.float32)

    pred_ids = predict_pyro_model(
        X_t,
        guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples,
    )
    pred_labels = label_encoder.inverse_transform(pred_ids)

    mean_probs, std_probs, prob_samples = predict_pyro_probabilities(
        X_t,
        guide,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        num_samples=num_samples,
    )
    return pred_labels, mean_probs, std_probs, prob_samples


#####################################################################
#                               Main                               #
#####################################################################


def main_inference():
    guide, scaler, label_encoder, features_to_keep, best_params = load_artifacts()
    hidden_size = best_params["hidden_size"]
    num_layers = best_params["num_layers"]
    output_dim = len(label_encoder.classes_)
    class_ids = label_encoder.classes_.tolist()

    for data_path in read_file_list("file_mock_list.txt"):
        if not os.path.isfile(data_path):
            print(f"[WARN] '{data_path}' not found; skipping.")
            continue

        df = pl.read_csv(data_path)
        print(f"\n› Processing '{data_path}'  (shape={df.shape})")

        pred_labels, mean_probs, std_probs, prob_samples = predict_new_data(
            df,
            guide,
            scaler,
            label_encoder,
            features_to_keep,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=output_dim,
        )

        # (1) SLD-EM point estimate on the means
        pi_hat = sld_em(mean_probs, init="soft")

        # (2) MC-within-bootstrap posterior
        pi_samples = mc_within_bootstrap_pi(prob_samples, B_total=BOOTSTRAP_B)

        base = Path(data_path).stem

        # (a) CSV with per-row predictions
        csv_out = Path(REPLICATION_FOLDER) / f"{base}_predictions.csv"
        with open(csv_out, "w", newline="") as fh:
            writer = csv.writer(fh)
            header = (
                ["RowIndex", "PredictedClass"]
                + [f"MeanProb_{cid}" for cid in class_ids]
                + [f"StdProb_{cid}" for cid in class_ids]
            )
            writer.writerow(header)
            for i, lbl in enumerate(pred_labels):
                writer.writerow([i, lbl, *mean_probs[i], *std_probs[i]])
        print(f"    ↳ Per-row summary saved to {csv_out}")

        # (b) JSON with MLE π̂
        mix_out = Path(REPLICATION_FOLDER) / f"{base}_class_mix_sld.json"
        with open(mix_out, "w") as fh:
            json.dump({cid: float(p) for cid, p in zip(class_ids, pi_hat)}, fh, indent=2)
        print(f"    ↳ Class-mix (SLD-EM) saved to {mix_out}")

        # (c) NPY with posterior draws
        npy_out = Path(REPLICATION_FOLDER) / f"{base}_pi_samples.npy"
        np.save(npy_out, pi_samples)
        print(f"    ↳ Posterior draws saved to {npy_out}")

        # (d) Plot PNG
        png_out = Path(REPLICATION_FOLDER) / f"{base}_class_mix_posterior.png"
        plot_class_mix_posterior(pi_samples, class_ids, png_out)

    print("\nAll files processed. Inference + MC-within-bootstrap complete!")


if __name__ == "__main__":
    main_inference()
