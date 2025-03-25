import os
import json
import pickle
import polars as pl
import numpy as np
import torch
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import pyro.infer.autoguide as autoguide

def bayesian_nn_model(x, y=None, hidden_size=32, num_layers=2, output_dim=2):
    input_dim = x.shape[1]
    hidden = x
    current_dim = input_dim
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
        hidden = torch.tanh(torch.matmul(hidden, w) + b)
        current_dim = hidden_size
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
    logits = torch.matmul(hidden, w_out) + b_out
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
    return logits

def main():
    OUTPUT_FOLDER = "replication_files"

    # Load artifacts
    params_file = os.path.join(OUTPUT_FOLDER, "bayesian_nn_pyro_params.pkl")
    scaler_file = os.path.join(OUTPUT_FOLDER, "scaler.pkl")
    encoder_file = os.path.join(OUTPUT_FOLDER, "label_encoder.pkl")
    features_file = os.path.join(OUTPUT_FOLDER, "features_to_keep.json")
    metrics_file  = os.path.join(OUTPUT_FOLDER, "metrics_pyro.json")

    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Missing guide parameters: {params_file}")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Missing scaler file: {scaler_file}")
    if not os.path.exists(encoder_file):
        raise FileNotFoundError(f"Missing label encoder: {encoder_file}")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Missing features-to-keep file: {features_file}")

    with open(params_file, "rb") as f:
        saved_state_dict = pickle.load(f)
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_file, "rb") as f:
        label_encoder = pickle.load(f)
    with open(features_file, "r") as f:
        features_to_keep = json.load(f)

    # Optional: load best hyperparams from metrics
    best_hidden_size = 32
    best_num_layers = 2
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
        best_params = metrics_data["best_hyperparams"]
        best_hidden_size = best_params["hidden_size"]
        best_num_layers  = best_params["num_layers"]

    output_dim = len(label_encoder.classes_)

    # Rebuild guide
    guide = autoguide.AutoNormal(bayesian_nn_model)
    dummy_x = torch.randn((1, len(features_to_keep)))
    dummy_y = torch.tensor([0])
    guide(dummy_x, dummy_y, hidden_size=best_hidden_size, num_layers=best_num_layers, output_dim=output_dim)
    guide.load_state_dict(saved_state_dict)
    guide.eval()

    # Load new data
    NEW_DATA_FILE = "new_dataset.csv"
    if not os.path.exists(NEW_DATA_FILE):
        print(f"New data file {NEW_DATA_FILE} not found. Exiting.")
        return

    new_df = pl.read_csv(NEW_DATA_FILE)
    existing_cols = [c for c in new_df.columns if c in features_to_keep]
    new_df_filtered = new_df.select(existing_cols).select(features_to_keep)
    X_new = new_df_filtered.to_numpy()
    X_new_scaled = scaler.transform(X_new)
    X_new_t = torch.tensor(X_new_scaled, dtype=torch.float32)

    # Get posterior samples
    num_samples_posterior = 500
    predictive = Predictive(
        bayesian_nn_model,
        guide=guide,
        num_samples=num_samples_posterior
    )
    samples = predictive(
        X_new_t, None,
        hidden_size=best_hidden_size,
        num_layers=best_num_layers,
        output_dim=output_dim
    )
    obs_samples = samples["obs"].cpu().numpy()  # shape: (num_samples, n_rows)

    # Print normal predictions (majority vote)
    def majority_vote(row_samples):
        return np.bincount(row_samples, minlength=output_dim).argmax()

    final_preds = np.apply_along_axis(majority_vote, 0, obs_samples)
    decoded_preds = label_encoder.inverse_transform(final_preds)
    print("\nMajority-vote predictions on new data:")
    for i, pred_label in enumerate(decoded_preds):
        print(f"Row {i} => {pred_label}")

    # Plot posterior distribution for selected rows
    # Example: plot for rows 0, 1, and 2, if they exist
    selected_rows = [0, 1, 2]
    class_names = label_encoder.classes_

    for row_idx in selected_rows:
        if row_idx >= X_new_t.shape[0]:
            break  # no more rows
        row_samples = obs_samples[:, row_idx]  # shape (num_samples_posterior,)
        counts = np.bincount(row_samples, minlength=output_dim)

        plt.figure()
        plt.bar(range(output_dim), counts)
        plt.xticks(range(output_dim), class_names, rotation=45)
        plt.xlabel("Class")
        plt.ylabel("Count of Posterior Samples")
        plt.title(f"Posterior Class Distribution for Row {row_idx}")
        plt.grid(True)
        plt.tight_layout()

        out_plot_name = f"posterior_distribution_row{row_idx}.png"
        out_plot_path = os.path.join(OUTPUT_FOLDER, out_plot_name)
        plt.savefig(out_plot_path)
        plt.close()
        print(f"Saved posterior distribution plot for row {row_idx} => {out_plot_path}")

    print("\nDone full posterior prediction with class distribution plots.")

if __name__ == "__main__":
    main()

