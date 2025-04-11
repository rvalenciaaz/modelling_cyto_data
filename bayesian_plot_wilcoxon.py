import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import os

# Load the probability samples from the pickle file.
with open("inference_probability_samples.pkl", "rb") as f:
    prob_samples = pickle.load(f)

# prob_samples is expected to have shape: (num_samples, n_data, output_dim)
num_samples, n_data, output_dim = prob_samples.shape
print(f"Loaded probability samples with shape: {prob_samples.shape}")

# Define fixed bins with a step size of 0.01 (from 0 to 1 inclusive)
bins = np.arange(0, 1.01, 0.01)

# Set significance threshold for the Wilcoxon signed-rank test.
alpha = 0.05

# Choose a colormap with enough distinct colors (here 'tab10').
colors = plt.get_cmap("tab10").colors

# Create an output directory for plots.
output_dir = "overlap_histograms"
os.makedirs(output_dir, exist_ok=True)

# Loop over all data points. (Consider limiting this if n_data is very large.)
for i in range(n_data):
    # Compute the mean probability for each class for data point i.
    means = prob_samples[:, i, :].mean(axis=0)  # shape: (output_dim,)

    # Identify the two classes with the highest mean probabilities.
    sorted_indices = np.argsort(means)[::-1]  # descending order
    idx1, idx2 = sorted_indices[:2]

    # Retrieve the probability samples for these two classes.
    samples1 = prob_samples[:, i, idx1]
    samples2 = prob_samples[:, i, idx2]

    # Perform the Wilcoxon signed-rank test on these paired samples.
    try:
        stat, p_value = wilcoxon(samples1, samples2)
    except Exception as e:
        print(f"Data point {i}: Wilcoxon test failed with error: {e}")
        continue

    # If the p-value is above alpha, assume the distributions overlap.
    if p_value > alpha:
        plt.figure(figsize=(10, 6))
        # Plot histograms for all classes.
        for class_idx in range(output_dim):
            class_probs = prob_samples[:, i, class_idx]
            plt.hist(
                class_probs,
                bins=bins,
                alpha=0.5,
                label=f"Species {class_idx + 1} (mean: {means[class_idx]:.2f})",
                color=colors[class_idx % len(colors)]
            )
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.title(f"Data Point {i}: Wilcoxon p-value = {p_value:.3f}")
        plt.legend()
        plt.tight_layout()

        # Save the figure.
        plot_filename = os.path.join(output_dir, f"overlap_data_point_{i}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Data point {i} plotted (p-value = {p_value:.3f}).")
