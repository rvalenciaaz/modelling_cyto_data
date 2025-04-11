#just histograms
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the probability samples from the pickle file.
with open("inference_probability_samples.pkl", "rb") as f:
    prob_samples = pickle.load(f)

# prob_samples is expected to have shape: (num_samples, n_data, output_dim)
num_samples, n_data, output_dim = prob_samples.shape
print("Loaded probability samples with shape:", prob_samples.shape)

# Compute the mean and standard deviation for each data point across samples (per class)
mean_probs = np.mean(prob_samples, axis=0)  # shape: (n_data, output_dim)
std_probs  = np.std(prob_samples, axis=0)    # shape: (n_data, output_dim)

# For each data point, find the winning class (the one with the highest mean probability)
max_means = []
corresponding_stds = []

for i in range(n_data):
    winning_class_idx = np.argmax(mean_probs[i])
    max_means.append(mean_probs[i, winning_class_idx])
    corresponding_stds.append(std_probs[i, winning_class_idx])

max_means = np.array(max_means)
corresponding_stds = np.array(corresponding_stds)

# --------------------------
# Histogram: Maximum Mean Probabilities
# --------------------------
plt.figure(figsize=(10, 6))
bins_means = np.arange(0, 1.01, 0.01)  # fixed bins from 0 to 1 with step 0.01
plt.hist(max_means, bins=bins_means, color="blue", alpha=0.7, edgecolor="black")
plt.xlabel("Maximum Mean Probability")
plt.ylabel("Frequency")
plt.title("Histogram of Maximum Mean Probabilities Across Data Points")
plt.tight_layout()
plt.savefig("histogram_max_mean_probabilities.png")
#plt.show()

# --------------------------
# Histogram: Standard Deviations for Winning Classes
# --------------------------
plt.figure(figsize=(10, 6))
bins_std = np.linspace(0, corresponding_stds.max(), 50)
plt.hist(corresponding_stds, bins=bins_std, color="green", alpha=0.7, edgecolor="black")
plt.xlabel("Standard Deviation of the Winning Class")
plt.ylabel("Frequency")
plt.title("Histogram of Standard Deviations for the Winning Class Across Data Points")
plt.tight_layout()
plt.savefig("histogram_winning_std.png")
#plt.show()
