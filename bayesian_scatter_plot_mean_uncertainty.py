import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the probability samples from the pickle file.
with open("inference_probability_samples.pkl", "rb") as f:
    prob_samples = pickle.load(f)

# prob_samples is expected to have shape: (num_samples, n_data, output_dim)
num_samples, n_data, output_dim = prob_samples.shape
print("Loaded probability samples with shape:", prob_samples.shape)

# --------------------------
# Compute mean and standard deviation (std) for each data point per class
# --------------------------
mean_probs = np.mean(prob_samples, axis=0)  # shape: (n_data, output_dim)
std_probs  = np.std(prob_samples, axis=0)    # shape: (n_data, output_dim)

# For each data point, select the winning class based on maximum mean probability
max_means = []
corresponding_stds = []
for i in range(n_data):
    winning_class_idx = np.argmax(mean_probs[i])
    max_means.append(mean_probs[i, winning_class_idx])
    corresponding_stds.append(std_probs[i, winning_class_idx])
max_means = np.array(max_means)
corresponding_stds = np.array(corresponding_stds)

# Scatter plot: Maximum Mean Probability vs. Standard Deviation
plt.figure(figsize=(10, 6))
plt.scatter(max_means, corresponding_stds, alpha=0.7, color="purple")
plt.xlabel("Maximum Mean Probability")
plt.ylabel("Standard Deviation")
plt.title("Scatter Plot: Maximum Mean Probability vs. Standard Deviation")

# Add vertical dashed lines at every 0.05 step and annotate count above each threshold.
ymax = np.max(corresponding_stds)
for thresh in np.arange(0.05, 1.05, 0.05):
    plt.axvline(x=thresh, color='gray', linestyle='dashed')
    count = np.sum(max_means >= thresh)
    plt.text(thresh, ymax * 0.95, f"{count}", rotation=90, va='top', ha='center', color='black')
plt.tight_layout()
plt.savefig("scatter_max_mean_vs_std.png")
#plt.show()

# --------------------------
# Compute median and median absolute deviation (MAD) for each data point per class.
# --------------------------
# Compute median per class for each data point.
median_probs = np.median(prob_samples, axis=0)  # shape: (n_data, output_dim)

# Compute the MAD:
# For each element, subtract the median (computed per data point per class), take the absolute value, then median over samples.
mad_probs = np.median(np.abs(prob_samples - np.median(prob_samples, axis=0, keepdims=True)), axis=0)

# For each data point, select the winning class based on maximum median probability.
winning_medians = []
winning_mads = []
for i in range(n_data):
    winning_class_idx = np.argmax(median_probs[i])
    winning_medians.append(median_probs[i, winning_class_idx])
    winning_mads.append(mad_probs[i, winning_class_idx])
winning_medians = np.array(winning_medians)
winning_mads = np.array(winning_mads)

# --------------------------
# Scatter plot: Winning Median vs. MAD (Median Absolute Deviation)
# --------------------------
plt.figure(figsize=(10, 6))
plt.scatter(winning_medians, winning_mads, alpha=0.7, color="orange")
plt.xlabel("Winning Median Probability")
plt.ylabel("Median Absolute Deviation (MAD)")
plt.title("Scatter Plot: Winning Median vs. MAD")

ymax_mad = np.max(winning_mads)
for thresh in np.arange(0.05, 1.05, 0.05):
    plt.axvline(x=thresh, color='gray', linestyle='dashed')
    count = np.sum(winning_medians >= thresh)
    plt.text(thresh, ymax_mad * 0.95, f"{count}", rotation=90, va='top', ha='center', color='black')
plt.tight_layout()
plt.savefig("scatter_winning_median_vs_mad.png")
#plt.show()
