import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import polars as pl

def load_predictions(predictions_csv="inference_predictions.csv"):
    # Use Polars for fast CSV reading
    return pl.read_csv(predictions_csv)

def load_probability_samples(prob_samples_file="inference_probability_samples.pkl"):
    with open(prob_samples_file, "rb") as f:
        return pickle.load(f)

def filter_predictions():
    # Load predictions and probability samples
    predictions_df = load_predictions()
    prob_samples = load_probability_samples()

    # Compute median probabilities over samples for each data point.
    # prob_samples has shape: (num_samples, n_data, output_dim)
    median_probs = np.median(prob_samples, axis=0)  # shape: (n_data, output_dim)

    # Compute the winning (maximum) median probability for each data point.
    winning_median_probs = np.max(median_probs, axis=1)  # shape: (n_data,)

    # Create a boolean mask for rows where the winning median probability is above 0.9.
    mask = winning_median_probs > 0.9
    print(f"Rows with winning median prob > 0.9: {np.count_nonzero(mask)} out of {winning_median_probs.shape[0]}")

    # Add the winning median probability as a new column to the predictions DataFrame.
    predictions_df = predictions_df.with_columns(pl.Series("win_median_prob", winning_median_probs))

    # Use Polars' filter method to keep only rows with win_median_prob > 0.9.
    filtered_df = predictions_df.filter(pl.col("win_median_prob") > 0.95)

    # Drop the temporary column.
    filtered_df = filtered_df.drop("win_median_prob")

    # Save the filtered predictions to a new CSV file.
    filtered_csv = "inference_filtered.csv"
    filtered_df.write_csv(filtered_csv)
    print(f"Saved filtered predictions => {filtered_csv}")

    # Group by "PredictedClass" column to get frequency counts.
    group_df = filtered_df.group_by("PredictedClass").agg(pl.count("PredictedClass").alias("count"))
    species = group_df["PredictedClass"].to_numpy()
    counts = group_df["count"].to_numpy()

    # Generate and save a barplot of the predicted class frequency.
    plt.figure(figsize=(10, 6))
    plt.bar(species, counts)
    plt.xlabel("Species")
    plt.ylabel("Predicted Class Frequency")
    plt.title("Frequency of Predicted Classes (Filtered: Winning Median Prob > 0.9)")
    barplot_filename = "predicted_class_frequency.png"
    plt.savefig(barplot_filename)
    plt.close()
    print(f"Saved barplot => {barplot_filename}")

if __name__ == "__main__":
    filter_predictions()
