import os
import glob
import numpy as np
import pandas as pd

# Get all CSV files starting with "species" in the current directory
csv_files = glob.glob("species*.csv")

# Read and process CSV files into a list of DataFrames
dfs = [
    pd.read_csv(file).assign(Label=os.path.splitext(os.path.basename(file))[0])
    for file in csv_files
]

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)
print("Combined DataFrame head:")
print(combined_df.head())

# Remove "species" from the Label column
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

# Select only numerical columns
numerical_data = combined_df.select_dtypes(include=[np.number])

# Compute the coefficient of variation (CV) for each numerical column
cv = numerical_data.std() / numerical_data.mean()
cv = cv.replace([np.inf, -np.inf], np.nan)

# Compute the mean absolute deviation (MAD) for each numerical column
mad = numerical_data.mad()

# Create a DataFrame to display the CV and MAD results using original feature names
metrics_df = pd.DataFrame({
    'Feature': cv.index,
    'Coefficient of Variation': cv.values,
    'Mean Absolute Deviation': mad.values
})
print("Metrics DataFrame:")
print(metrics_df)
