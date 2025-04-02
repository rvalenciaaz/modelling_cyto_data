import pandas as pd
import glob
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from scipy.stats import median_abs_deviation
# Use glob to get all CSV files starting with "species" in the current directory
csv_files = glob.glob("species*.csv")

# List to hold DataFrames
dfs = []

for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract label from the filename, e.g., 'species1.csv' -> 'species1'
    label = file.split('.')[0]
    
    # Add the Label column
    df['Label'] = label
    
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Display the first few rows of the concatenated DataFrame
print(combined_df.head())

combined_df["Label"]=combined_df["Label"].str.replace("species","")

# Assume `data` is your DataFrame (after any cleaning such as removing constant columns)
numerical_data = combined_df.select_dtypes(include=[np.number])

cv_results = {}

for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val = numerical_data[col].std()
    # Avoid division by zero: if mean is 0, set CV to NaN.
    cv = std_val / mean_val if mean_val != 0 else np.nan
    mad = median_abs_deviation(numerical_data[col].values)

    subscript = col #match.group(1) if match else col

    cv_results[subscript] = [subscript,cv,mad]

# Convert the results into a DataFrame for easy viewing
cv_df = pd.DataFrame(list(cv_results.values()), 
                     columns=['Feature', 'Coefficient of Variation','MAD'])

cv_2=cv_df.loc[cv_df["MAD"]<=5].copy()

com_after_low=combined_df[cv_2["Feature"].tolist()].copy()
print(com_after_low)
