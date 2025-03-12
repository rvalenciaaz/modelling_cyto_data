# data_utils.py
import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

def load_and_combine_csv(pattern="species*.csv", subsample=None):
    """
    Reads multiple CSV files matching `pattern`. 
    If subsample is not None, each file is subsampled.
    Adds a 'Label' column derived from filename.
    Returns the combined dataframe.
    """
    csv_files = glob.glob(pattern)
    df_list = []
    for file_path in csv_files:
        temp_df = pd.read_csv(file_path)
        # Subsample if requested
        if subsample is not None:
            temp_df = temp_df.sample(n=min(len(temp_df), subsample), random_state=42)
        label = file_path.split('.')[0]  # e.g. "species1.csv" -> "species1"
        temp_df['Label'] = label
        df_list.append(temp_df)

    combined_df = pd.concat(df_list, ignore_index=True)
    # Optionally remove "species" prefix
    combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

    return combined_df

def mad_filter(df, label_col="Label", mad_threshold=5):
    """
    Filters numeric features by Median Absolute Deviation (MAD).
    Keeps columns with MAD >= mad_threshold + the label column.
    Returns a filtered dataframe.
    """
    import warnings
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")  # for small subsets

    numerical_data = df.select_dtypes(include=[np.number])
    mad_values = {}

    for col in numerical_data.columns:
        col_arr = numerical_data[col].values
        mad_val = median_abs_deviation(col_arr, scale='normal')
        mad_values[col] = mad_val

    features_to_keep = [col for col, m in mad_values.items() if m >= mad_threshold]
    features_to_keep.append(label_col)
    filtered_df = df[features_to_keep].copy()
    return filtered_df
