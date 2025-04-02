import numpy as np
import polars as pl
from scipy.stats import median_abs_deviation

def mad_feature_filter(data: pl.DataFrame, label_col="Label", mad_threshold=5):
    """
    Computes MAD for numeric columns and selects those >= mad_threshold.
    Returns a new DataFrame with filtered numeric columns + the label_col.
    """
    # Identify numeric columns
    numeric_cols = []
    for c in data.columns:
        if data.schema[c] in [pl.Float64, pl.Int64]:
            numeric_cols.append(c)

    # Calculate MAD for each numeric column
    keep_features = []
    for col in numeric_cols:
        col_data = data[col].drop_nulls().to_numpy()
        if len(col_data) == 0:
            continue
        mad_val = median_abs_deviation(col_data, scale='normal')
        if mad_val >= mad_threshold:
            keep_features.append(col)

    # Return filtered DataFrame
    return data.select(keep_features + [label_col]), keep_features
