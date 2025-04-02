import glob
import os
import polars as pl

def read_and_combine_csv(label_prefix="species", pattern="species*.csv"):
    """
    Reads all CSV files matching the given pattern (e.g. 'species*.csv'),
    concatenates them, and adds a 'Label' column derived from the filename.
    """
    csv_files = glob.glob(pattern)
    df_list = []
    for file_path in csv_files:
        temp_df = pl.read_csv(file_path)
        # Extract the label from the filename, e.g. "species1.csv" => "1"
        label_str = os.path.splitext(os.path.basename(file_path))[0]  # e.g., "species1"
        label_str = label_str.replace(label_prefix, "")               # e.g., "1"
        temp_df = temp_df.with_columns(pl.lit(label_str).alias("Label"))
        df_list.append(temp_df)

    if not df_list:
        raise ValueError("No CSV files found matching pattern!")

    combined_df = pl.concat(df_list, how="vertical")
    return combined_df
