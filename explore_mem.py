import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import median_abs_deviation

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def get_numeric_cols(sample_file, label_prefix="species"):
    """
    Quickly read a small chunk of the first CSV just to identify which columns are numeric
    (excluding the Label).
    """
    # Read just a small piece to infer dtypes
    df_sample = pd.read_csv(sample_file, nrows=100)
    # Potentially rename the label column or exclude it if it exists
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols

def read_in_chunks(file, usecols=None, chunksize=50_000):
    """
    Generator function that reads a CSV in chunks, yielding each chunk.
    """
    for chunk in pd.read_csv(file, usecols=usecols, chunksize=chunksize):
        yield chunk

# ---------------------------------------------------------------------
# STEP 0: Identify your CSVs and which columns are numeric
# ---------------------------------------------------------------------
csv_files = glob.glob("species*.csv")
csv_files.sort()  # just for consistency

# We'll gather numeric column names from the first file:
if not csv_files:
    raise ValueError("No CSV files found matching 'species*.csv'!")

numeric_columns = get_numeric_cols(csv_files[0])
# We'll keep 'Label' separately
all_columns = numeric_columns + ['Label']

# ---------------------------------------------------------------------
# STEP 1: FIRST PASS - Compute median for each numeric column
#         (We do it across *all* CSV files in a chunked manner.)
# ---------------------------------------------------------------------

# We need to store *all values* for each column or use a streaming median approach.
# If storing all values is too large, consider an approximate approach.
all_values = {col: [] for col in numeric_columns}

for file in csv_files:
    # Extract the label from filename, e.g., species1.csv -> '1'
    label_name = file.split('.')[0].replace("species", "")
    
    # We'll read only numeric columns plus we can handle the label separately.
    # In pass 1, we only need numeric data for medians.
    for chunk in read_in_chunks(file, usecols=numeric_columns):
        # Convert columns to float32 or a smaller dtype if possible (helps memory).
        chunk = chunk.astype(np.float32, copy=False)
        
        # Accumulate values for each column
        for col in numeric_columns:
            # extend the list
            all_values[col].extend(chunk[col].values)

# Now compute the median for each numeric column
medians = {}
for col in numeric_columns:
    medians[col] = np.median(all_values[col])
    # We can free memory once done
    all_values[col] = []  # clear out

# ---------------------------------------------------------------------
# STEP 2: SECOND PASS - Compute Median Absolute Deviation (MAD)
#         We know the medians, so now we can compute absolute deviations.
# ---------------------------------------------------------------------
abs_devs = {col: [] for col in numeric_columns}

for file in csv_files:
    for chunk in read_in_chunks(file, usecols=numeric_columns):
        # convert to float32
        chunk = chunk.astype(np.float32, copy=False)
        
        for col in numeric_columns:
            # absolute deviation from the median
            dev = np.abs(chunk[col] - medians[col])
            abs_devs[col].extend(dev.values)

# Now compute the MAD for each column
mads = {}
for col in numeric_columns:
    # scipy.stats.median_abs_deviation by default normalizes; 
    # we can pass scale='raw' to get raw median of absolute deviations
    # or just compute ourselves with np.median.
    mads[col] = median_abs_deviation(abs_devs[col], scale='raw')
    abs_devs[col] = []  # free memory

# ---------------------------------------------------------------------
# STEP 3: FILTER columns by your MAD <= 5 criterion
# ---------------------------------------------------------------------
good_cols = [col for col in numeric_columns if mads[col] <= 5.0]
print("Columns passing MAD <= 5:", good_cols)

# ---------------------------------------------------------------------
# STEP 4: FINAL READ (optional) - Build the final (reduced) DataFrame
#         If you truly need one big DataFrame in memory, do it now but
#         only load the good columns + Label. This should be smaller.
# ---------------------------------------------------------------------
final_dfs = []
for file in csv_files:
    label_name = file.split('.')[0].replace("species", "")
    usecols_needed = good_cols[:]  # copy of good_cols
    # We also want to read the label column:
    # If label column is in the CSV, we can just specify it in usecols.
    # Or we can read it separately. We'll assume it's named exactly "Label" in the CSV
    # If not, you might read the entire file once and rename it, etc.
    
    # If you *don't* already have "Label" in the file as a column, you'll have to
    # read the entire set of columns or handle differently. 
    # Here, let's assume "speciesX.csv" does NOT have a 'Label' column by default,
    # and we are *adding* it ourselves. 
    # So we read only the good numeric columns:
    chunk_iter = pd.read_csv(file, usecols=good_cols, chunksize=50_000)
    
    for chunk in chunk_iter:
        chunk = chunk.astype(np.float32, copy=False)
        # Add label
        chunk['Label'] = label_name  # e.g., "1", "2", ...
        final_dfs.append(chunk)

# Concatenate all smaller chunks
final_df = pd.concat(final_dfs, ignore_index=True)
del final_dfs  # free up the list

# final_df now contains only the "good" numeric columns + "Label"
print(final_df.head())
print("Final DataFrame shape:", final_df.shape)
