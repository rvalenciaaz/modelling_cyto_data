import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap.umap_ as umap  # Requires 'umap-learn' package

# Fix random seeds for reproducibility
np.random.seed(42)

# ---------------------------------------------------------
# 1. READING & SUBSAMPLING CSV FILES
# ---------------------------------------------------------
print("Reading CSV files and subsampling...")

csv_files = glob.glob("species*.csv")
df_list = []

for file_path in csv_files:
    temp_df = pd.read_csv(file_path)
    # Subsample up to 10,000 rows
    temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)
    
    # Create label from filename, e.g. "species1.csv" -> "species1"
    label = file_path.split('.')[0]
    temp_df['Label'] = label
    df_list.append(temp_df)

combined_df = pd.concat(df_list, ignore_index=True)

# Clean up the 'Label' column (remove "species" prefix)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

print(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. SELECTING NUMERIC FEATURES
# ---------------------------------------------------------
print("Selecting numeric features...")

numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = combined_df[numeric_cols + ["Label"]].copy()

print(f"Number of numeric features: {len(numeric_cols)}")

# ---------------------------------------------------------
# 3. PREPARE DATA (SCALE)
# ---------------------------------------------------------
X = df_numeric.drop(columns=["Label"])
y = df_numeric["Label"]

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 4. PCA, UMAP, and t-SNE
# ---------------------------------------------------------
print("Performing PCA, UMAP, and t-SNE (2D)...")

# 4a. PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4b. UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

# 4c. t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# ---------------------------------------------------------
# 5. PLOTTING
# ---------------------------------------------------------
print("Plotting each 2D embedding...")

# Define your preferred numeric label order:
hue_order = [str(i) for i in range(1, 10)]

# 5a. PCA Plot
pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Label": y.values
})

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x="PC1", 
    y="PC2",
    hue="Label",
    hue_order=hue_order,      # Force legend order from "1" through "9"
    palette="tab10",
    alpha=0.7,
    edgecolor=None
)
plt.title("PCA (2D) of Combined Species Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("pca_scatter.png", dpi=150)
plt.close()

# 5b. UMAP Plot
umap_df = pd.DataFrame({
    "UMAP1": X_umap[:, 0],
    "UMAP2": X_umap[:, 1],
    "Label": y.values
})

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=umap_df,
    x="UMAP1", 
    y="UMAP2",
    hue="Label",
    hue_order=hue_order,      # Force legend order
    palette="tab10",
    alpha=0.7,
    edgecolor=None
)
plt.title("UMAP (2D) of Combined Species Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("umap_scatter.png", dpi=150)
plt.close()

# 5c. t-SNE Plot
tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "Label": y.values
})

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="Label",
    hue_order=hue_order,      # Force legend order
    palette="tab10",
    alpha=0.7,
    edgecolor=None
)
plt.title("t-SNE (2D) of Combined Species Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("tsne_scatter.png", dpi=150)
plt.close()

print("Done! Plots saved as 'pca_scatter.png', 'umap_scatter.png', and 'tsne_scatter.png'.")