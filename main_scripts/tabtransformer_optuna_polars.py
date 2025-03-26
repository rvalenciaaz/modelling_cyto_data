import os
import glob
import json
import datetime
import numpy as np

# Polars
import polars as pl

# Scipy
from scipy.stats import median_abs_deviation

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Sklearn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# For hyperparameter optimization
import optuna

# For plotting
import matplotlib.pyplot as plt

# Fix seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Make a folder to store replication files and artifacts
os.makedirs("artifacts", exist_ok=True)

# ---------------------------------------------------------
# 0. LOGGING UTILITY FOR TIMESTAMPS
# ---------------------------------------------------------
log_steps = []
def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# ---------------------------------------------------------
# 1. READING & SUBSAMPLING CSV FILES WITH POLARS
# ---------------------------------------------------------
log_message("Reading CSV files (species*.csv) using Polars and subsampling if needed...")
csv_files = glob.glob("species*.csv")  # Will get a list of species*.csv filenames

df_list = []
for file_path in csv_files:
    temp_df = pl.read_csv(file_path)
    # Optional subsampling, e.g. limit to 10,000 rows
    # temp_df = temp_df.sample(frac=min(1.0, 10_000 / temp_df.height), seed=42)

    # Extract label from the filename, e.g. "species1.csv" -> "species1"
    label = file_path.split('.')[0]  # e.g. "species1"

    # Add a Label column
    temp_df = temp_df.with_columns(pl.lit(label).alias("Label"))
    df_list.append(temp_df)

# Concatenate all data
combined_df = pl.concat(df_list, how="vertical")

# Remove the literal "species" prefix in the Label column
combined_df = combined_df.with_columns([
    pl.col("Label").str.replace("species", "", literal=True)
])

log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. FILTERING NUMERIC FEATURES BASED ON MAD
# ---------------------------------------------------------
log_message("Filtering numeric features based on MAD...")

# Select numeric columns
numeric_cols_polars = combined_df.select(pl.col(pl.INTEGER_DTYPES + pl.FLOAT_DTYPES))
numeric_col_names = numeric_cols_polars.columns

cv_results = []
for col in numeric_col_names:
    # Mean
    mean_val = numeric_cols_polars.select(pl.col(col).mean()).to_numpy()[0, 0]
    # Std
    std_val = numeric_cols_polars.select(pl.col(col).std()).to_numpy()[0, 0]

    # Coefficient of Variation (CV)
    cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

    # Median Absolute Deviation (MAD)
    col_vals = numeric_cols_polars.select(pl.col(col)).to_numpy()[:, 0]
    mad = median_abs_deviation(col_vals, scale='normal')

    cv_results.append((col, cv, mad))

cv_df = pl.DataFrame(cv_results, schema=["Feature", "CV", "MAD"])

MAD_THRESHOLD = 5
features_to_keep = (
    cv_df
    .filter(pl.col("MAD") >= MAD_THRESHOLD)
    .select("Feature")
    .to_series()
    .to_list()
)

# Keep only the filtered numeric columns + Label
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df.select(cols_to_keep)
log_message(f"Number of numeric features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 2.1 IDENTIFY CATEGORICAL VS. NUMERIC COLUMNS
# ---------------------------------------------------------
categorical_cols = []
numeric_cols = []

for col in final_df.columns:
    if col == "Label":
        continue
    # Check dtype
    dtype = final_df.schema[col]
    # Count unique
    unique_count = final_df.select(pl.col(col).n_unique()).to_numpy()[0, 0]

    # Heuristics: if it's string-like or has very few unique values, treat as categorical
    if dtype == pl.Utf8 or unique_count < 20:
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)

log_message(f"Identified potential categorical columns: {categorical_cols}")
log_message(f"Identified numeric columns: {numeric_cols}")

# ---------------------------------------------------------
# 2.2 ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------------
label_encoders = {}

temp_df_for_encoding = final_df  # We'll mutate this as we encode
for c in categorical_cols:
    le = LabelEncoder()
    # Extract column values from Polars, convert all to string if needed
    col_vals = temp_df_for_encoding.select(pl.col(c)).to_series().to_list()
    col_vals_str = [str(x) for x in col_vals]

    encoded_vals = le.fit_transform(col_vals_str)
    label_encoders[c] = le

    # Replace with the encoded values
    temp_df_for_encoding = temp_df_for_encoding.with_columns(
        pl.Series(name=c, values=encoded_vals)
    )

final_df = temp_df_for_encoding

# ---------------------------------------------------------
# 3. PREPARE FEATURES & LABELS
# ---------------------------------------------------------
# Convert final polars DF to numpy for scikit-learn / PyTorch processing
X_categ = final_df.select(categorical_cols).to_numpy() if len(categorical_cols) > 0 else np.empty((final_df.shape[0], 0))
X_numeric = final_df.select(numeric_cols).to_numpy() if len(numeric_cols) > 0 else np.empty((final_df.shape[0], 0))

y_vals = final_df.select("Label").to_numpy().ravel()  # shape (N,)

main_label_encoder = LabelEncoder()
y_encoded = main_label_encoder.fit_transform(y_vals)

# ---------------------------------------------------------
# 3.1 TRAIN/TEST SPLIT
# ---------------------------------------------------------
log_message("Splitting data into training and test sets...")
X_categ_train, X_categ_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_categ, X_numeric, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
log_message(f"Train set size: {X_categ_train.shape[0]}, Test set size: {X_categ_test.shape[0]}")

# ---------------------------------------------------------
# 3.2 OPTIONAL SCALING FOR NUMERIC FEATURES
# ---------------------------------------------------------
scaler = StandardScaler()
if X_num_train.shape[1] > 0:
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_test  = scaler.transform(X_num_test)

# ---------------------------------------------------------
# 4. DEFINE THE TABTRANSFORMER MODULE
# ---------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

class TabTransformer(nn.Module):
    def __init__(
        self,
        categories: list,        # list of cardinalities for each categorical column
        num_continuous: int,
        transformer_dim: int = 32,
        depth: int = 2,
        heads: int = 2,
        dim_forward: int = 64,
        dropout: float = 0.1,
        mlp_hidden_dims: list = [64, 32],
        num_classes: int = 2
    ):
        super().__init__()
        self.num_categs = len(categories)
        self.num_continuous = num_continuous
        self.transformer_dim = transformer_dim

        self.category_embeds = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=transformer_dim)
            for cardinality in categories
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(dim=transformer_dim,
                                    num_heads=heads,
                                    mlp_hidden_dim=dim_forward,
                                    dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(transformer_dim)

        mlp_input_dim = transformer_dim + num_continuous
        mlp_layers = []
        in_dim = mlp_input_dim
        for hdim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, hdim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = hdim
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x_categ, x_cont):
        batch_size = x_categ.shape[0]

        # Embed categorical features
        if self.num_categs > 0:
            cat_embeds = []
            for i, embed in enumerate(self.category_embeds):
                cat_embeds.append(embed(x_categ[:, i]))
            x_cat = torch.stack(cat_embeds, dim=1)
        else:
            # If no categorical features, create a dummy zero tensor
            x_cat = torch.zeros(batch_size, 1, self.transformer_dim, device=x_categ.device)

        # CLS token
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        if self.num_categs > 0:
            x_cat = torch.cat([cls_token_expanded, x_cat], dim=1)
        else:
            x_cat = cls_token_expanded

        # Pass through transformer layers
        for encoder in self.transformer_encoders:
            x_cat = encoder(x_cat)
        x_cat = self.norm(x_cat)

        # Take the CLS embedding and concat with continuous
        x_cls = x_cat[:, 0, :]
        x_full = torch.cat([x_cls, x_cont], dim=-1)

        # Final classification head
        out = self.mlp_head(x_full)
        return out

# ---------------------------------------------------------
# 5. DEFINE A PYTORCH CLASSIFIER (SCIKIT-LEARN STYLE)
# ---------------------------------------------------------
class TabTransformerClassifierWithVal:
    def __init__(self,
                 transformer_dim=32,
                 depth=2,
                 heads=2,
                 dim_forward=64,
                 dropout=0.1,
                 mlp_hidden_dims=[64, 32],
                 learning_rate=1e-3,
                 batch_size=32,
                 epochs=10,
                 verbose=True):
        self.transformer_dim = transformer_dim
        self.depth = depth
        self.heads = heads
        self.dim_forward = dim_forward
        self.dropout = dropout
        self.mlp_hidden_dims = mlp_hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.classes_ = None
        self.train_losses_ = []
        self.val_losses_ = []
        self.num_continuous_ = 0
        self.categorical_cardinalities_ = []

    def fit(self, X_categ, X_cont, y, X_categ_val=None, X_cont_val=None, y_val=None):
        X_categ_t = torch.from_numpy(X_categ).long() if X_categ.size > 0 else torch.empty((len(y), 0), dtype=torch.long)
        X_cont_t  = torch.from_numpy(X_cont).float() if X_cont.size > 0 else torch.empty((len(y), 0), dtype=torch.float)
        y_t       = torch.from_numpy(y).long()

        self.num_continuous_ = X_cont_t.shape[1]
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        has_val = (X_categ_val is not None) and (X_cont_val is not None) and (y_val is not None)

        X_categ_t = X_categ_t.to(self.device_)
        X_cont_t  = X_cont_t.to(self.device_)
        y_t       = y_t.to(self.device_)

        if has_val:
            X_categ_val_t = torch.from_numpy(X_categ_val).long() if X_categ_val.size > 0 else torch.empty((len(y_val), 0), dtype=torch.long)
            X_cont_val_t  = torch.from_numpy(X_cont_val).float() if X_cont_val.size > 0 else torch.empty((len(y_val), 0), dtype=torch.float)
            y_val_t       = torch.from_numpy(y_val).long()

            X_categ_val_t = X_categ_val_t.to(self.device_)
            X_cont_val_t  = X_cont_val_t.to(self.device_)
            y_val_t       = y_val_t.to(self.device_)

        if X_categ.shape[1] > 0:
            max_per_column = (X_categ.max(axis=0) + 1).tolist()
            self.categorical_cardinalities_ = [int(m) for m in max_per_column]
        else:
            self.categorical_cardinalities_ = []

        self.model_ = TabTransformer(
            categories=self.categorical_cardinalities_,
            num_continuous=self.num_continuous_,
            transformer_dim=self.transformer_dim,
            depth=self.depth,
            heads=self.heads,
            dim_forward=self.dim_forward,
            dropout=self.dropout,
            mlp_hidden_dims=self.mlp_hidden_dims,
            num_classes=num_classes
        ).to(self.device_)

        param_count = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        if self.verbose:
            print(f"TabTransformer param count: {param_count}")

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_categ_t, X_cont_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for batch_cat, batch_cont, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_cat, batch_cont)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            self.train_losses_.append(avg_epoch_loss)

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_categ_val_t, X_cont_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
            else:
                val_loss = np.nan

            self.val_losses_.append(val_loss)

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        return self

    def predict(self, X_categ, X_cont):
        self.model_.eval()
        X_categ_t = torch.from_numpy(X_categ).long() if X_categ.size > 0 else torch.empty((len(X_categ), 0), dtype=torch.long)
        X_cont_t  = torch.from_numpy(X_cont).float() if X_cont.size > 0 else torch.empty((len(X_categ), 0), dtype=torch.float)
        X_categ_t = X_categ_t.to(self.device_)
        X_cont_t  = X_cont_t.to(self.device_)
        with torch.no_grad():
            logits = self.model_(X_categ_t, X_cont_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X_categ, X_cont):
        self.model_.eval()
        X_categ_t = torch.from_numpy(X_categ).long() if X_categ.size > 0 else torch.empty((len(X_categ), 0), dtype=torch.long)
        X_cont_t  = torch.from_numpy(X_cont).float() if X_cont.size > 0 else torch.empty((len(X_categ), 0), dtype=torch.float)
        X_categ_t = X_categ_t.to(self.device_)
        X_cont_t  = X_cont_t.to(self.device_)
        with torch.no_grad():
            logits = self.model_(X_categ_t, X_cont_t)
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

    def score(self, X_categ, X_cont, y_true):
        preds = self.predict(X_categ, X_cont)
        return accuracy_score(y_true, preds)

    def get_train_val_losses(self):
        return self.train_losses_, self.val_losses_

# ---------------------------------------------------------
# 6. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ---------------------------------------------------------
def objective(trial):
    # Suggest hyperparameters
    transformer_dim = trial.suggest_categorical("transformer_dim", [16, 32, 64])
    depth = trial.suggest_int("depth", 3, 20)
    # Only allow heads that evenly divide transformer_dim
    valid_heads = [h for h in [1, 2, 3, 4] if transformer_dim % h == 0]
    heads = trial.suggest_categorical("heads", valid_heads)
    dim_forward = trial.suggest_int("dim_forward", 32, 128, step=32)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    mlp_hidden_dim1 = trial.suggest_int("mlp_hidden_dim1", 32, 128, step=32)
    mlp_hidden_dim2 = trial.suggest_int("mlp_hidden_dim2", 16, 64, step=16)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    mlp_hidden_dims = [mlp_hidden_dim1, mlp_hidden_dim2]

    # Create a hold-out split from the training set for fast evaluation
    X_cat_tr, X_cat_val, X_num_tr, X_num_val, y_tr, y_val = train_test_split(
        X_categ_train, X_num_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Use a smaller number of epochs for optimization
    clf = TabTransformerClassifierWithVal(
        transformer_dim=transformer_dim,
        depth=depth,
        heads=heads,
        dim_forward=dim_forward,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=30,  # Fewer epochs for quick evaluation
        verbose=False
    )

    clf.fit(X_cat_tr, X_num_tr, y_tr, X_cat_val, X_num_val, y_val)
    val_accuracy = clf.score(X_cat_val, X_num_val, y_val)
    return val_accuracy

log_message("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)
best_params = study.best_params
log_message(f"Best hyperparameters found: {best_params}")

# ---------------------------------------------------------
# 6.1 FINAL TRAINING WITH BEST HYPERPARAMETERS (100 EPOCHS)
# ---------------------------------------------------------
final_clf = TabTransformerClassifierWithVal(
    transformer_dim=best_params["transformer_dim"],
    depth=best_params["depth"],
    heads=best_params["heads"],
    dim_forward=best_params["dim_forward"],
    dropout=best_params["dropout"],
    mlp_hidden_dims=[best_params["mlp_hidden_dim1"], best_params["mlp_hidden_dim2"]],
    learning_rate=best_params["learning_rate"],
    batch_size=best_params["batch_size"],
    epochs=100,  # Extended training epochs
    verbose=True
)

log_message("Fitting final TabTransformer model with optimized hyperparameters (100 epochs)...")
final_clf.fit(X_categ_train, X_num_train, y_train)

# ---------------------------------------------------------
# 6.2 5-FOLD CROSS-VALIDATION FOR UNCERTAINTY ESTIMATES (100 EPOCHS)
# ---------------------------------------------------------
log_message("Performing 5-fold CV with optimized hyperparameters for uncertainty estimates...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_train_losses = []
fold_val_losses = []
fold_val_accuracies = []

fold_idx = 1
for train_index, val_index in kf.split(X_categ_train, y_train):
    X_cat_tr_fold, X_cat_val_fold = X_categ_train[train_index], X_categ_train[val_index]
    X_num_tr_fold, X_num_val_fold = X_num_train[train_index], X_num_train[val_index]
    y_tr_fold, y_val_fold = y_train[train_index], y_train[val_index]

    clf_fold = TabTransformerClassifierWithVal(
        transformer_dim=best_params["transformer_dim"],
        depth=best_params["depth"],
        heads=best_params["heads"],
        dim_forward=best_params["dim_forward"],
        dropout=best_params["dropout"],
        mlp_hidden_dims=[best_params["mlp_hidden_dim1"], best_params["mlp_hidden_dim2"]],
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        epochs=100,
        verbose=False
    )

    clf_fold.fit(X_cat_tr_fold, X_num_tr_fold, y_tr_fold,
                 X_cat_val_fold, X_num_val_fold, y_val_fold)

    tr_losses, val_losses = clf_fold.get_train_val_losses()
    fold_train_losses.append(tr_losses)
    fold_val_losses.append(val_losses)

    y_val_pred = clf_fold.predict(X_cat_val_fold, X_num_val_fold)
    val_acc = accuracy_score(y_val_fold, y_val_pred)
    fold_val_accuracies.append(val_acc)
    log_message(f"Fold {fold_idx} complete. Validation Accuracy: {val_acc:.4f}")
    fold_idx += 1

mean_cv_val_acc = np.mean(fold_val_accuracies)
std_cv_val_acc  = np.std(fold_val_accuracies)
log_message(f"5-Fold CV Validation Accuracy: Mean = {mean_cv_val_acc:.4f}, Std = {std_cv_val_acc:.4f}")

# ---------------------------------------------------------
# 6.3 PLOT THE TRAIN & VAL LOSSES (ALL FOLDS) AND AGGREGATED LOSS
# ---------------------------------------------------------
epochs_range = np.arange(1, final_clf.epochs + 1)

fold_train_losses_arr = np.array(fold_train_losses)
fold_val_losses_arr   = np.array(fold_val_losses)

# Plot all folds individually in one figure
plt.figure(figsize=(10, 6))
for i in range(len(fold_train_losses)):
    plt.plot(epochs_range, fold_train_losses[i],
             label=f"Train Loss (Fold {i+1})", alpha=0.6)
    plt.plot(epochs_range, fold_val_losses[i],
             label=f"Val Loss (Fold {i+1})", alpha=0.6, linestyle='--')

plt.title("Train & Validation Loss Curves (All Folds)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("artifacts/tabtransformer_cv_all_folds_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved per-fold train/val loss plot to 'artifacts/tabtransformer_cv_all_folds_loss.png'.")

# Plot mean ± std for aggregated loss
mean_train = fold_train_losses_arr.mean(axis=0)
std_train  = fold_train_losses_arr.std(axis=0)
mean_val   = fold_val_losses_arr.mean(axis=0)
std_val    = fold_val_losses_arr.std(axis=0)

plt.figure(figsize=(8, 5))
plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train,
                 alpha=0.2, label="Train ±1 Std")
plt.plot(epochs_range, mean_train, label="Mean Train Loss")
plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val,
                 alpha=0.2, label="Val ±1 Std")
plt.plot(epochs_range, mean_val, label="Mean Val Loss")
plt.title("Mean ± 1 Std Train/Val Loss (5-fold CV, TabTransformer)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/tabtransformer_cv_mean_confidence_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved mean ± std train/val loss plot to 'artifacts/tabtransformer_cv_mean_confidence_loss.png'.")

# ---------------------------------------------------------
# 7. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ---------------------------------------------------------
log_message("Evaluating final model on the test set...")
y_test_pred = final_clf.predict(X_categ_test, X_num_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
log_message(f"Final Model Accuracy on Test Set: {test_accuracy:.4f}")

class_report = classification_report(y_test, y_test_pred, target_names=main_label_encoder.classes_)
log_message("\nClassification Report:")
log_message(class_report)

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=main_label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (TabTransformer)")
plt.tight_layout()
plt.savefig("artifacts/confusion_matrix_tabtransformer.png", bbox_inches='tight')
plt.close()
log_message("Saved confusion matrix to 'artifacts/confusion_matrix_tabtransformer.png'.")

# ---------------------------------------------------------
# 8. SAVE RESULTS & LOGS FOR REPRODUCIBILITY
# ---------------------------------------------------------
metrics_dict = {
    "test_accuracy": float(test_accuracy),
    "classification_report": class_report,
    "cv_val_accuracies": [float(acc) for acc in fold_val_accuracies],
    "cv_val_accuracy_mean": float(mean_cv_val_acc),
    "cv_val_accuracy_std": float(std_cv_val_acc)
}
with open("artifacts/metrics_tabtransformer.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
log_message("Saved metrics to 'artifacts/metrics_tabtransformer.json'.")

# Save confusion matrix
np.save("artifacts/confusion_matrix_tabtransformer.npy", cm)

# Save the final model weights
torch.save(final_clf.model_.state_dict(), "artifacts/tabtransformer_model_state.pth")
log_message("Saved TabTransformer model's state_dict to 'artifacts/tabtransformer_model_state.pth'.")

# Save training logs
with open("artifacts/log_steps_tabtransformer.json", "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log to 'artifacts/log_steps_tabtransformer.json'.")

# ---------------------------------------------------------
# 9. SAVE ARTIFACTS FOR FULL REPLICATION
# ---------------------------------------------------------
import pickle

# Store the scaler
with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Store the per-column label encoders
with open("artifacts/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Store the main label encoder
with open("artifacts/main_label_encoder.pkl", "wb") as f:
    pickle.dump(main_label_encoder, f)

# Store the final list of columns
replication_info = {
    "numeric_features_kept": features_to_keep,
    "final_categorical_cols": categorical_cols,
    "final_numeric_cols": numeric_cols
}
with open("artifacts/selected_features.json", "w") as f:
    json.dump(replication_info, f, indent=2)

# Save data used for plotting (fold losses, etc.)
np.save("artifacts/fold_train_losses.npy", fold_train_losses_arr)
np.save("artifacts/fold_val_losses.npy", fold_val_losses_arr)
np.save("artifacts/epochs_range.npy", epochs_range)

log_message("Saved scaler, encoders, selected features, and plot data for full replication in the 'artifacts' folder.")
log_message("All done!")

