import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import json
import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science','nature'])
# Fix seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------
# 0. LOGGING UTILITY FOR TIMESTAMPS
# ---------------------------------------------------------
log_steps = []
def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# ---------------------------------------------------------
# 1. READING & SUBSAMPLING CSV FILES
# ---------------------------------------------------------
log_message("Reading CSV files and subsampling...")
csv_files = glob.glob("species*.csv")  # Will get a list of species*.csv filenames
df_list = []
for file_path in csv_files:
    temp_df = pd.read_csv(file_path)
    #subsampling is optional here
    #temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)
    label = file_path.split('.')[0]  # e.g. "species1.csv" -> "species1"
    temp_df['Label'] = label
    df_list.append(temp_df)
combined_df = pd.concat(df_list, ignore_index=True)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)
log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. FILTERING FEATURES BASED ON MAD
# ---------------------------------------------------------
log_message("Filtering numeric features based on MAD...")
numerical_data = combined_df.select_dtypes(include=[np.number])
cv_results = {}
for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val = numerical_data[col].std()
    cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    mad = median_abs_deviation(numerical_data[col].values, scale='normal')
    cv_results[col] = [col, cv, mad]
cv_df = pd.DataFrame(cv_results.values(), columns=['Feature', 'CV', 'MAD'])
MAD_THRESHOLD = 5
features_to_keep = cv_df.loc[cv_df["MAD"] >= MAD_THRESHOLD, "Feature"].tolist()  # filtering by MAD
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()
log_message(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT (RAW DATA; SCALING IS PART OF THE PIPELINE)
# ---------------------------------------------------------
log_message("Splitting data into training and test sets...")
X = final_df.drop(columns=["Label"]).values
y = final_df["Label"].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
log_message(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ---------------------------------------------------------
# 4. DEFINE THE CONFIGURABLE NN MODEL
# ---------------------------------------------------------
class ConfigurableNN(nn.Module):
    """
    A feedforward neural network with a variable number of hidden layers.
    Each hidden layer has 'hidden_size' units followed by ReLU.
    """
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=1):
        super(ConfigurableNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 5. DEFINE A SINGLE PYTORCH CLASSIFIER THAT TRACKS VALIDATION LOSS
# ---------------------------------------------------------
class PyTorchNNClassifierWithVal(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=32, learning_rate=1e-3, batch_size=32,
                 epochs=10, num_layers=1, verbose=True):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_layers = num_layers
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim_ = None
        self.output_dim_ = None
        self.model_ = None
        self.classes_ = None
        self.train_losses_ = []
        self.val_losses_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.input_dim_ = X.shape[1]
        self.output_dim_ = len(np.unique(y))
        self.classes_ = np.unique(y)
        self.model_ = ConfigurableNN(
            input_dim=self.input_dim_,
            hidden_size=self.hidden_size,
            output_dim=self.output_dim_,
            num_layers=self.num_layers
        ).to(self.device)
        param_count = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        print(f"Model with {self.num_layers} hidden layers and hidden_size {self.hidden_size} "
              f"has {param_count} trainable parameters.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        X_torch = torch.from_numpy(X).float().to(self.device)
        y_torch = torch.from_numpy(y).long().to(self.device)
        dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.train_losses_ = []
        self.val_losses_ = []
        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.train_losses_.append(avg_epoch_loss)
            if X_val is not None and y_val is not None:
                self.model_.eval()
                X_val_torch = torch.from_numpy(X_val).float().to(self.device)
                y_val_torch = torch.from_numpy(y_val).long().to(self.device)
                with torch.no_grad():
                    val_outputs = self.model_(X_val_torch)
                    val_loss = criterion(val_outputs, y_val_torch).item()
            else:
                val_loss = np.nan
            self.val_losses_.append(val_loss)
            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_epoch_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
        return self

    def predict(self, X):
        self.model_.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model_(X_torch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        self.model_.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model_(X_torch)
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

    def get_train_val_losses(self):
        return self.train_losses_, self.val_losses_

# ---------------------------------------------------------
# 6. SET UP PIPELINE & GRID SEARCH USING THE SAME NN (WITH VAL)
# ---------------------------------------------------------
log_message("=== Building Pipeline & Grid Search (using NN with Val) ===")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nn', PyTorchNNClassifierWithVal(verbose=True))
])
param_grid = {
    'nn__hidden_size':   [16, 32],
    'nn__num_layers':    [1, 2, 3],
    'nn__learning_rate': [1e-3, 1e-4],
    'nn__batch_size':    [16, 32],
    'nn__epochs':        [30]  # fewer epochs during grid search
}
grid_nn = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=2,
    return_train_score=True
)
log_message("Starting GridSearchCV for the pipeline...")
grid_nn.fit(X_train, y_train)
log_message("Grid search complete.")
best_params = grid_nn.best_params_
log_message(f"Best NN Pipeline Parameters: {best_params}")

# ---------------------------------------------------------
# 7. FINAL REFIT ON TRAINING SET WITH MORE EPOCHS
# ---------------------------------------------------------
log_message("Re-fitting best pipeline on entire training set with 100 epochs...")
best_pipeline = grid_nn.best_estimator_
best_pipeline.set_params(nn__epochs=100)
best_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------
# 7.1. 5-FOLD CROSS-VALIDATION WITH BEST HYPERPARAMS
# ---------------------------------------------------------
log_message("Performing 5-fold CV with the best hyperparameters for uncertainty estimates...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_train_losses = []
fold_val_losses = []
fold_val_accuracies = []  # store validation accuracy per fold
best_nn_params = best_pipeline.named_steps['nn'].get_params()
fold_idx = 1
for train_index, val_index in kf.split(X_train, y_train):
    X_tr_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_tr_fold, y_val_fold = y_train[train_index], y_train[val_index]
    scaler_fold = StandardScaler()
    X_tr_fold_scaled = scaler_fold.fit_transform(X_tr_fold)
    X_val_fold_scaled = scaler_fold.transform(X_val_fold)
    clf_fold = PyTorchNNClassifierWithVal(**best_nn_params)
    clf_fold.fit(X_tr_fold_scaled, y_tr_fold, X_val_fold_scaled, y_val_fold)
    tr_losses, val_losses = clf_fold.get_train_val_losses()
    fold_train_losses.append(tr_losses)
    fold_val_losses.append(val_losses)
    y_val_pred = clf_fold.predict(X_val_fold_scaled)
    fold_val_acc = accuracy_score(y_val_fold, y_val_pred)
    fold_val_accuracies.append(fold_val_acc)
    log_message(f"Fold {fold_idx} complete. Validation Accuracy: {fold_val_acc:.4f}")
    fold_idx += 1
mean_cv_val_acc = np.mean(fold_val_accuracies)
std_cv_val_acc = np.std(fold_val_accuracies)
log_message(f"5-Fold CV Validation Accuracy: Mean = {mean_cv_val_acc:.4f}, Std = {std_cv_val_acc:.4f}")

# Define the epoch range (all folds use the same number of epochs)
epochs_range = np.arange(1, best_nn_params["epochs"] + 1)

# -------------------------------
# SAVE DATA FOR PLOT REPRODUCTION
# -------------------------------
np.savez("cv_plot_data.npz",
         epochs_range=epochs_range,
         fold_train_losses=np.array(fold_train_losses),
         fold_val_losses=np.array(fold_val_losses),
         mean_train=np.array(fold_train_losses).mean(axis=0),
         std_train=np.array(fold_train_losses).std(axis=0),
         mean_val=np.array(fold_val_losses).mean(axis=0),
         std_val=np.array(fold_val_losses).std(axis=0),
         fold_val_accuracies=np.array(fold_val_accuracies))
log_message("Saved CV plot data to 'cv_plot_data.npz'.")

plt.figure(figsize=(10, 6))
for i in range(len(fold_train_losses)):
    plt.plot(epochs_range, fold_train_losses[i], label=f"Train Loss (Fold {i+1})", alpha=0.6)
    plt.plot(epochs_range, fold_val_losses[i], label=f"Val Loss (Fold {i+1})", alpha=0.6, linestyle='--')
plt.title("Train & Validation Loss Curves (All Folds)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("cv_all_folds_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved plot of train/val loss per fold to 'cv_all_folds_loss.png'.")

train_losses_arr = np.array(fold_train_losses)
val_losses_arr   = np.array(fold_val_losses)
mean_train = train_losses_arr.mean(axis=0)
std_train  = train_losses_arr.std(axis=0)
mean_val   = val_losses_arr.mean(axis=0)
std_val    = val_losses_arr.std(axis=0)
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, mean_train, label="Mean Train Loss", color='blue')
plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train,
                 alpha=0.2, color='blue')
plt.plot(epochs_range, mean_val, label="Mean Val Loss", color='orange')
plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val,
                 alpha=0.2, color='orange')
plt.title("Mean ± 1 Std Train/Val Loss (5-fold CV)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("cv_mean_confidence_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved mean ± std train/val loss plot to 'cv_mean_confidence_loss.png'.")

# ---------------------------------------------------------
# 8. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ---------------------------------------------------------
log_message("Evaluating final model on test set...")
y_pred = best_pipeline.predict(X_test)  # Pipeline applies scaling automatically
test_accuracy = accuracy_score(y_test, y_pred)
log_message(f"Final Model Accuracy on Test Set: {test_accuracy:.4f}")
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
log_message("\nClassification Report:")
log_message(class_report)

# Save confusion matrix data to reproduce the confusion matrix plot later
cm = confusion_matrix(y_test, y_pred)
np.save("confusion_matrix.npy", cm)
log_message("Saved confusion matrix data to 'confusion_matrix.npy'.")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Final Model)")
plt.tight_layout()
plt.savefig("confusion_matrix_nn.png", bbox_inches='tight')
plt.close()
log_message("Saved confusion matrix to 'confusion_matrix_nn.png'.")

metrics_dict = {
    "test_accuracy": float(test_accuracy),
    "classification_report": class_report,
    "cv_val_accuracies": fold_val_accuracies,
    "cv_val_accuracy_mean": float(mean_cv_val_acc),
    "cv_val_accuracy_std": float(std_cv_val_acc)
}
with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
log_message("Saved metrics to 'metrics.json'.")

torch.save(best_pipeline.named_steps['nn'].model_.state_dict(), "best_model_state.pth")
log_message("Saved final model's state_dict to 'best_model_state.pth'.")
best_estimator_data = {
    "state_dict": best_pipeline.named_steps['nn'].model_.state_dict(),
    "params": best_pipeline.named_steps['nn'].get_params()
}
torch.save(best_estimator_data, "best_estimator.pth")
log_message("Saved final estimator (weights + params) to 'best_estimator.pth'.")

with open("log_steps.json", "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log with timestamps to 'log_steps.json'.")
log_message("All done!")
