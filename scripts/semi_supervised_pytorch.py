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
from sklearn.semi_supervised import SelfTrainingClassifier  # for semi-supervised learning

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
    # subsampling is optional here
    # temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)
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
# 3.1. SIMULATE A SEMI-SUPERVISED SETTING
# ---------------------------------------------------------
# In a semi-supervised setting, only a fraction of the training data is labeled.
# Here we randomly mask 50% of the training labels (set them to -1).
log_message("Masking 50% of training labels to simulate unlabeled data...")
y_train_ss = y_train.copy()
mask = np.random.rand(len(y_train_ss)) < 0.5
y_train_ss[mask] = -1  # -1 indicates an unlabeled sample in SelfTrainingClassifier

# ---------------------------------------------------------
# 4. DEFINE THE CONFIGURABLE NN MODEL (UNCHANGED)
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
# 5. DEFINE THE PYTORCH CLASSIFIER (UNCHANGED)
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
        self.output_dim_ = len(np.unique(y[y != -1]))  # exclude unlabeled (-1)
        self.classes_ = np.unique(y[y != -1])
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
        # Only use samples with labels (i.e. y != -1)
        labeled_idx = np.where(y != -1)[0]
        X_labeled = X[labeled_idx]
        y_labeled = y[labeled_idx]
        X_torch = torch.from_numpy(X_labeled).float().to(self.device)
        y_torch = torch.from_numpy(y_labeled).long().to(self.device)
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
# 6. SET UP PIPELINE & SELF-TRAINING WRAPPER
# ---------------------------------------------------------
log_message("=== Building Self-Training Pipeline for Semi-Supervised Learning ===")
base_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nn', PyTorchNNClassifierWithVal(verbose=True))
])
# Wrap the base pipeline in a SelfTrainingClassifier.
# The classifier will iteratively assign pseudo-labels to the unlabeled (-1) samples.
self_training_clf = SelfTrainingClassifier(
    base_estimator=base_pipeline,
    threshold=0.9,  # only assign pseudo-labels for predictions with >=90% confidence
    criterion='threshold',
    verbose=True
)

# We can use GridSearchCV over the parameters of the base estimator.
param_grid = {
    'base_estimator__nn__hidden_size':   [16, 32],
    'base_estimator__nn__num_layers':    [1, 2, 3],
    'base_estimator__nn__learning_rate': [1e-3, 1e-4],
    'base_estimator__nn__batch_size':    [16, 32],
    'base_estimator__nn__epochs':        [30]  # fewer epochs during grid search
}
grid_ss = GridSearchCV(
    estimator=self_training_clf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,  # safer with custom PyTorch models
    verbose=2,
    return_train_score=True
)
log_message("Starting GridSearchCV for the self-training pipeline...")
grid_ss.fit(X_train, y_train_ss)
log_message("Grid search complete.")
best_params = grid_ss.best_params_
log_message(f"Best Self-Training Pipeline Parameters: {best_params}")

# ---------------------------------------------------------
# 7. FINAL REFIT ON TRAINING SET WITH MORE EPOCHS
# ---------------------------------------------------------
log_message("Re-fitting best self-training pipeline on entire training set with 100 epochs...")
best_self_training_pipeline = grid_ss.best_estimator_
# Increase epochs in the base estimator before final refit.
best_self_training_pipeline.base_estimator.set_params(nn__epochs=100)
best_self_training_pipeline.fit(X_train, y_train_ss)

# ---------------------------------------------------------
# 7.1. 5-FOLD CROSS-VALIDATION WITH SEMI-SUPERVISED LEARNING
# ---------------------------------------------------------
log_message("Performing 5-fold CV with semi-supervised learning...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_val_accuracies = []
fold_idx = 1
for train_index, val_index in kf.split(X_train, y_train_ss):
    X_tr_fold, X_val_fold = X_train[train_index], X_train[val_index]
    # For the training fold, re-mask 50% as unlabeled
    y_tr_fold = y_train[train_index].copy()
    mask_fold = np.random.rand(len(y_tr_fold)) < 0.5
    y_tr_fold[mask_fold] = -1
    # Use the same best parameters from grid search for each fold
    fold_clf = SelfTrainingClassifier(
        base_estimator=base_pipeline,
        threshold=0.9,
        criterion='threshold',
        verbose=False
    )
    fold_clf.set_params(**best_params)
    fold_clf.fit(X_tr_fold, y_tr_fold)
    # For evaluation, use the true labels from y_train (the original fully labeled values)
    y_val_true = y_train[val_index]
    y_val_pred = fold_clf.predict(X_val_fold)
    fold_acc = accuracy_score(y_val_true, y_val_pred)
    fold_val_accuracies.append(fold_acc)
    log_message(f"Fold {fold_idx} complete. Validation Accuracy: {fold_acc:.4f}")
    fold_idx += 1
mean_cv_val_acc = np.mean(fold_val_accuracies)
std_cv_val_acc = np.std(fold_val_accuracies)
log_message(f"5-Fold CV Validation Accuracy (semi-supervised): Mean = {mean_cv_val_acc:.4f}, Std = {std_cv_val_acc:.4f}")

# ---------------------------------------------------------
# 8. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ---------------------------------------------------------
log_message("Evaluating final semi-supervised model on test set...")
# The self-training classifier uses the base pipeline internally.
y_pred = best_self_training_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
log_message(f"Final Model Accuracy on Test Set: {test_accuracy:.4f}")
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
log_message("\nClassification Report:")
log_message(class_report)

# Save confusion matrix data for later reproduction of the plot
cm = confusion_matrix(y_test, y_pred)
np.save("confusion_matrix.npy", cm)
log_message("Saved confusion matrix data to 'confusion_matrix.npy'.")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Semi-Supervised Model)")
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

# Save the final model weights from the base estimator.
# Note: To reload the model later, you would need to extract the base_estimator and load its state.
torch.save(best_self_training_pipeline.base_estimator.named_steps['nn'].model_.state_dict(), "best_model_state.pth")
log_message("Saved final model's state_dict to 'best_model_state.pth'.")
best_estimator_data = {
    "state_dict": best_self_training_pipeline.base_estimator.named_steps['nn'].model_.state_dict(),
    "params": best_self_training_pipeline.base_estimator.named_steps['nn'].get_params()
}
torch.save(best_estimator_data, "best_estimator.pth")
log_message("Saved final estimator (weights + params) to 'best_estimator.pth'.")

with open("log_steps.json", "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log with timestamps to 'log_steps.json'.")
log_message("All done!")
