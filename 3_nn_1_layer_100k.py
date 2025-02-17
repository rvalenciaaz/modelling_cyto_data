import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation  # deviation unsensitive to outliers

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# For modeling and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.base import BaseEstimator, ClassifierMixin

# SHAP for feature importance
import shap

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Fix random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------
# 1. READING & SUBSAMPLING CSV FILES
# ---------------------------------------------------------
print("Reading CSV files and subsampling...")

csv_files = glob.glob("species*.csv")  # obtaining sample file
df_list = []

for file_path in csv_files:
    temp_df = pd.read_csv(file_path)
    # Seed is important here; sample 10k out of 100k (optional)
    #temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)
    label = file_path.split('.')[0]  # e.g. "species1.csv" -> "species1"
    temp_df['Label'] = label
    df_list.append(temp_df)

combined_df = pd.concat(df_list, ignore_index=True)
# Clean up the 'Label' column (remove "species" prefix)
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)

print(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. FILTERING FEATURES BASED ON MAD
# ---------------------------------------------------------
print("Filtering numeric features based on MAD...")

numerical_data = combined_df.select_dtypes(include=[np.number])  # only numeric columns
cv_results = {}
for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val = numerical_data[col].std()
    cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan  # CV in %
    mad = median_abs_deviation(numerical_data[col].values, scale='normal')
    cv_results[col] = [col, cv, mad]

cv_df = pd.DataFrame(cv_results.values(), columns=['Feature', 'CV', 'MAD'])

MAD_THRESHOLD = 5  # adjustable threshold
features_to_keep = cv_df.loc[cv_df["MAD"] >= MAD_THRESHOLD, "Feature"].tolist()

# Final DataFrame
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()

print(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. FEATURE SCALING & TRAIN/TEST SPLIT
# ---------------------------------------------------------
print("Splitting into train/test and scaling features...")

X = final_df.drop(columns=["Label"])
y = final_df["Label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Use fixed random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data split complete.")
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ---------------------------------------------------------
# 4. BUILDING A PYTORCH MODEL WRAPPER
# ---------------------------------------------------------

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network with 1 hidden layer.
    """
    def __init__(self, input_dim, hidden_size, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PyTorchNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn style wrapper for a simple PyTorch feedforward network.
    """
    def __init__(self, hidden_size=32, learning_rate=1e-3, batch_size=32, epochs=10, verbose=False):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        # Will be set after fitting:
        self.input_dim_ = None
        self.output_dim_ = None
        self.model_ = None
        self.classes_ = None
        self.train_losses_ = []  # For plotting epoch loss

        # Set device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Using device: {self.device}")

    def fit(self, X, y):
        """
        X and y are numpy arrays.
        """
        self.input_dim_ = X.shape[1]
        self.output_dim_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        # Construct and move the model to device
        self.model_ = SimpleNN(self.input_dim_, self.hidden_size, self.output_dim_)
        self.model_.to(self.device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        # Convert X, y to torch tensors
        X_torch = torch.from_numpy(X).float().to(self.device)
        y_torch = torch.from_numpy(y).long().to(self.device)

        # DataLoader
        dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.train_losses_ = []  # reset for each new fit
        self.model_.train()

        for epoch in range(self.epochs):
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

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_epoch_loss:.4f}")

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
        """
        Required by GridSearchCV. We'll use accuracy_score internally.
        """
        preds = self.predict(X)
        return accuracy_score(y, preds)


# ---------------------------------------------------------
# 5. GRID SEARCH OVER NEURAL NETWORK HYPERPARAMETERS
# ---------------------------------------------------------
print("\n=== Neural Network: Grid Search ===")

param_grid_nn = {
    "hidden_size": [32, 64],
    "learning_rate": [1e-3, 1e-4],
    "batch_size": [32, 64],
    "epochs": [10, 20]  # fewer epochs in the grid search for speed
}

nn_estimator = PyTorchNNClassifier(verbose=False)

grid_nn = GridSearchCV(
    estimator=nn_estimator,
    param_grid=param_grid_nn,
    cv=3,
    scoring='accuracy',
    n_jobs=1,  # safer for PyTorch
    verbose=2
)

print("Starting GridSearchCV for PyTorch NN...")
grid_nn.fit(X_train_scaled, y_train)

print("Neural Network Grid Search complete.")
print(f"Best NN Parameters: {grid_nn.best_params_}")

# ---------------------------------------------------------
# 6. FINAL RE-FIT WITH 100 EPOCHS + LOSS PLOT
# ---------------------------------------------------------
print("Re-fitting best NN model on entire training set with 100 epochs...")

best_nn = grid_nn.best_estimator_
best_nn.epochs = 100  # Increase to 100 for final training
best_nn.verbose = True
best_nn.fit(X_train_scaled, y_train)

# Plot training loss vs. epoch
plt.figure()
plt.plot(range(1, best_nn.epochs + 1), best_nn.train_losses_, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Epoch (Final Model)")
plt.savefig("training_loss.png", bbox_inches='tight')
plt.close()

print("Final model training complete (100 epochs). Loss plot saved to 'training_loss.png'.")

# ---------------------------------------------------------
# 7. MULTI-CLASS EVALUATION (FINAL MODEL)
# ---------------------------------------------------------
print("\nEvaluating final model on test set...")

y_pred = best_nn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Final Model Accuracy: {accuracy:.4f}")

# Generate classification report
class_report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)
print("\nClassification Report:")
print(class_report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Final Model)")
plt.savefig("confusion_matrix_nn.png", bbox_inches='tight')
plt.close()

print("Classification metrics and confusion matrix saved.")

# ---------------------------------------------------------
# 7a. SAVE METRICS TO A TEXT FILE
# ---------------------------------------------------------
with open("classification_metrics.txt", "w") as f:
    f.write(f"Final Model Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("Metrics have been saved to 'classification_metrics.txt'.")

# ---------------------------------------------------------
# 7b. SAVE MODEL WEIGHTS FOR LATER LOADING
# ---------------------------------------------------------
# Save only the model's parameter state dict (recommended practice).
torch.save(best_nn.model_.state_dict(), "model_weights.pth")
print("Model weights have been saved to 'model_weights.pth'.")

'''
# ---------------------------------------------------------
# 8. SHAP EXPLANATIONS (OPTIONAL SECTION FOR MULTI-CLASS)
# ---------------------------------------------------------
print("\nCalculating SHAP values for feature importance...")

best_nn.model_.eval()

# For large data, sample a subset
shap_X_sample = X_train_scaled[:200]
shap_X_tensor = torch.from_numpy(shap_X_sample).float().to(best_nn.device)

explainer = shap.GradientExplainer(
    (best_nn.model_, best_nn.model_.fc1),
    shap_X_tensor
)

shap_values = explainer.shap_values(shap_X_tensor)

# shap_values may be a list (one array per class).
if isinstance(shap_values, list):
    shap_values = np.array(shap_values)  # shape: (n_classes, n_samples, n_features)

abs_shap = np.abs(shap_values)
mean_shap = np.mean(abs_shap, axis=(0,1))  # average across classes & samples
feature_names = X.columns

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "MeanAbsSHAP": mean_shap
}).sort_values("MeanAbsSHAP", ascending=False)

print("\nTop 10 features by mean absolute SHAP value:")
print(shap_df.head(10))

shap_df.to_csv("shap_feature_importances_nn.csv", index=False)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, shap_X_sample, feature_names=feature_names)
plt.savefig("shap_summary_plot.png", bbox_inches='tight')
plt.close()

# SHAP bar plot
plt.figure()
sns.barplot(data=shap_df.head(20), x="MeanAbsSHAP", y="Feature", color='skyblue')
plt.title("Top 20 Feature Importances by SHAP (NN)")
plt.tight_layout()
plt.savefig("shap_barplot_nn.png")
plt.close()
'''

print("\nAll done! Check the CSV and PNG files for results and plots, "
      "plus 'classification_metrics.txt' for saved metrics, "
      "and 'model_weights.pth' for model weights.")
