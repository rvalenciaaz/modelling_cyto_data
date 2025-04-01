import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation

# -----------------------------
# For modeling and evaluation using RAPIDS
# -----------------------------
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# For hyperparameter optimization
import optuna
from functools import partial

# For GPU arrays
import cupy as cp

# ============================================
# 0. GLOBALS & REPRODUCIBILITY
# ============================================
RANDOM_SEED = 42
N_TRIALS = 30  # Number of Optuna trials per model
np.random.seed(RANDOM_SEED)

# ============================================
# 1. READING & SUBSAMPLING CSV FILES
# ============================================
print("Reading CSV files and subsampling...")

csv_files = glob.glob("species*.csv")
df_list = []

for file_path in csv_files:
    temp_df = pd.read_csv(file_path)
    # (Optional) Subsample up to 10,000 rows:
    # temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=RANDOM_SEED)

    # Create a label from the filename, e.g. "species1.csv" -> "1"
    label = file_path.split('.')[0].replace("species", "")
    temp_df['Label'] = label
    df_list.append(temp_df)

combined_df = pd.concat(df_list, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

# ============================================
# 2. FILTERING FEATURES BASED ON MAD
# ============================================
print("Filtering numeric features based on MAD...")

numerical_data = combined_df.select_dtypes(include=[np.number])
cv_results = {}

for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val = numerical_data[col].std()
    cv_val = std_val / mean_val if mean_val != 0 else np.nan
    mad_val = median_abs_deviation(numerical_data[col].values, scale='normal')
    cv_results[col] = [col, cv_val, mad_val]

cv_df = pd.DataFrame(cv_results.values(), columns=['Feature', 'CV', 'MAD'])

MAD_THRESHOLD = 5
features_to_keep = cv_df.loc[cv_df["MAD"] >= MAD_THRESHOLD, "Feature"].tolist()

cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()
print(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ============================================
# 3. FEATURE SCALING & TRAIN/TEST SPLIT
# ============================================
print("Splitting into train/test and scaling features...")

X = final_df.drop(columns=["Label"])
y = final_df["Label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled_cpu = scaler.fit_transform(X_train)
X_test_scaled_cpu  = scaler.transform(X_test)

print("Data split complete.")
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ============================================
# 3A. CONVERT ARRAYS TO CUPY (GPU) FOR XGBOOST
# ============================================
X_train_scaled_gpu = cp.asarray(X_train_scaled_cpu)
X_test_scaled_gpu  = cp.asarray(X_test_scaled_cpu)
y_train_gpu        = cp.asarray(y_train)
y_test_gpu         = cp.asarray(y_test)

# ============================================
# 4. OBJECTIVE FUNCTIONS (Optuna)
# ============================================
#
# Notice how each objective's signature matches exactly
# the arguments passed in the loop (partial).
#

def objective_rf(trial, X_cpu, y_cpu):
    """Uses CPU arrays -> cross_val_score from sklearn."""
    n_estimators = trial.suggest_int("n_estimators", 100, 400, step=100)
    max_depth = trial.suggest_int("max_depth", 5, 30, step=5)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X_cpu, y_cpu, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean()

def objective_lr(trial, X_cpu, y_cpu):
    """Uses CPU arrays -> cross_val_score from sklearn."""
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    model = LogisticRegression(
        C=C,
        random_state=RANDOM_SEED,
        max_iter=2000
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X_cpu, y_cpu, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean()

def gpu_cv_xgb(model, X_gpu, y_gpu, n_splits=3):
    """
    Custom CV loop to keep data on GPU for XGBoost.
    Returns a list of fold accuracies.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    # Convert CuPy data to CPU just for the StratifiedKFold indexing:
    X_cpu = cp.asnumpy(X_gpu)
    y_cpu = cp.asnumpy(y_gpu)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_accuracies = []

    for train_idx, val_idx in skf.split(X_cpu, y_cpu):
        # Slice the CuPy arrays using CPU-based indices
        X_train_fold = X_gpu[train_idx]
        y_train_fold = y_gpu[train_idx]
        X_val_fold   = X_gpu[val_idx]
        y_val_fold   = y_gpu[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred_val = model.predict(X_val_fold)

        # Convert predictions to CPU for accuracy_score
        fold_acc = accuracy_score(cp.asnumpy(y_val_fold), cp.asnumpy(y_pred_val))
        fold_accuracies.append(fold_acc)

    return fold_accuracies

def objective_xgb(trial, X_gpu, y_gpu):
    """
    XGBoost on GPU with custom CV loop, so we never fall back to CPU
    and never get the device mismatch warning.
    """
    n_estimators = trial.suggest_int("n_estimators", 100, 300, step=100)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",  # GPU-based
        device="cuda",           # Let XGBoost know to use CUDA
        random_state=RANDOM_SEED,
        eval_metric='mlogloss'
    )

    cv_scores = gpu_cv_xgb(model, X_gpu, y_gpu, n_splits=3)
    return np.mean(cv_scores)

def objective_svm(trial, X_cpu, y_cpu):
    """Uses CPU arrays -> cross_val_score from sklearn."""
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    if kernel == "rbf":
        gamma = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
    else:
        gamma = "auto"

    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=RANDOM_SEED
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X_cpu, y_cpu, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean()

# ============================================
# 5. RUN OPTUNA STUDIES IN A LOOP
# ============================================
#
# NOTICE that for XGBoost, we store objective_xgb plus GPU data
# and pass them to partial(obj_func, X_gpu=..., y_gpu=...),
# so the signature matches exactly.
#
model_objectives = {
    "RandomForest":       (objective_rf,  X_train_scaled_cpu, y_train),
    "LogisticRegression": (objective_lr,  X_train_scaled_cpu, y_train),
    "XGBoost":            (objective_xgb, X_train_scaled_gpu, y_train_gpu),
    "SVM":                (objective_svm, X_train_scaled_cpu, y_train)
}

best_models = {}
studies = {}
best_params_dict = {}

for model_name, (obj_func, X_data, y_data) in model_objectives.items():
    print(f"\n=== {model_name}: Optuna Hyperparameter Tuning ===")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    # Corrected partial usage:
    if model_name == "XGBoost":
        # XGBoost uses GPU function -> pass as X_gpu, y_gpu
        study.optimize(partial(obj_func, X_gpu=X_data, y_gpu=y_data),
                       n_trials=N_TRIALS, show_progress_bar=True)
    else:
        # CPU models -> pass as X_cpu, y_cpu
        study.optimize(partial(obj_func, X_cpu=X_data, y_cpu=y_data),
                       n_trials=N_TRIALS, show_progress_bar=True)

    print(f"{model_name} best trial:", study.best_trial.params)
    studies[model_name] = study
    best_params_dict[model_name] = study.best_trial.params

    # Train final model on the full training set
    if model_name == "RandomForest":
        best_models[model_name] = RandomForestClassifier(
            n_estimators=study.best_trial.params["n_estimators"],
            max_depth=study.best_trial.params["max_depth"],
            random_state=RANDOM_SEED
        )
        best_models[model_name].fit(X_train_scaled_cpu, y_train)

    elif model_name == "LogisticRegression":
        best_models[model_name] = LogisticRegression(
            C=study.best_trial.params["C"],
            random_state=RANDOM_SEED,
            max_iter=2000
        )
        best_models[model_name].fit(X_train_scaled_cpu, y_train)

    elif model_name == "XGBoost":
        best_models[model_name] = XGBClassifier(
            n_estimators=study.best_trial.params["n_estimators"],
            max_depth=study.best_trial.params["max_depth"],
            learning_rate=study.best_trial.params["learning_rate"],
            tree_method="hist",
            device="cuda",
            random_state=RANDOM_SEED,
            eval_metric='mlogloss'
        )
        # Fit using GPU arrays
        best_models[model_name].fit(X_train_scaled_gpu, y_train_gpu)

    elif model_name == "SVM":
        kernel = study.best_trial.params["kernel"]
        gamma  = (study.best_trial.params["gamma"] if kernel == "rbf" else "auto")
        best_models[model_name] = SVC(
            C=study.best_trial.params["C"],
            kernel=kernel,
            gamma=gamma,
            random_state=RANDOM_SEED
        )
        best_models[model_name].fit(X_train_scaled_cpu, y_train)

    # Save each study
    joblib.dump(study, f"study_{model_name.lower()}.pkl")
    study.trials_dataframe().to_csv(f"study_{model_name.lower()}_trials.csv", index=False)

# Consolidate best hyperparameters
params_records = []
for m_name, p_dict in best_params_dict.items():
    row = {"Model": m_name}
    row.update(p_dict)
    params_records.append(row)

pd.DataFrame(params_records).to_csv("all_best_params.csv", index=False)

# ============================================
# 6. FINAL EVALUATION & REPORT
# ============================================

def evaluate_model_classic(model, model_name, X_train_cpu, y_train_cpu, X_test_cpu, y_test_cpu, encoder):
    """
    Evaluate models that can handle CPU arrays via cross_val_score.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train_cpu, y_train_cpu, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"\n--- {model_name} 5-Fold Cross-Validation (CPU) ---")
    print(f"Fold Accuracies: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}, Std Dev: {cv_scores.std():.4f}")

    # Save CV results
    cv_df = pd.DataFrame({"Fold": range(1, 6), "Accuracy": cv_scores})
    cv_df.to_csv(f"cv_results_{model_name.lower()}.csv", index=False)

    # Test evaluation
    y_pred = model.predict(X_test_cpu)
    test_accuracy = accuracy_score(y_test_cpu, y_pred)
    print(f"\n--- {model_name} Test Set Evaluation (CPU) ---")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    report_dict = classification_report(y_test_cpu, y_pred, target_names=encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    print("\nClassification Report:")
    print(report_df)
    report_df.to_csv(f"classification_report_{model_name.lower()}.csv")

    # Confusion matrices
    cm = confusion_matrix(y_test_cpu, y_pred)
    cm_norm = confusion_matrix(y_test_cpu, y_pred, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.savefig(f"confusion_matrix_{model_name.lower()}.png")
    plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, cmap='Blues',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_,
                fmt=".2f")
    plt.title(f"Normalized Confusion Matrix ({model_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"confusion_matrix_{model_name.lower()}_normalized.png")
    plt.close()

    # Save raw matrix data
    pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)\
        .to_csv(f"confusion_matrix_{model_name.lower()}_data.csv")
    pd.DataFrame(cm_norm, index=encoder.classes_, columns=encoder.classes_)\
        .to_csv(f"confusion_matrix_{model_name.lower()}_normalized_data.csv")

    return {
        "Model": model_name,
        "CV_Accuracy_Mean": cv_scores.mean(),
        "CV_Accuracy_STD": cv_scores.std(),
        "Test_Accuracy": test_accuracy
    }

def evaluate_model_xgb_gpu(model, X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu, encoder, model_name="XGBoost"):
    """
    Custom evaluation for XGBoost with GPU data.
    We do 5-fold CV on GPU, then test on GPU.
    """
    from sklearn.metrics import accuracy_score
    # We'll do a 5-fold custom CV on GPU
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    X_train_cpu = cp.asnumpy(X_train_gpu)
    y_train_cpu = cp.asnumpy(y_train_gpu)

    fold_accuracies = []
    for train_idx, val_idx in skf.split(X_train_cpu, y_train_cpu):
        X_tr_fold = X_train_gpu[train_idx]
        y_tr_fold = y_train_gpu[train_idx]
        X_val_fold = X_train_gpu[val_idx]
        y_val_fold = y_train_gpu[val_idx]

        # Fit each fold
        temp_model = XGBClassifier(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            tree_method="hist",
            device="cuda",
            random_state=RANDOM_SEED,
            eval_metric='mlogloss'
        )
        temp_model.fit(X_tr_fold, y_tr_fold)
        y_pred_val = temp_model.predict(X_val_fold)

        acc = accuracy_score(cp.asnumpy(y_val_fold), cp.asnumpy(y_pred_val))
        fold_accuracies.append(acc)

    cv_mean = np.mean(fold_accuracies)
    cv_std  = np.std(fold_accuracies)

    print(f"\n--- {model_name} 5-Fold Cross-Validation (GPU) ---")
    print("Fold Accuracies:", fold_accuracies)
    print(f"Mean CV Accuracy: {cv_mean:.4f}, Std Dev: {cv_std:.4f}")

    # Save fold results
    cv_df = pd.DataFrame({"Fold": range(1, 6), "Accuracy": fold_accuracies})
    cv_df.to_csv(f"cv_results_{model_name.lower()}.csv", index=False)

    # Evaluate on test set (GPU)
    y_pred_test = model.predict(X_test_gpu)
    test_accuracy = accuracy_score(
        cp.asnumpy(y_test_gpu),
        cp.asnumpy(y_pred_test)
    )
    print(f"\n--- {model_name} Test Set Evaluation (GPU) ---")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report on CPU
    y_test_cpu = cp.asnumpy(y_test_gpu)
    y_pred_cpu = cp.asnumpy(y_pred_test)

    report_dict = classification_report(y_test_cpu, y_pred_cpu, target_names=encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    print("\nClassification Report:")
    print(report_df)
    report_df.to_csv(f"classification_report_{model_name.lower()}.csv")

    # Confusion matrices
    cm = confusion_matrix(y_test_cpu, y_pred_cpu)
    cm_norm = confusion_matrix(y_test_cpu, y_pred_cpu, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.savefig(f"confusion_matrix_{model_name.lower()}.png")
    plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, cmap='Blues',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_,
                fmt=".2f")
    plt.title(f"Normalized Confusion Matrix ({model_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"confusion_matrix_{model_name.lower()}_normalized.png")
    plt.close()

    # Save raw data
    pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)\
        .to_csv(f"confusion_matrix_{model_name.lower()}_data.csv")
    pd.DataFrame(cm_norm, index=encoder.classes_, columns=encoder.classes_)\
        .to_csv(f"confusion_matrix_{model_name.lower()}_normalized_data.csv")

    return {
        "Model": model_name,
        "CV_Accuracy_Mean": cv_mean,
        "CV_Accuracy_STD": cv_std,
        "Test_Accuracy": test_accuracy
    }

# ============================================
# 7. FINAL EVALUATION
# ============================================
print("\n=== Final Evaluation with 5-Fold Cross-Validation & Test Sets ===")
results = []

for model_name, model in best_models.items():
    if model_name == "XGBoost":
        # Evaluate on GPU
        metrics = evaluate_model_xgb_gpu(
            model=model,
            X_train_gpu=X_train_scaled_gpu,
            y_train_gpu=y_train_gpu,
            X_test_gpu=X_test_scaled_gpu,
            y_test_gpu=y_test_gpu,
            encoder=label_encoder,
            model_name=model_name
        )
    else:
        # Evaluate on CPU
        metrics = evaluate_model_classic(
            model=model,
            model_name=model_name,
            X_train_cpu=X_train_scaled_cpu,
            y_train_cpu=y_train,
            X_test_cpu=X_test_scaled_cpu,
            y_test_cpu=y_test,
            encoder=label_encoder
        )
    results.append(metrics)

metrics_df = pd.DataFrame(results)
print("\n=== Combined Model Metrics ===")
print(metrics_df)
metrics_df.to_csv("classification_metrics_all_models.csv", index=False)

# ============================================
# 8. SAVE MODELS & ARTIFACTS
# ============================================
print("\nSaving trained models and artifacts for replication...")
for model_name, model in best_models.items():
    joblib.dump(model, f"best_{model_name.lower()}_model.pkl")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

pd.DataFrame(features_to_keep, columns=["Feature"]).to_csv("features_to_keep.csv", index=False)

print("\nAll done! Check the CSV, PKL, and PNG files for detailed results.")
