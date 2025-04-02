# main.py
import glob
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn & RAPIDS / XGBoost
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC
from xgboost import XGBClassifier

# Stats
from scipy.stats import median_abs_deviation

# GPU / CuPy
import cupy as cp

# Hyperparameter optimization
import optuna
from functools import partial

# Import Optuna objective functions from train_utils.py
from train_utils import objective_rf, objective_lr, objective_xgb, objective_svm

# ============================================
# 0. GLOBALS & REPRODUCIBILITY
# ============================================
RANDOM_SEED = 42
N_TRIALS = 30  # Number of Optuna trials per model
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # ============================================
    # 1. READING & SUBSAMPLING CSV FILES
    # ============================================
    print("Reading CSV files and subsampling...")

    csv_files = glob.glob("species*.csv")
    df_list = []

    for file_path in csv_files:
        temp_df = pd.read_csv(file_path)
        # Optionally subsample each CSV up to 10,000 rows:
        # temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=RANDOM_SEED)

        # Create a label from filename, e.g. "species1.csv" -> "1"
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
    # After train/test split (and any preprocessing)
    np.savez("data_for_calibration.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    scaler = StandardScaler()
    X_train_scaled_cpu = scaler.fit_transform(X_train)
    X_test_scaled_cpu  = scaler.transform(X_test)

    print("Data split complete.")
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Convert arrays to CuPy (GPU) for XGBoost
    X_train_scaled_gpu = cp.asarray(X_train_scaled_cpu)
    X_test_scaled_gpu  = cp.asarray(X_test_scaled_cpu)
    y_train_gpu        = cp.asarray(y_train)
    y_test_gpu         = cp.asarray(y_test)

    # ============================================
    # 5. RUN OPTUNA STUDIES & TRAIN FINAL MODELS
    # ============================================
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

        if model_name == "XGBoost":
            study.optimize(partial(obj_func, X_gpu=X_data, y_gpu=y_data),
                           n_trials=N_TRIALS, show_progress_bar=True)
        else:
            study.optimize(partial(obj_func, X_cpu=X_data, y_cpu=y_data),
                           n_trials=N_TRIALS, show_progress_bar=True)

        print(f"{model_name} best trial:", study.best_trial.params)
        studies[model_name] = study
        best_params_dict[model_name] = study.best_trial.params

        # Train final model using best params on the FULL training set
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
            best_models[model_name].fit(X_train_scaled_gpu, y_train_gpu)

        elif model_name == "SVM":
            kernel = study.best_trial.params["kernel"]
            gamma  = study.best_trial.params["gamma"] if kernel == "rbf" else "auto"
            best_models[model_name] = SVC(
                C=study.best_trial.params["C"],
                kernel=kernel,
                gamma=gamma,
                random_state=RANDOM_SEED
            )
            best_models[model_name].fit(X_train_scaled_cpu, y_train)

        # Save each study
        joblib.dump(study, os.path.join(OUTPUT_DIR, f"study_{model_name.lower()}.pkl"))
        study.trials_dataframe().to_csv(
            os.path.join(OUTPUT_DIR, f"study_{model_name.lower()}_trials.csv"),
            index=False
        )

    # Save consolidated best hyperparameters
    params_records = []
    for m_name, p_dict in best_params_dict.items():
        row = {"Model": m_name}
        row.update(p_dict)
        params_records.append(row)
    pd.DataFrame(params_records).to_csv(
        os.path.join(OUTPUT_DIR, "all_best_params.csv"), index=False
    )

    # ============================================
    # 6. FINAL EVALUATION & REPORT
    # ============================================
    def evaluate_model_classic(model, model_name, X_train_cpu, y_train_cpu, 
                               X_test_cpu, y_test_cpu, encoder):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X_train_cpu, y_train_cpu, cv=cv,
                                    scoring='accuracy', n_jobs=-1)

        print(f"\n--- {model_name} 5-Fold Cross-Validation (CPU) ---")
        print(f"Fold Accuracies: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f}, Std Dev: {cv_scores.std():.4f}")

        cv_df = pd.DataFrame({"Fold": range(1, 6), "Accuracy": cv_scores})
        cv_df.to_csv(
            os.path.join(OUTPUT_DIR, f"cv_results_{model_name.lower()}.csv"),
            index=False
        )

        y_pred = model.predict(X_test_cpu)
        test_accuracy = accuracy_score(y_test_cpu, y_pred)
        print(f"\n--- {model_name} Test Set Evaluation (CPU) ---")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        report_dict = classification_report(
            y_test_cpu, y_pred,
            target_names=encoder.classes_,
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        print("\nClassification Report:")
        print(report_df)
        report_df.to_csv(
            os.path.join(OUTPUT_DIR, f"classification_report_{model_name.lower()}.csv")
        )

        cm = confusion_matrix(y_test_cpu, y_pred)
        cm_norm = confusion_matrix(y_test_cpu, y_pred, normalize='true')

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix ({model_name})")
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}.png"))
        plt.close()

        plt.figure(figsize=(8,6))
        sns.heatmap(cm_norm, annot=True, cmap='Blues',
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_,
                    fmt=".2f")
        plt.title(f"Normalized Confusion Matrix ({model_name})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}_normalized.png"))
        plt.close()

        pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(
            os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}_data.csv")
        )
        pd.DataFrame(cm_norm, index=encoder.classes_, columns=encoder.classes_).to_csv(
            os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}_normalized_data.csv")
        )

        return {
            "Model": model_name,
            "CV_Accuracy_Mean": cv_scores.mean(),
            "CV_Accuracy_STD": cv_scores.std(),
            "Test_Accuracy": test_accuracy
        }

    def evaluate_model_xgb_gpu(model, X_train_gpu, y_train_gpu, 
                               X_test_gpu, y_test_gpu, encoder, 
                               model_name="XGBoost"):
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        X_train_cpu = cp.asnumpy(X_train_gpu)
        y_train_cpu = cp.asnumpy(y_train_gpu)

        fold_accuracies = []
        for train_idx, val_idx in skf.split(X_train_cpu, y_train_cpu):
            X_tr_fold = X_train_gpu[train_idx]
            y_tr_fold = y_train_gpu[train_idx]
            X_val_fold = X_train_gpu[val_idx]
            y_val_fold = y_train_gpu[val_idx]

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

        cv_df = pd.DataFrame({"Fold": range(1, 6), "Accuracy": fold_accuracies})
        cv_df.to_csv(
            os.path.join(OUTPUT_DIR, f"cv_results_{model_name.lower()}.csv"),
            index=False
        )

        y_pred_test = model.predict(X_test_gpu)
        test_accuracy = accuracy_score(
            cp.asnumpy(y_test_gpu),
            cp.asnumpy(y_pred_test)
        )
        print(f"\n--- {model_name} Test Set Evaluation (GPU) ---")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        y_test_cpu = cp.asnumpy(y_test_gpu)
        y_pred_cpu = cp.asnumpy(y_pred_test)

        report_dict = classification_report(
            y_test_cpu, y_pred_cpu, target_names=encoder.classes_, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        print("\nClassification Report:")
        print(report_df)
        report_df.to_csv(
            os.path.join(OUTPUT_DIR, f"classification_report_{model_name.lower()}.csv")
        )

        cm = confusion_matrix(y_test_cpu, y_pred_cpu)
        cm_norm = confusion_matrix(y_test_cpu, y_pred_cpu, normalize='true')

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix ({model_name})")
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}.png"))
        plt.close()

        plt.figure(figsize=(8,6))
        sns.heatmap(cm_norm, annot=True, cmap='Blues',
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_,
                    fmt=".2f")
        plt.title(f"Normalized Confusion Matrix ({model_name})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}_normalized.png"))
        plt.close()

        pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(
            os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}_data.csv")
        )
        pd.DataFrame(cm_norm, index=encoder.classes_, columns=encoder.classes_).to_csv(
            os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}_normalized_data.csv")
        )

        return {
            "Model": model_name,
            "CV_Accuracy_Mean": cv_mean,
            "CV_Accuracy_STD": cv_std,
            "Test_Accuracy": test_accuracy
        }

    print("\n=== Final Evaluation with 5-Fold Cross-Validation & Test Sets ===")
    results = []

    for model_name, model in best_models.items():
        if model_name == "XGBoost":
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
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "classification_metrics_all_models.csv"), index=False)

    # ============================================
    # 7. SAVE MODELS & ARTIFACTS
    # ============================================
    print("\nSaving trained models and artifacts for replication...")
    for model_name, model in best_models.items():
        joblib.dump(model, os.path.join(OUTPUT_DIR, f"best_{model_name.lower()}_model.pkl"))

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
    pd.DataFrame(features_to_keep, columns=["Feature"]).to_csv(
        os.path.join(OUTPUT_DIR, "features_to_keep.csv"), index=False
    )

    print("\nAll done! Check the 'outputs' folder for CSV, PKL, and PNG files.")

if __name__ == "__main__":
    main()
