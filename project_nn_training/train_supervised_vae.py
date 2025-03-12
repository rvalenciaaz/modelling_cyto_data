# train_supervised_vae.py

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import json

from logging_utils import get_logger
from gpu_logging import log_gpu_usage_periodically
from data_utils import load_and_combine_csv, mad_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from pipeline_vae import (
    build_supervised_vae_pipeline,
    get_param_grid_supervised_vae,
    build_grid_search,
    kfold_evaluate_supervised_vae
)

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

def main():
    logger = get_logger(log_file='training_supervised_vae.log')
    logger.info("===== Starting SUPERVISED VAE Training Script =====")

    set_random_seeds(42)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info("CUDA is available -> Using GPU.")
        log_gpu_usage_periodically(interval_sec=900, logfile="gpu_usage.log")
    else:
        logger.info("CUDA not available -> Using CPU.")

    # 1. Load data
    combined_df = load_and_combine_csv(pattern="species*.csv", subsample=None)
    logger.info(f"Combined data shape: {combined_df.shape}")

    # 2. Filter by MAD
    filtered_df = mad_filter(combined_df, label_col="Label", mad_threshold=5)
    logger.info(f"Filtered data shape: {filtered_df.shape}")

    # 3. Prepare X, y
    feature_cols = [col for col in filtered_df.columns if col != "Label"]
    X = filtered_df[feature_cols].values.astype(np.float32)
    y_raw = filtered_df["Label"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    logger.info(f"Features shape = {X.shape}, #Classes = {len(label_encoder.classes_)}")

    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 5. Build pipeline & GridSearch
    sup_vae_pipeline = build_supervised_vae_pipeline(verbose=False)
    param_grid = get_param_grid_supervised_vae()
    gs = build_grid_search(sup_vae_pipeline, param_grid, cv=3, scoring='accuracy')

    # 6. Grid Search
    logger.info("Running GridSearchCV for Supervised VAE (classification)...")
    gs.fit(X_train, y_train)
    logger.info("GridSearch complete.")
    best_params = gs.best_params_
    logger.info(f"Best params: {best_params}")

    # 7. Re-fit with more epochs
    logger.info("Refitting best supervised VAE with more epochs (50) on the full training set...")
    best_estimator = gs.best_estimator_
    best_estimator.set_params(vae__epochs=50, vae__verbose=False)
    best_estimator.fit(X_train, y_train)

    # 8. K-Fold CV for final model
    logger.info("Performing 5-Fold CV with the final Supervised VAE pipeline...")
    fold_accs = kfold_evaluate_supervised_vae(best_estimator, X_train, y_train, n_splits=5)
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    logger.info(f"Fold accuracies: {fold_accs}")
    logger.info(f"Mean CV accuracy = {mean_acc:.4f}, Std Dev = {std_acc:.4f}")

    # 9. Evaluate on held-out test set
    y_pred = best_estimator.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    logger.info("\nClassification Report:\n" + class_report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Supervised VAE)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_supervised_vae.png", bbox_inches='tight')
    plt.close()
    logger.info("Saved 'confusion_matrix_supervised_vae.png'.")

    # 10. Save metrics
    metrics = {
        "best_params": best_params,
        "cv_fold_accuracies": fold_accs,
        "cv_mean_acc": float(mean_acc),
        "cv_std_acc": float(std_acc),
        "test_accuracy": float(test_acc),
        "classification_report": class_report
    }
    with open("metrics_supervised_vae.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved 'metrics_supervised_vae.json'.")

    # 11. Save final model
    import torch
    sup_vae_data = {
        # pipeline step name is 'vae'
        "state_dict": best_estimator.named_steps['vae'].model_.state_dict(),
        "params": best_estimator.named_steps['vae'].get_params(),
        "label_encoder": label_encoder.classes_.tolist(),
        "feature_cols": feature_cols
    }
    torch.save(sup_vae_data, "best_supervised_vae.pth")
    logger.info("Saved 'best_supervised_vae.pth' with supervised VAE weights & params.")

    logger.info("===== Supervised VAE Training Complete! =====")


if __name__ == "__main__":
    main()
