# train.py
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Local modules
from logging_utils import get_logger
from gpu_logging import log_gpu_usage_periodically
from data_utils import load_and_combine_csv, mad_filter
from pipeline import (
    build_semi_supervised_pipeline,
    get_param_grid,
    build_grid_search,
    kfold_evaluate
)

def set_random_seeds(seed=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For fully deterministic runs (note: may slow training)
        cudnn.deterministic = True
        cudnn.benchmark = False

def main():
    # 1. Configure logger
    logger = get_logger(log_file='training.log')
    logger.info("===== Starting Training Script =====")

    # 2. Set seeds
    set_random_seeds(42)

    # 3. Check CUDA & start GPU logging if available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info("CUDA is available -> Using GPU.")
        log_gpu_usage_periodically(interval_sec=900, logfile="gpu_usage.log")
    else:
        logger.info("CUDA not available -> Using CPU.")

    # 4. Load data
    logger.info("Loading and combining CSV files...")
    combined_df = load_and_combine_csv(pattern="species*.csv", subsample=None)
    logger.info(f"Combined data shape: {combined_df.shape}")

    # 5. Filter features by MAD
    logger.info("Applying MAD filter to numeric columns...")
    filtered_df = mad_filter(combined_df, label_col="Label", mad_threshold=5)
    logger.info(f"Filtered data shape: {filtered_df.shape}")

    # 6. Prepare X, y, label encoder
    feature_names = filtered_df.drop(columns=["Label"]).columns.tolist()
    X = filtered_df.drop(columns=["Label"]).values
    y_raw = filtered_df["Label"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    # 7. Train/Test split
    logger.info("Splitting data into train & test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 8. Mask 50% of training labels for semi-supervised
    logger.info("Masking 50% of training labels to simulate unlabeled samples...")
    y_train_ss = y_train.copy()
    mask = np.random.rand(len(y_train_ss)) < 0.5
    y_train_ss[mask] = -1

    # 9. Build pipeline & GridSearchCV
    logger.info("Building semi-supervised pipeline & preparing GridSearch...")
    pipeline = build_semi_supervised_pipeline(verbose=False)
    param_grid = get_param_grid()
    grid_search = build_grid_search(pipeline, param_grid, cv=3)

    # 10. Run grid search
    logger.info("Running GridSearchCV...")
    grid_search.fit(X_train, y_train_ss)
    logger.info("GridSearchCV complete.")
    best_params = grid_search.best_params_
    logger.info(f"Best hyperparams: {best_params}")

    # 11. Re-fit with more epochs
    logger.info("Refitting best model with more epochs (e.g., 50) on full training set...")
    best_estimator = grid_search.best_estimator_
    best_estimator.base_estimator.set_params(nn__epochs=50, nn__verbose=False)
    best_estimator.fit(X_train, y_train_ss)

    # 12. K-Fold evaluation (optional)
    logger.info("Performing K-Fold evaluation on training set with best estimator...")
    fold_accs = kfold_evaluate(best_estimator.base_estimator, X_train, y_train, n_splits=5)
    mean_cv = np.mean(fold_accs)
    std_cv = np.std(fold_accs)
    logger.info(f"5-Fold CV accuracies: {fold_accs}")
    logger.info(f"Mean CV = {mean_cv:.4f}, Std Dev = {std_cv:.4f}")

    # 13. Final test evaluation
    logger.info("Evaluating on test set...")
    y_pred = best_estimator.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    logger.info("\nClassification Report:\n" + class_report)

    # 14. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Semi-Supervised Model)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_nn.png", bbox_inches='tight')
    plt.close()
    logger.info("Saved 'confusion_matrix_nn.png'.")

    # 15. Save metrics
    metrics_dict = {
        "test_accuracy": float(test_acc),
        "classification_report": class_report,
        "cv_fold_accuracies": fold_accs,
        "cv_mean_accuracy": float(mean_cv),
        "cv_std_accuracy": float(std_cv),
        "best_params": best_params
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info("Saved metrics to 'metrics.json'.")

    # 16. Save final model & label info
    #    We store:
    #      - PyTorch model state dict
    #      - Model parameters
    #      - Label encoder classes
    #      - Feature names (for reference)
    best_estimator_data = {
        "state_dict": best_estimator.base_estimator.named_steps['nn'].model_.state_dict(),
        "params": best_estimator.base_estimator.named_steps['nn'].get_params(),
        "label_encoder_classes": label_encoder.classes_.tolist(),
        "feature_names": feature_names
    }
    torch.save(best_estimator_data, "best_estimator.pth")
    logger.info("Saved best_estimator.pth (model weights + params + label classes).")

    logger.info("===== Training Complete! =====")


if __name__ == "__main__":
    main()
