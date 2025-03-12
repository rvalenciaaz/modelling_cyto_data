# train_vae.py

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import json

from logging_utils import get_logger
from gpu_logging import log_gpu_usage_periodically
from data_utils import load_and_combine_csv, mad_filter

from pipeline_vae import (
    build_vae_pipeline,
    get_param_grid_vae,
    build_grid_search,
    kfold_evaluate_vae
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
    logger = get_logger(log_file='training_vae.log')
    logger.info("===== Starting UNSUPERVISED VAE Training Script =====")

    set_random_seeds(42)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info("CUDA is available -> Using GPU.")
        log_gpu_usage_periodically(interval_sec=900, logfile="gpu_usage.log")
    else:
        logger.info("CUDA not available -> Using CPU.")

    # 1. Load data
    #    Even though we have "Label" in data, let's ignore it for unsupervised VAE.
    combined_df = load_and_combine_csv(pattern="species*.csv", subsample=None)
    logger.info(f"Combined data shape: {combined_df.shape}")

    # 2. MAD filter
    #    If you want purely unsupervised, you can remove "Label" or do the same approach.
    filtered_df = mad_filter(combined_df, label_col="Label", mad_threshold=5)
    logger.info(f"Filtered data shape: {filtered_df.shape}")

    # We don't need the label for unsupervised. 
    # Let's just do X = numeric features, ignoring 'Label'
    feature_cols = [col for col in filtered_df.columns if col != "Label"]
    X = filtered_df[feature_cols].values.astype(np.float32)

    logger.info(f"Feature shape for VAE: {X.shape}")

    # 3. Build pipeline & grid search
    vae_pipeline = build_vae_pipeline(verbose=False)
    param_grid = get_param_grid_vae()
    # We'll use scoring='neg_mean_squared_error' or just the pipeline's .score() 
    # But let's define a custom build_grid_search that uses pipeline's .score() method:
    gs = build_grid_search(vae_pipeline, param_grid, cv=3, scoring=None)  
    # scoring=None -> uses pipeline.score() by default, which for VAE is negative reconstruction

    # 4. Grid Search
    logger.info("Running GridSearchCV for VAE...")
    gs.fit(X)  # no labels needed
    logger.info("Grid search complete.")
    best_params = gs.best_params_
    logger.info(f"Best params for VAE: {best_params}")

    # 5. Re-fit with more epochs
    logger.info("Refitting best VAE with more epochs (e.g., 50) on entire dataset...")
    best_estimator = gs.best_estimator_
    best_estimator.set_params(vae__epochs=50, vae__verbose=False)
    best_estimator.fit(X)

    # 6. K-Fold cross-validation 
    logger.info("K-Fold CV evaluation for the final unsupervised VAE pipeline...")
    scores = kfold_evaluate_vae(best_estimator, X, n_splits=5)
    scores = np.array(scores)
    mean_score = scores.mean()
    std_score = scores.std()
    logger.info(f"5-Fold CV VAE scores (negative recon error): {scores}")
    logger.info(f"Mean = {mean_score:.4f}, Std = {std_score:.4f}")

    # 7. Final "score" on entire dataset (no separate test set here, purely unsupervised)
    final_score = best_estimator.score(X)  # negative recon error
    logger.info(f"Final negative recon error on entire dataset: {final_score:.4f}")

    # 8. Save results
    metrics = {
        "best_params": best_params,
        "cv_scores": scores.tolist(),
        "cv_mean": float(mean_score),
        "cv_std": float(std_score),
        "final_score": float(final_score)
    }
    with open("metrics_vae.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to 'metrics_vae.json'.")

    # 9. Save final model 
    # The actual VAE model is best_estimator.named_steps['vae'].model_
    # We can also store feature names or other data if needed.
    vae_data = {
        "state_dict": best_estimator.named_steps['vae'].model_.state_dict(),
        "params": best_estimator.named_steps['vae'].get_params(),
        "feature_cols": feature_cols
    }
    torch.save(vae_data, "best_vae.pth")
    logger.info("Saved 'best_vae.pth' with unsupervised VAE weights and params.")

    logger.info("===== Unsupervised VAE Training Complete! =====")


if __name__ == "__main__":
    main()
