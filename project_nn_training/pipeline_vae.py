# pipeline_vae.py

from sklearn.pipeline import Pipeline
from models_vae import PyTorchVAEWrapper, PyTorchSupervisedVAEWrapper
from sklearn.model_selection import GridSearchCV, KFold

def build_vae_pipeline(verbose=True):
    """
    Creates a pipeline for an UNSUPERVISED VAE (no scaler if you prefer to do 
    everything inside the model, but let's add a possibility of scaling).
    """
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('vae', PyTorchVAEWrapper(verbose=verbose))
    ])
    return pipeline

def build_supervised_vae_pipeline(verbose=True):
    """
    Creates a pipeline for a SUPERVISED VAE (again with an optional scaler).
    """
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('vae', PyTorchSupervisedVAEWrapper(verbose=verbose))
    ])
    return pipeline

def get_param_grid_vae():
    """
    Example param grid for the UNSUPERVISED VAE pipeline.
    The pipeline steps are ('scaler', 'vae'), so we reference the VAE with 'vae__<param>'.
    """
    param_grid = {
        'vae__latent_dim': [4, 8],
        'vae__hidden_size': [64, 128],
        'vae__learning_rate': [1e-3, 1e-4],
        'vae__epochs': [10],   # short for demonstration
        # Add more if desired
    }
    return param_grid

def get_param_grid_supervised_vae():
    """
    Example param grid for the SUPERVISED VAE pipeline.
    """
    param_grid = {
        'vae__latent_dim': [4, 8],
        'vae__hidden_size': [64, 128],
        'vae__learning_rate': [1e-3, 1e-4],
        'vae__epochs': [10],  # short for demonstration
    }
    return param_grid

def build_grid_search(estimator, param_grid, cv=3, scoring='accuracy'):
    """
    Build a GridSearchCV from the given estimator & param_grid.
    """
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        verbose=2,
        return_train_score=True
    )
    return gs

def kfold_evaluate_vae(estimator, X, n_splits=5, random_state=42):
    """
    K-fold cross-validation for the unsupervised VAE. We'll measure the average reconstruction
    (i.e., we use estimator.score() = negative loss).
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        # We don't have labels for unsupervised
        est = estimator.__class__()
        est.set_params(**estimator.get_params())  # replicate settings
        est.fit(X_train)
        fold_score = est.score(X_val)
        scores.append(fold_score)
    return scores

def kfold_evaluate_supervised_vae(estimator, X, y, n_splits=5, random_state=42):
    """
    K-fold cross-validation for the supervised VAE. We measure classification accuracy.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    from sklearn.base import clone
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # clone the pipeline
        est = clone(estimator)
        est.fit(X_train, y_train)
        preds = est.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
    return scores
