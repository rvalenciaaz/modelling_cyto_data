# pipeline.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from models import PyTorchNNClassifierWithVal

def build_semi_supervised_pipeline(verbose=True):
    """
    Returns a pipeline (StandardScaler + PyTorchNNClassifierWithVal)
    wrapped in SelfTrainingClassifier for semi-supervised learning.
    """
    from sklearn.semi_supervised import SelfTrainingClassifier
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nn', PyTorchNNClassifierWithVal(verbose=verbose))
    ])
    self_training_clf = SelfTrainingClassifier(
        base_estimator=base_pipeline,
        threshold=0.9,
        criterion='threshold',
        verbose=verbose
    )
    return self_training_clf

def build_supervised_pipeline(verbose=True):
    """
    Returns a pipeline for normal supervised classification:
    StandardScaler -> PyTorchNNClassifierWithVal
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nn', PyTorchNNClassifierWithVal(verbose=verbose))
    ])
    return pipeline

def get_param_grid_semi_supervised():
    """
    Example param grid for the SEMI-SUPERVISED pipeline's PyTorchNNClassifierWithVal hyperparams.
    (using the prefix 'base_estimator__nn__' because of SelfTrainingClassifier).
    """
    param_grid = {
        'base_estimator__nn__hidden_size':   [16, 32],
        'base_estimator__nn__num_layers':    [1, 2],
        'base_estimator__nn__learning_rate': [1e-3, 1e-4],
        'base_estimator__nn__batch_size':    [64],
        'base_estimator__nn__epochs':        [10],
    }
    return param_grid

def get_param_grid_supervised():
    """
    Example param grid for the SUPERVISED pipeline's PyTorchNNClassifierWithVal hyperparams.
    (here the prefix is just 'nn__' because it's directly in the pipeline).
    """
    param_grid = {
        'nn__hidden_size':   [16, 32],
        'nn__num_layers':    [1, 2],
        'nn__learning_rate': [1e-3, 1e-4],
        'nn__batch_size':    [64],
        'nn__epochs':        [10],
        # 'nn__early_stopping': [False, True], # optionally tune this
    }
    return param_grid

def build_grid_search(estimator, param_grid, cv=3):
    """
    Wrap the estimator in a GridSearchCV with the provided parameter grid.
    """
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=1,  # safer with custom PyTorch models
        verbose=2,
        return_train_score=True
    )
    return gs

def kfold_evaluate_supervised(estimator, X_train, y_train, n_splits=5, random_state=42):
    """
    Performs normal K-fold cross-validation on labeled data (no semi-supervised masking).
    Returns the list of fold accuracies.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    from sklearn.base import clone

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accuracies = []

    for train_index, val_index in kf.split(X_train, y_train):
        X_tr = X_train[train_index]
        X_val = X_train[val_index]
        y_tr = y_train[train_index]
        y_val = y_train[val_index]

        # Clone the pipeline for this fold
        fold_clf = clone(estimator)
        fold_clf.fit(X_tr, y_tr)

        preds = fold_clf.predict(X_val)
        fold_acc = accuracy_score(y_val, preds)
        fold_accuracies.append(fold_acc)

    return fold_accuracies
