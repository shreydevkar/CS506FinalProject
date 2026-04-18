"""Hyperparameter tuning with chronological (time-series) cross-validation.

Why time-series CV: the default KFold shuffles rows, which leaks future information
into the training folds for a time series. sklearn's TimeSeriesSplit instead uses
expanding-window folds (train on [0:t], validate on [t:t+k]), preserving temporal
ordering. This matches how the model will actually be used — training on the past to
predict the future.

Grids are intentionally small so a tune run completes in ~minutes; extend if needed.
"""
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb


RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8, None],
    "min_samples_leaf": [5, 10, 20],
    "max_features": ["sqrt", 0.5],
}

XGB_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 8],
    "learning_rate": [0.05, 0.1],
    "reg_lambda": [1.0, 5.0],
}


def _grid_combinations(grid):
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def _cv_score(model_cls, params, X, y, n_splits=4, fixed_kwargs=None):
    """Mean MSE across expanding-window folds. Lower is better."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        model = model_cls(**{**(fixed_kwargs or {}), **params})
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores.append(mean_squared_error(y.iloc[val_idx], preds))
    return float(np.mean(scores))


def tune_random_forest(X_train, y_train, n_splits=4):
    """Grid search over RF_GRID using chronological CV. Returns (best_params, best_cv_mse)."""
    best = (None, float("inf"))
    fixed = {"random_state": 42, "n_jobs": -1}
    for params in _grid_combinations(RF_GRID):
        score = _cv_score(RandomForestRegressor, params, X_train, y_train,
                          n_splits=n_splits, fixed_kwargs=fixed)
        if score < best[1]:
            best = (params, score)
    return best


def tune_xgboost(X_train, y_train, n_splits=4):
    """Grid search over XGB_GRID using chronological CV. Returns (best_params, best_cv_mse)."""
    best = (None, float("inf"))
    fixed = {"random_state": 42, "verbosity": 0, "n_jobs": -1}
    for params in _grid_combinations(XGB_GRID):
        score = _cv_score(xgb.XGBRegressor, params, X_train, y_train,
                          n_splits=n_splits, fixed_kwargs=fixed)
        if score < best[1]:
            best = (params, score)
    return best


def train_tuned(X_train, y_train, n_splits=4):
    """Tune RF + XGB and return trained best models plus their chosen params.

    Use this in place of train_random_forest / train_xgboost when you want
    per-ticker tuned models.
    """
    rf_params, rf_cv = tune_random_forest(X_train, y_train, n_splits=n_splits)
    xgb_params, xgb_cv = tune_xgboost(X_train, y_train, n_splits=n_splits)

    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)
    rf.fit(X_train, y_train)
    xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1, **xgb_params)
    xgb_model.fit(X_train, y_train)

    return {
        "rf": rf,
        "xgb": xgb_model,
        "rf_params": rf_params,
        "xgb_params": xgb_params,
        "rf_cv_mse": rf_cv,
        "xgb_cv_mse": xgb_cv,
    }
