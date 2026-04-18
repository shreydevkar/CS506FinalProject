import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def evaluate_model(model, X, y):
    preds = model.predict(X)

    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }


def baseline_model(df, split_idx, test_len):
    # IMPORTANT: Calculate baseline only on test set for fair comparison
    # Baseline predicts RV_t+1 = RV_t (volatility persistence assumption)
    # This naive model serves as a benchmark for ML models
    df_clean = df.dropna()

    baseline_preds = df_clean["RV"].iloc[split_idx : split_idx + test_len]
    actual = df_clean["Target"].iloc[split_idx : split_idx + test_len]

    mse = mean_squared_error(actual, baseline_preds)

    return mse


def walk_forward_evaluate(fit_fn, X, y, n_splits=5):
    """Walk-forward cross-validation for time-series models.

    Why: a single chronological 80/20 split yields one evaluation point — the
    numbers depend heavily on what happened in the last 20% of data. Walk-forward
    CV trains on expanding past windows and evaluates on the next future block,
    giving n_splits evaluations we can average. sklearn's TimeSeriesSplit implements
    the expanding-window scheme correctly (no shuffling, no future leakage).

    fit_fn(X_train, y_train) -> trained model.

    Returns per-fold metrics plus mean/std across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = fit_fn(X.iloc[train_idx], y.iloc[train_idx])
        metrics = evaluate_model(model, X.iloc[val_idx], y.iloc[val_idx])
        metrics["fold"] = fold_idx
        metrics["n_train"] = len(train_idx)
        metrics["n_val"] = len(val_idx)
        fold_metrics.append(metrics)

    mse_values = [m["MSE"] for m in fold_metrics]
    mae_values = [m["MAE"] for m in fold_metrics]
    r2_values = [m["R2"] for m in fold_metrics]

    return {
        "folds": fold_metrics,
        "MSE_mean": float(np.mean(mse_values)),
        "MSE_std": float(np.std(mse_values)),
        "MAE_mean": float(np.mean(mae_values)),
        "MAE_std": float(np.std(mae_values)),
        "R2_mean": float(np.mean(r2_values)),
        "R2_std": float(np.std(r2_values)),
    }


def walk_forward_baseline(df, n_splits=5):
    """Walk-forward persistence baseline for apples-to-apples comparison with walk_forward_evaluate."""
    df_clean = df.dropna()
    X = df_clean[["RV"]]  # only shape matters here; we predict RV_t directly
    y = df_clean["Target"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mses = []
    for train_idx, val_idx in tscv.split(X):
        preds = df_clean["RV"].iloc[val_idx].values
        actual = y.iloc[val_idx].values
        mses.append(mean_squared_error(actual, preds))
    return {
        "MSE_mean": float(np.mean(mses)),
        "MSE_std": float(np.std(mses)),
        "folds_MSE": mses,
    }