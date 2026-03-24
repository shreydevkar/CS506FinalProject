import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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