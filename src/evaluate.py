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


def baseline_model(df):
    # naive baseline: RV_t+1 = RV_t
    baseline_preds = df["RV"].shift(0)
    actual = df["Target"]

    mask = ~baseline_preds.isna() & ~actual.isna()

    mse = mean_squared_error(actual[mask], baseline_preds[mask])

    return mse