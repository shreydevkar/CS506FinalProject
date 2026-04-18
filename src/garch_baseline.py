"""GARCH(1,1) benchmark for volatility prediction.

Why include GARCH alongside the persistence baseline:
- Persistence (RV_t+1 = RV_t) is a trivial baseline — it tells you whether ML beats
  "do nothing". But volatility's classical forecasting baseline in finance is GARCH,
  because it explicitly models the two stylized facts of financial volatility:
    (1) volatility clustering (past shocks predict current volatility),
    (2) mean reversion (volatility drifts toward a long-run level).
- A volatility-prediction project that doesn't at least report GARCH is skipping
  the benchmark practitioners actually use. This makes the ML numbers defensible
  rather than compared only against a toy.

Implementation: rolling 1-step-ahead forecasts — refit the GARCH(1,1) on each
expanding training window, then produce sigma^2 forecast for t+1. We compare this
sigma^2 forecast directly against realized volatility r_{t+1}^2 (both represent
variance at horizon 1), using MSE on the same test indices the ML models use.
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error


def garch_forecast_series(df, test_start_idx, refit_every=20):
    """Rolling 1-step variance forecasts from GARCH(1,1) for the test window.

    df: DataFrame with log_return column (engineered by feature_engineering.add_features).
    test_start_idx: first integer index (in df.dropna()) where testing begins.
    refit_every: refit the GARCH model every N steps (not every step) for speed.
        Between refits, we reuse parameters and roll the conditional variance forward.

    Returns a pandas Series of predicted next-day variances aligned to df's test index.
    """
    df_clean = df.dropna().copy()

    # arch expects a series of returns (scaled up to avoid numerical issues; finance
    # log-returns are ~0.01 scale, and GARCH optimization is happier at ~1).
    returns = df_clean["log_return"] * 100.0

    test_idx = df_clean.index[test_start_idx:]
    preds = []
    model_res = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step, target_idx in enumerate(test_idx):
            if model_res is None or step % refit_every == 0:
                # Fit on all data up to (but excluding) the test target
                history = returns.iloc[: test_start_idx + step]
                model = arch_model(history, vol="GARCH", p=1, q=1, rescale=False, mean="zero")
                model_res = model.fit(disp="off", show_warning=False)

            # 1-step-ahead variance forecast in the rescaled units
            forecast = model_res.forecast(horizon=1, reindex=False)
            sigma2_scaled = float(forecast.variance.values[-1, 0])
            # Convert back to original log-return variance (undo the *100 scaling)
            preds.append(sigma2_scaled / (100.0 ** 2))

    return pd.Series(preds, index=test_idx, name="garch_pred_var")


def garch_test_mse(df, test_start_idx, refit_every=20):
    """Return GARCH(1,1) MSE on the test window. Target is realized volatility RV_{t+1}."""
    df_clean = df.dropna()
    preds = garch_forecast_series(df, test_start_idx, refit_every=refit_every)
    actual = df_clean["Target"].iloc[test_start_idx : test_start_idx + len(preds)]
    return mean_squared_error(actual.values, preds.values)
