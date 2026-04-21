import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (  # noqa: E402
    BASE_FEATURES,
    SENTIMENT_FEATURES,
    add_features,
    prepare_data,
)
from src.evaluate import baseline_model, evaluate_model  # noqa: E402


@pytest.fixture
def synthetic_ohlcv():
    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range("2015-01-02", periods=n, freq="B")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    df = pd.DataFrame(
        {
            "Open": price,
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Adj Close": price,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_add_features_produces_expected_columns(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv)
    for col in BASE_FEATURES + ["log_return", "Target"]:
        assert col in out.columns, f"missing engineered column: {col}"


def test_realized_volatility_equals_squared_log_return(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv).dropna()
    np.testing.assert_allclose(out["RV"].values, out["log_return"].values ** 2)


def test_target_is_next_day_rv_no_leakage(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv)
    aligned = out[["RV", "Target"]].dropna()
    np.testing.assert_allclose(
        aligned["Target"].values[:-1],
        aligned["RV"].values[1:],
    )


def test_rsi_stays_in_valid_range(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv).dropna()
    assert out["RSI_14"].between(0, 100).all()


def test_prepare_data_split_is_chronological(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv)
    X_train, X_test, y_train, y_test = prepare_data(out, test_size=0.2)
    assert X_train.index.max() < X_test.index.min()
    assert len(X_train) + len(X_test) == len(out.dropna())


def test_prepare_data_feature_count_matches_flag(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv)
    X_train, _, _, _ = prepare_data(out, use_sentiment=False)
    assert list(X_train.columns) == BASE_FEATURES


def test_prepare_data_raises_when_sentiment_requested_without_columns(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv)
    with pytest.raises(ValueError, match="sentiment columns missing"):
        prepare_data(out, use_sentiment=True)


def test_add_features_merges_sentiment_and_fills_neutral(synthetic_ohlcv):
    sent_idx = synthetic_ohlcv.index[:10]
    sentiment_df = pd.DataFrame(
        {
            "sentiment_mean": np.linspace(-0.5, 0.5, 10),
            "sentiment_std": np.full(10, 0.1),
            "news_count": np.arange(1, 11),
        },
        index=sent_idx,
    )
    out = add_features(synthetic_ohlcv, sentiment_df=sentiment_df)
    for col in SENTIMENT_FEATURES:
        assert col in out.columns
    # Days outside the sentiment window should be neutral-filled, not NaN
    after = out.loc[out.index > sent_idx[-1], ["sentiment_mean", "sentiment_std", "news_count"]]
    assert (after == 0.0).all().all()


def test_evaluate_model_returns_expected_metrics(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv)
    X_train, X_test, y_train, y_test = prepare_data(out)
    model = LinearRegression().fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert set(metrics.keys()) == {"MSE", "MAE", "R2"}
    assert metrics["MSE"] >= 0
    assert metrics["MAE"] >= 0


def test_baseline_model_returns_nonnegative_mse(synthetic_ohlcv):
    out = add_features(synthetic_ohlcv)
    df_clean = out.dropna()
    split_idx = int(len(df_clean) * 0.8)
    test_len = len(df_clean) - split_idx
    mse = baseline_model(out, split_idx, test_len)
    assert mse >= 0
