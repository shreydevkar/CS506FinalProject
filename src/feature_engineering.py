import numpy as np
import pandas as pd

BASE_FEATURES = [
    "RV",
    "rolling_vol_5",
    "rolling_vol_10",
    "SMA_10",
    "SMA_20",
    "volume_change",
]

SENTIMENT_FEATURES = [
    "sentiment_mean",
    "sentiment_std",
    "news_count",
]


def add_features(df, sentiment_df=None):
    """Engineer features for predicting next-day volatility: volatility patterns, price trends, volume signals.

    If sentiment_df is provided (indexed by date, columns: sentiment_mean/sentiment_std/news_count),
    it is merged on the date index BEFORE Target is computed, and missing days are filled with
    neutral sentiment (0 mean, 0 std, 0 count).
    """
    df = df.copy()

    # Logarithmic returns: normalized price changes (log scale) for statistical stability
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Realized volatility: squared returns measuring daily price movement magnitude
    df["RV"] = df["log_return"] ** 2

    # Rolling volatility (5 & 10-day): capture volatility clustering—high vol tends to persist
    df["rolling_vol_5"] = df["RV"].rolling(5).mean()
    df["rolling_vol_10"] = df["RV"].rolling(10).mean()

    # Moving averages (10 & 20-day): identify market regime
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # Volume change: volume spikes often precede volatility spikes
    df["volume_change"] = df["Volume"].pct_change()

    # Merge daily sentiment (before computing Target to avoid leakage from future sentiment rows)
    if sentiment_df is not None and len(sentiment_df) > 0:
        sent = sentiment_df.copy()
        sent.index = pd.to_datetime(sent.index).normalize()
        idx = pd.to_datetime(df.index).normalize()
        df = df.assign(
            sentiment_mean=idx.map(sent["sentiment_mean"]).astype(float),
            sentiment_std=idx.map(sent["sentiment_std"]).astype(float),
            news_count=idx.map(sent["news_count"]).astype(float),
        )
        # Neutral fill for dates with no headlines
        df[["sentiment_mean", "sentiment_std", "news_count"]] = (
            df[["sentiment_mean", "sentiment_std", "news_count"]].fillna(0.0)
        )

    # Target: tomorrow's RV (what we predict). Computed LAST so no merged column can leak into it.
    df["Target"] = df["RV"].shift(-1)

    return df


def prepare_data(df, test_size=0.2, use_sentiment=False):
    """Split data chronologically for time-series modeling (no lookahead bias).

    use_sentiment=True appends sentiment_mean/sentiment_std/news_count to the feature set.
    Those columns must already exist on df (call add_features with sentiment_df first).
    """
    df = df.dropna()

    features = list(BASE_FEATURES)
    if use_sentiment:
        missing = [c for c in SENTIMENT_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"use_sentiment=True but sentiment columns missing: {missing}")
        features = features + SENTIMENT_FEATURES

    X = df[features]
    y = df["Target"]

    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test
