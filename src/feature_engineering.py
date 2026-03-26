import numpy as np
import pandas as pd

def add_features(df):
    """Engineer features for predicting next-day volatility: volatility patterns, price trends, volume signals."""
    df = df.copy()

    # Logarithmic returns: normalized price changes (log scale) for statistical stability
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Realized volatility: squared returns measuring daily price movement magnitude
    df["RV"] = df["log_return"] ** 2

    # Target: tomorrow's RV (what we predict)
    df["Target"] = df["RV"].shift(-1)

    # Rolling volatility (5 & 10-day): capture volatility clustering—high vol tends to persist
    df["rolling_vol_5"] = df["RV"].rolling(5).mean()
    df["rolling_vol_10"] = df["RV"].rolling(10).mean()

    # Moving averages (10 & 20-day): identify market regime—volatility differs in trends vs choppy sideways
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # Volume change: volume spikes often precede volatility spikes (market reaction signal)
    df["volume_change"] = df["Volume"].pct_change()

    return df


def prepare_data(df, test_size=0.2):
    """Split data chronologically for time-series modeling (no lookahead bias)."""
    df = df.dropna()

    # Define prediction features
    features = [
        "RV",                # Current volatility
        "rolling_vol_5",     # 5-day volatility trend
        "rolling_vol_10",    # 10-day volatility trend
        "SMA_10",            # 10-day price trend
        "SMA_20",            # 20-day price trend
        "volume_change"      # Daily volume change
    ]

    X = df[features]
    y = df["Target"]

    # Chronological split: train on past, test on future (prevents lookahead bias in time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]              # Historical training set
    X_test = X.iloc[split_idx:]               # Future test set
    y_train = y.iloc[:split_idx]              # Training targets
    y_test = y.iloc[split_idx:]               # Testing targets

    return X_train, X_test, y_train, y_test