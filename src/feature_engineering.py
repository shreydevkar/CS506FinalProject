import numpy as np
import pandas as pd

def add_features(df):
    df = df.copy()

    # Log returns
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Realized volatility proxy
    df["RV"] = df["log_return"] ** 2

    # Target (next day volatility)
    df["Target"] = df["RV"].shift(-1)

    # Rolling volatility
    df["rolling_vol_5"] = df["RV"].rolling(5).mean()
    df["rolling_vol_10"] = df["RV"].rolling(10).mean()

    # Moving averages
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # Volume change
    df["volume_change"] = df["Volume"].pct_change()

    return df


def prepare_data(df, test_size=0.2):
    df = df.dropna()

    features = [
        "RV",
        "rolling_vol_5",
        "rolling_vol_10",
        "SMA_10",
        "SMA_20",
        "volume_change"
    ]

    X = df[features]
    y = df["Target"]

    # IMPORTANT: Time series split to prevent data leakage
    # Use chronological split: train on past data, test on future data
    # This ensures realistic evaluation of predictive models
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test