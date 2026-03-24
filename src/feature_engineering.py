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


def prepare_data(df):
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

    return X, y