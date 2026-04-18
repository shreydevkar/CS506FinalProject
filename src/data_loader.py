import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start="2013-01-01", end="2018-01-01"):
    data = yf.download(ticker, start=start, end=end)

    os.makedirs("data/raw", exist_ok=True)
    data.to_csv(f"data/raw/{ticker}.csv")

    return data


def load_data(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)