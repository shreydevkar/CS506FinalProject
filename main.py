from src.data_loader import fetch_stock_data
from src.feature_engineering import add_features, prepare_data, BASE_FEATURES, SENTIMENT_FEATURES
from src.model import train_linear_regression, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, baseline_model
import argparse


def _train_and_eval(X_train, X_test, y_train, y_test):
    lr = train_linear_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    return {
        "models": {"lr": lr, "rf": rf, "xgb": xgb_model},
        "eval": {
            "lr": evaluate_model(lr, X_test, y_test),
            "rf": evaluate_model(rf, X_test, y_test),
            "xgb": evaluate_model(xgb_model, X_test, y_test),
        },
    }


def run_pipeline(ticker, use_sentiment=False):
    #  Step 1: Data ingestion
    df = fetch_stock_data(ticker)

    #  Step 2: Feature engineering (optionally with sentiment)
    sentiment_df = None
    if use_sentiment:
        from src.news_sentiment import build_sentiment_features
        print(f"[pipeline] Fetching/loading sentiment features for {ticker}")
        sentiment_df = build_sentiment_features(ticker)
        print(f"[pipeline] Sentiment coverage: {len(sentiment_df)} days")
    df = add_features(df, sentiment_df=sentiment_df)

    #  Step 3: Train-test split
    X_train, X_test, y_train, y_test = prepare_data(df, use_sentiment=use_sentiment)

    #  Forecasting preparation
    features = list(BASE_FEATURES) + (SENTIMENT_FEATURES if use_sentiment else [])
    df_clean = df.dropna()
    last_features = df_clean.iloc[-1:][features]

    #  Step 4+5: Train and evaluate
    trained = _train_and_eval(X_train, X_test, y_train, y_test)
    lr, rf, xgb_model = trained["models"]["lr"], trained["models"]["rf"], trained["models"]["xgb"]

    label = "with sentiment" if use_sentiment else "baseline (no sentiment)"
    print(f"\nTest Set Evaluation [{label}]:")
    print("Baseline MSE:", baseline_model(df, len(X_train), len(y_test)))
    print("\nLinear Regression:", trained["eval"]["lr"])
    print("\nRandom Forest:", trained["eval"]["rf"])
    print("\nXGBoost:", trained["eval"]["xgb"])

    #  Step 6: Next-day forecast
    print(f"\nNext Day Volatility Forecast [{label}]:")
    lr_pred = lr.predict(last_features)[0]
    rf_pred = rf.predict(last_features)[0]
    xgb_pred = xgb_model.predict(last_features)[0]
    print(f"Linear Regression: {lr_pred:.6f}")
    print(f"Random Forest: {rf_pred:.6f}")
    print(f"XGBoost: {xgb_pred:.6f}")
    print(f"Baseline (current RV): {last_features['RV'].iloc[0]:.6f}")

    return {
        "df": df,
        "df_clean": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": trained["models"],
        "predictions": {
            "lr_pred": lr_pred,
            "rf_pred": rf_pred,
            "xgb_pred": xgb_pred,
            "baseline": last_features["RV"].iloc[0],
        },
        "eval": {
            "baseline_mse": baseline_model(df, len(X_train), len(y_test)),
            **trained["eval"],
        },
        "last_features": last_features,
        "use_sentiment": use_sentiment,
        "sentiment_df": sentiment_df,
    }


def run_comparison(ticker):
    """Run pipeline twice (without and with sentiment) and return both result dicts."""
    baseline = run_pipeline(ticker, use_sentiment=False)
    sentiment = run_pipeline(ticker, use_sentiment=True)
    return {"baseline": baseline, "sentiment": sentiment}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run volatility analysis pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--use-sentiment", action="store_true", help="Include news sentiment features")
    parser.add_argument("--compare", action="store_true", help="Run both variants and compare")
    args = parser.parse_args()

    if args.compare:
        run_comparison(args.ticker)
    else:
        run_pipeline(args.ticker, use_sentiment=args.use_sentiment)
