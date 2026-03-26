from src.data_loader import fetch_stock_data
from src.feature_engineering import add_features, prepare_data
from src.model import train_linear_regression, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, baseline_model
import argparse

def run_pipeline(ticker):
    # --- Step 1: Data ingestion ---
    # Fetch historical price data for ticker and save to local raw data folder.
    # This function should be repeatable (same input -> same file output) and deterministic.
    df = fetch_stock_data(ticker)

    # --- Step 2: Feature engineering ---
    # Add volatility features (squared returns, rolling vol), technicals (SMA), and volume change.
    # Target for modeling is RV shifted by -1 (next-day realized volatility proxy).
    df = add_features(df)

    # --- Step 3: Train-test split ---
    # Use a chronological split (first 80% train, last 20% test) to avoid lookahead bias.
    X_train, X_test, y_train, y_test = prepare_data(df)

    # --- Forecasting preparation ---
    # Keep the latest feature vector (most recent day after dropna) for out-of-sample forecast.
    features = [
        "RV",
        "rolling_vol_5",
        "rolling_vol_10",
        "SMA_10",
        "SMA_20",
        "volume_change"
    ]
    df_clean = df.dropna()
    last_features = df_clean.iloc[-1:][features]

    # --- Step 4: Model training ---
    # Train each model only on training data; this is the core predictive training stage.
    lr = train_linear_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # --- Step 5: Model evaluation (test period) ---
    # Evaluate on held-out unseen data (future time in sequence) to measure real-world performance.
    print("Test Set Evaluation:")
    print("Baseline MSE:", baseline_model(df, len(X_train), len(y_test)))

    print("\nLinear Regression:", evaluate_model(lr, X_test, y_test))
    print("\nRandom Forest:", evaluate_model(rf, X_test, y_test))
    print("\nXGBoost:", evaluate_model(xgb_model, X_test, y_test))

    # --- Step 6: Next-day forecast (out-of-sample) ---
    # Use most recent clean feature row to simulate a live prediction for tomorrow.
    print("\n=== Next Day Volatility Forecast ===")
    lr_pred = lr.predict(last_features)[0]
    rf_pred = rf.predict(last_features)[0]
    xgb_pred = xgb_model.predict(last_features)[0]

    print(f"Linear Regression: {lr_pred:.6f}")
    print(f"Random Forest: {rf_pred:.6f}")
    print(f"XGBoost: {xgb_pred:.6f}")
    print(f"Baseline (current RV): {last_features['RV'].iloc[0]:.6f}")

    # Return structured outputs for reuse (e.g., notebook visualizations)
    return {
        'df': df,
        'df_clean': df_clean,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'models': {
            'lr': lr,
            'rf': rf,
            'xgb': xgb_model,
        },
        'predictions': {
            'lr_pred': lr_pred,
            'rf_pred': rf_pred,
            'xgb_pred': xgb_pred,
            'baseline': last_features['RV'].iloc[0]
        },
        'eval': {
            'baseline_mse': baseline_model(df, len(X_train), len(y_test)),
            'lr': evaluate_model(lr, X_test, y_test),
            'rf': evaluate_model(rf, X_test, y_test),
            'xgb': evaluate_model(xgb_model, X_test, y_test),
        },
        'last_features': last_features
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run volatility analysis pipeline')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol (default: AAPL)')
    args = parser.parse_args()
    run_pipeline(args.ticker)