from src.data_loader import fetch_stock_data
from src.feature_engineering import add_features, prepare_data
from src.model import train_linear_regression, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, baseline_model
import argparse

def run_pipeline(ticker):
    # Step 1: Data
    df = fetch_stock_data(ticker)

    # Step 2: Features
    df = add_features(df)

    # Step 3: Prepare
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Get last available features for forecasting (from the end of training data)
    # IMPORTANT: Use the most recent data point for prediction
    # This ensures we're forecasting beyond the test set with available information
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

    # Step 4: Models
    # IMPORTANT: Train models only on historical training data
    # This prevents the models from seeing future test data during training
    lr = train_linear_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Step 5: Evaluation on test set
    # IMPORTANT: Evaluate models on unseen test data (future time periods)
    # This prevents overfitting and gives realistic performance estimates
    print("Test Set Evaluation:")
    print("Baseline MSE:", baseline_model(df, len(X_train), len(y_test)))

    print("\nLinear Regression:", evaluate_model(lr, X_test, y_test))
    print("\nRandom Forest:", evaluate_model(rf, X_test, y_test))
    print("\nXGBoost:", evaluate_model(xgb_model, X_test, y_test))

    # Step 6: Forecast next day's volatility
    # IMPORTANT: Use the most recent complete data point for out-of-sample prediction
    # This simulates real-world forecasting beyond the test set
    print("\n=== Next Day Volatility Forecast ===")
    lr_pred = lr.predict(last_features)[0]
    rf_pred = rf.predict(last_features)[0]
    xgb_pred = xgb_model.predict(last_features)[0]

    print(f"Linear Regression: {lr_pred:.6f}")
    print(f"Random Forest: {rf_pred:.6f}")
    print(f"XGBoost: {xgb_pred:.6f}")
    print(f"Baseline (current RV): {last_features['RV'].iloc[0]:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run volatility analysis pipeline')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol (default: AAPL)')
    args = parser.parse_args()
    run_pipeline(args.ticker)