from src.data_loader import fetch_stock_data
from src.feature_engineering import add_features, prepare_data
from src.model import train_linear_regression, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, baseline_model

def run_pipeline():
    # Step 1: Data
    df = fetch_stock_data("AAPL")

    # Step 2: Features
    df = add_features(df)

    # Step 3: Prepare
    X, y = prepare_data(df)

    # Step 4: Models
    lr = train_linear_regression(X, y)
    rf = train_random_forest(X, y)
    xgb_model = train_xgboost(X, y)

    # Step 5: Evaluation
    print("Baseline MSE:", baseline_model(df))

    print("\nLinear Regression:", evaluate_model(lr, X, y))
    print("\nRandom Forest:", evaluate_model(rf, X, y))
    print("\nXGBoost:", evaluate_model(xgb_model, X, y))


if __name__ == "__main__":
    run_pipeline()