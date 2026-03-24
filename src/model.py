from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1, max_features='sqrt')
    # Adjusting hyperparameters for better performance (min_samples_leaf=1, max_features='sqrt')
    model.fit(X, y)
    return model


def train_xgboost(X, y):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    return model